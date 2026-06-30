import os
import re
import json
import shutil
import requests
import time
import argparse
import configparser
import logging
from datetime import datetime
try:
    from langchain.chains import LLMChain
except ImportError:
    # LangChain 1.x doesn't have chains module
    LLMChain = None

try:
    from langchain.prompts import PromptTemplate
except ImportError:
    # LangChain 1.x moved prompts to langchain_core
    from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI as NewChatOpenAI
try:
    from langchain_community.chat_models import ChatOllama
except Exception:
    ChatOllama = None  # type: ignore
from dotenv import load_dotenv, dotenv_values
from typing import List, Dict, Any, Optional
from openai import OpenAI
from github import Github, Auth
from translation_state import TranslationStateManager
from file_diff_analyzer import FileDiffAnalyzer

# Matches every mask placeholder produced by _mask_non_translatable
# (e.g. __HTML_OPEN_3__, __GITBOOK_TAG_0__, __CODE_BLOCK_1__, __LINK_URL_2__).
PLACEHOLDER_RE = re.compile(r'__[A-Z_]+\d+__')
# Detects Japanese text that must be translated (Hiragana, Katakana, CJK Kanji).
JP_CHAR_RE = re.compile('[぀-ヿ一-鿿]')

# Load environment variables
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(ENV_PATH)

class TranslationAgent:
    def __init__(self, github_repo: str = None, branch: str = None, target_language: str = None,
                 target_languages: List[str] = None, target_paths: str = None,
                 dictionary_file: str = "./dictionary.txt", config_file: str = "./config.ini",
                 output_naming: str = None, push_option: str = None):
        # Read parameters from config file first
        # inline_comment_prefixes strips trailing "# ..." comments from values so
        # that e.g. an API key written as "key  # note" yields just "key" rather
        # than the key plus a non-ASCII comment that breaks HTTP headers.
        self.config = configparser.ConfigParser(inline_comment_prefixes=('#',))
        # Read config file in UTF-8 for compatibility
        self.config.read(config_file, encoding='utf-8')
        # Read model-related env vars only from .env (do not use OS env vars)
        self._env = dotenv_values(ENV_PATH)
        
        # Set defaults and read values from config
        self.github_repo = github_repo or (self.config.get("github", "repo", fallback=None) if self.config.has_section("github") else None)
        
        # Multiple languages support
        if target_languages:
            self.target_languages = target_languages
        elif target_language:
            self.target_languages = [target_language]
        else:
            config_langs = self.config.get("translation", "target_languages", fallback=None) if self.config.has_section("translation") else None
            if config_langs:
                self.target_languages = [lang.strip() for lang in config_langs.split(',')]
            else:
                default_lang = self.config.get("translation", "target_language", fallback="zh_cn") if self.config.has_section("translation") else "zh_cn"
                self.target_languages = [default_lang]

        # Normalize target languages to internal canonical forms
        self.target_languages = [self._normalize_target_language_internal(lang) for lang in self.target_languages]
        
        # Set current target_language to first language (for backward compatibility)
        self.target_language = self.target_languages[0] if self.target_languages else "zh_cn"
        
        # Glob pattern for target files
        self.target_paths = target_paths or (self.config.get("github", "target_paths", fallback="**/*.md") if self.config.has_section("github") else "**/*.md")
        
        # Output naming convention: suffix or directory
        self.output_naming = output_naming or (self.config.get("output", "naming", fallback="suffix") if self.config.has_section("output") else "suffix")
        
        # Push option: none or push_same_repo_direct
        self.push_option = push_option or (self.config.get("github", "push_option", fallback="none") if self.config.has_section("github") else "none")
        
        self.dictionary_file = dictionary_file or (self.config.get("translation", "dictionary_file", fallback="./dictionary.txt") if self.config.has_section("translation") else "./dictionary.txt")
        self.temp_dir = self.config.get("paths", "temp_dir", fallback="./temp_repo") if self.config.has_section("paths") else "./temp_repo"
        self.download_dir = self.config.get("paths", "download_dir", fallback="./downloaded_files") if self.config.has_section("paths") else "./downloaded_files"
        self.translated_dir = self.config.get("paths", "translated_dir", fallback="./translated_results") if self.config.has_section("paths") else "./translated_results"
        # Process branch configuration (supports comma-separated list)
        branch_str = branch or (self.config.get("github", "branch", fallback=None) if self.config.has_section("github") else None)
        if branch_str:
            self.branches = [b.strip() for b in branch_str.split(',')]
        else:
            self.branches = []
        
        # Read API keys from .env or config (prefer .env; do not use OS env vars)
        self.github_token = self._env.get("GITHUB_TOKEN") or (self.config.get("api", "github_token", fallback=None) if self.config.has_section("api") else None)
        self.openai_api_key = self._env.get("OPENAI_API_KEY") or (self.config.get("api", "openai_api_key", fallback=None) if self.config.has_section("api") else None)
        self.dashscope_api_key = self._env.get("DASHSCOPE_API_KEY") or (self.config.get("api", "dashscope_api_key", fallback=None) if self.config.has_section("api") else None)
        self.deepseek_api_key = self._env.get("DEEPSEEK_API_KEY") or (self.config.get("api", "deepseek_api_key", fallback=None) if self.config.has_section("api") else None)
        self.claude_api_key = self._env.get("CLAUDE_API_KEY") or (self.config.get("api", "claude_api_key", fallback=None) if self.config.has_section("api") else None)
        # Google Gemini API key (used for English review)
        self.google_api_key = self._env.get("GOOGLE_API_KEY") or (self.config.get("api", "google_api_key", fallback=None) if self.config.has_section("api") else None)
        # Agnes API key (OpenAI-compatible endpoint)
        self.agnes_api_key = self._env.get("AGNES_API_KEY") or (self.config.get("api", "agnes_api_key", fallback=None) if self.config.has_section("api") else None)
        self.azure_openai_endpoint = self._env.get("AZURE_OPENAI_ENDPOINT") or (self.config.get("api", "azure_openai_endpoint", fallback=None) if self.config.has_section("api") else None)
        self.azure_openai_api_key = self._env.get("AZURE_OPENAI_API_KEY") or (self.config.get("api", "azure_openai_api_key", fallback=None) if self.config.has_section("api") else None)

        # Normalize target language label used by dictionary and prompts
        self.target_label = self._normalize_target_label(self.target_language)
        
        # Ensure temp_dir exists
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)
            print(f"Created temp directory: {self.temp_dir}")
        
        # Initialize logger
        self._setup_logger()
        
        # Initialize translation state manager and file diff analyzer
        self.state_manager = TranslationStateManager(os.path.join(self.temp_dir, "translation_state.json"))
        self.diff_analyzer = FileDiffAnalyzer()
        
        # Other initialization
        self.dictionary = self._load_dictionary()
        self.github = None
        self.repo = None
        self.branch_sha = None
        self.github_username = None
        self.github_repo_name = None
        self.full_repo_name = None
        self.local_to_repo_path: Dict[str, str] = {}
        self.local_to_branch: Dict[str, str] = {}
        if self.github_repo:
            self._parse_github_repo()

    def _setup_logger(self):
        """Initialize logger with file and console handlers"""
        # Create logger
        self.logger = logging.getLogger('TranslationAgent')
        self.logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers
        if self.logger.handlers:
            return
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler (detailed, all levels)
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TranslationAgent.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler (simpler, INFO and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Logger initialized. Log file: {log_file}")
    
    def _normalize_target_label(self, tl: str) -> str:
        """Normalize target language label to zh-CN / zh-TW / en / th"""
        if not tl:
            return "zh-CN"
        lower = tl.lower()
        if lower in ("zh_cn", "zh-cn", "cn", "zh"):
            return "zh-CN"
        if lower in ("zh_tw", "zh-tw", "tc", "tw"):
            return "zh-TW"
        if lower in ("th", "thai"):
            return "th"
        return "en"

    def _normalize_target_language_internal(self, tl: str) -> str:
        """
        Normalize CLI/config target language to internal canonical forms.
        - zh_cn: simplified Chinese
        - zh_tw: traditional Chinese (Taiwan)
        - en: English
        - th: Thai
        """
        if not tl:
            return "zh_cn"
        lower = str(tl).strip().lower()
        if lower in ("zh_cn", "zh-cn", "cn", "zh"):
            return "zh_cn"
        if lower in ("zh_tw", "zh-tw", "tc", "tw"):
            return "zh_tw"
        if lower in ("th", "thai"):
            return "th"
        if lower in ("en", "english"):
            return "en"
        if lower in ("zh-cn", "zh_tw", "zh_cn", "zh-tw"):
            return lower.replace("-", "_")
        return "en"

    def _select_ollama_model(self, failed_model_name: Optional[str], task_type: str) -> str:
        """Map failed upstream model names to Ollama model names."""
        model = (failed_model_name or "").lower()
        # Required mappings (plus a couple of common variants).
        if "gpt-5.1" in model:
            return "gpt-oss:120b"
        if "gpt-3.5-turbo" in model or "gpt-35" in model:
            return "gpt-oss:20b"
        if "qwen3-max" in model or model == "qwen-max" or "qwen-max" in model:
            return "qwen3:30b"
        if "deepseek-chat" in model or "deepseek-r1" in model:
            return "deepseek-r1:14b"
        # Default: safe mid-sized model for "possible" Ollama fallback.
        return "gpt-oss:20b"

    def _ollama_invoke_text(self, prompt_text: str, failed_model_name: Optional[str], task_type: str) -> str:
        """Invoke Ollama via LangChain with model auto-selection."""
        if ChatOllama is None:
            raise RuntimeError("ChatOllama is not available (langchain-ollama not installed).")
        ollama_model = self._select_ollama_model(failed_model_name, task_type)
        self.logger.warning(f"Using Ollama fallback for {task_type} (model: {ollama_model})")
        ollama_llm = self._build_ollama_llm(ollama_model, task_type)
        result = ollama_llm.invoke(prompt_text)
        if hasattr(result, "content"):
            return result.content
        return str(result)

    def _parse_github_repo(self):
        """Parse GitHub repository URL and extract owner/repo"""
        # Support multiple GitHub URL formats
        patterns = [
            r'https://github.com/([^/]+)/([^/]+)(?:\.git)?',
            r'git@github.com:([^/]+)/([^/]+)(?:\.git)?',
            r'([^/]+)/([^/]+)'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, self.github_repo)
            if match:
                self.github_username, self.github_repo_name = match.groups()
                break
        
        if not self.github_username or not self.github_repo_name:
            raise ValueError(f"Failed to parse GitHub repository URL: {self.github_repo}")
        
        # Build full repository name
        self.full_repo_name = f"{self.github_username}/{self.github_repo_name}"
        
    def connect_to_github_repo(self):
        """Connect to GitHub repository"""
        try:
            # Use recommended authentication
            auth = Auth.Token(self.github_token)
            self.github = Github(auth=auth)
            self.repo = self.github.get_repo(self.full_repo_name)
            
            # Store mapping from branch name to SHA
            self.branch_shas = {}
            
            # Fetch specified branches
            if self.branches:
                for branch_name in self.branches:
                    try:
                        branch = self.repo.get_branch(branch_name)
                        self.branch_shas[branch_name] = branch.commit.sha
                    except Exception as e:
                        print(f"Failed to fetch branch {branch_name}: {e}")
            
            # If none specified or all failed, try default branches
            if not self.branch_shas:
                try:
                    branch = self.repo.get_branch("master")
                    self.branch_shas[branch.name] = branch.commit.sha
                except:
                    try:
                        branch = self.repo.get_branch("main")
                        self.branch_shas[branch.name] = branch.commit.sha
                    except Exception as e:
                        print(f"Failed to fetch default branch: {e}")
                        return False
            
            print(f"Connected to repository: {self.full_repo_name}")
            print(f"Fetched branches: {', '.join(self.branch_shas.keys())}")
            print(f"Repository name: {self.repo.name}")
            print(f"Stars: {self.repo.stargazers_count}")
            if self.repo.description:
                print(f"Description: {self.repo.description}")
            if self.repo.language:
                print(f"Primary language: {self.repo.language}")
            
            return True
        except Exception as e:
            print(f"Failed to connect to GitHub repository: {e}")
            return False
    
    def find_md_files_in_repo(self, path=".", branch_name=None, branch_sha=None):
        """Recursively find all MD files in the specified branch/repo"""
        if not self.repo:
            print("Error: not connected to GitHub repository")
            return []
        
        # If branch SHA not provided, use the first available one
        if not branch_sha and self.branch_shas:
            branch_sha = next(iter(self.branch_shas.values()))
            branch_name = next(iter(self.branch_shas.keys()))
        
        md_files = []
        try:
            contents = self.repo.get_contents(path, ref=branch_sha)
            
            for content in contents:
                if content.type == "file" and content.name.endswith(".md"):
                    md_files.append({
                        "name": content.name,
                        "path": content.path,
                        "url": content.download_url,
                        "size": content.size,
                        "sha": content.sha,
                        "branch": branch_name
                    })
                    print(f"Found MD file: {content.path} (branch: {branch_name})")
                elif content.type == "dir":
                    # Recursively scan subdirectory
                    print(f"Scanning subdirectory: {content.path} (branch: {branch_name})")
                    md_files.extend(self.find_md_files_in_repo(content.path, branch_name, branch_sha))
            
            return md_files
        except Exception as e:
            print(f"Error finding MD files ({path}, branch: {branch_name}): {e}")
            return []
    
    def download_file(self, file_info, overwrite=False):
        """Download specified file and organize directories per branch"""
        # Get the branch the file belongs to
        branch_name = file_info.get("branch", "main")
        
        # Create separate download directory for each branch
        branch_download_dir = os.path.join(self.download_dir, branch_name)
        
        if not os.path.exists(branch_download_dir):
            os.makedirs(branch_download_dir)
            print(f"Created branch download directory: {branch_download_dir}")
        
        # Ensure the file path directory exists
        file_path = os.path.join(branch_download_dir, file_info["path"])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Check if file already exists
        if os.path.exists(file_path) and not overwrite:
            print(f"Skip download: {file_info['path']} (branch: {branch_name}) (already exists)")
            return file_path
        
        try:
            # Add headers to avoid API limits
            headers = {
                "Authorization": f"token {self.github_token}"
            }
            
            print(f"Start download: {file_info['path']}")
            # Request to download the file
            response = requests.get(file_info["url"], headers=headers, stream=True)
            response.raise_for_status()
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded_size = 0
            
            # Save file
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        # Show download progress
                        if total_size > 0:
                            progress = downloaded_size / total_size * 100
                            print(f"Download progress: {progress:.1f}% ({downloaded_size/1024:.1f} KB / {total_size/1024:.1f} KB)", end="\r")
            print()  # newline
            
            # Format file size
            size_mb = file_info["size"] / (1024 * 1024)
            print(f"Download finished: {file_info['path']} ({size_mb:.2f} MB)")
            
            # Add delay to avoid hitting GitHub API rate limits
            time.sleep(1)
            return file_path
        except Exception as e:
            print(f"Failed to download file ({file_info['path']}): {e}")
            # If download failed and file exists, delete incomplete file
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted incomplete file: {file_path}")
                except:
                    pass
            return None
    
    def download_md_files(self, overwrite=False) -> List[str]:
        """Download all found MD files and organize by branch"""
        md_files = []
        try:
            # Parse owner and repo from URL
            if not self.github_username or not self.github_repo_name:
                self._parse_github_repo()
            
            # Connect to GitHub repository
            if not self.connect_to_github_repo():
                return []
            
            # Find all MD files in repository (by branch)
            print("\nSearching MD files for all branches...")
            md_file_infos = []
            branch_stats = {}
            
            # Find files for each branch
            if self.branch_shas:
                for branch_name, branch_sha in self.branch_shas.items():
                    print(f"Finding MD files in branch {branch_name}...")
                    branch_files = self.find_md_files_in_repo(".", branch_name, branch_sha)
                    md_file_infos.extend(branch_files)
                    branch_stats[branch_name] = len(branch_files)
                    print(f"  Branch {branch_name}: found {len(branch_files)} MD files")
            else:
                # If no branch info, use default search method
                print("Using default method to find MD files")
                md_file_infos = self.find_md_files_in_repo()
            
            if not md_file_infos:
                print("No MD files found")
                return []
            
            print(f"\nTotal found {len(md_file_infos)} MD files, starting download...")
            
            # Sum file sizes
            total_size = sum(f["size"] for f in md_file_infos)
            total_size_mb = total_size / (1024 * 1024)
            print(f"Total file size: {total_size_mb:.2f} MB")
            
            # Create map: file path -> SHA
            file_sha_map = {}
            
            success_count = 0
            for i, file_info in enumerate(md_file_infos, 1):
                branch_name = file_info.get("branch", "unknown")
                print(f"\nDownloading file {i}/{len(md_file_infos)} (branch: {branch_name}):")
                print(f"Name: {file_info['name']}")
                print(f"Path: {file_info['path']}")
                print(f"SHA: {file_info['sha']}")
                
                file_path = self.download_file(file_info, overwrite)
                if file_path:
                    md_files.append(file_path)
                    # Save file path and mapping of SHA
                    file_sha_map[file_path] = file_info['sha']
                    # Map local download path to repo relative path and branch
                    self.local_to_repo_path[file_path] = file_info['path']
                    self.local_to_branch[file_path] = branch_name
                    success_count += 1
            
            # Save file SHA map for later use
            self.file_sha_map = file_sha_map
            
            print(f"\nDownload complete! Successfully downloaded {success_count}/{len(md_file_infos)} files")
            # Print per-branch stats
            print("Branch download statistics:")
            branch_downloaded = {}
            for branch_name in branch_stats:
                branch_downloaded[branch_name] = 0
            
            for f in md_file_infos:
                branch_name = f.get("branch", "unknown")
                # Check whether file was downloaded successfully
                file_path = os.path.join(self.download_dir, branch_name, f["path"]) 
                if os.path.exists(file_path):
                    branch_downloaded[branch_name] += 1
            
            for branch_name, count in branch_downloaded.items():
                total = branch_stats.get(branch_name, 0)
                print(f"  Branch {branch_name}: downloaded {count}/{total} files")
            print(f"Downloaded files saved in: {os.path.abspath(self.download_dir)}")
            
            return md_files
        except Exception as e:
            print(f"An error occurred when downloading files directly from GitHub: {e}")
            return []



    def _load_dictionary(self) -> Dict[str, str]:
        """Load terminology dictionary, preferring dictionary.json
        Return format: {target_lang: {jp_term: translated_term}}
        """
        dictionary: Dict[str, Dict[str, str]] = {}
        try:
            base_no_ext = os.path.splitext(self.dictionary_file)[0]
            # Map target_label to actual file suffix (handle both underscore and hyphen formats)
            lang_suffix_map = {
                "en": "_en.json",
                "zh-CN": "_zh-cn.json",  # Changed from _zh.json
                "zh-TW": "_zh-tw.json",  # Changed from _zh-TW.json
                "th": "_th.json"
            }
            lang_suffix = lang_suffix_map.get(self.target_label, ".json")
            lang_json_path = base_no_ext + lang_suffix
            json_path = base_no_ext + ".json"

            if os.path.exists(lang_json_path):
                with open(lang_json_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                    if isinstance(data, dict):
                        dictionary[self.target_label] = {}
                        for k, v in data.items():
                            if isinstance(v, str):
                                dictionary[self.target_label][k] = v
                    print(f"Loaded language-specific JSON dictionary: {lang_json_path}")
            elif os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                    
                    # Supported array format: [{"Key": "...", "Japanese": "...", "English": "...", "Chinese": "...", "Chinese_Traditional": "...", "Thai": "..."}]
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                jp_term = item.get("Japanese", "")
                                en_term = item.get("English", "")
                                zh_cn_term = item.get("Chinese", "")
                                zh_tw_term = item.get("Chinese_Traditional", "")
                                th_term = item.get("Thai", "")
                                
                                if jp_term:
                                    # Initialize per-language dictionaries
                                    if "en" not in dictionary:
                                        dictionary["en"] = {}
                                    if "zh-CN" not in dictionary:
                                        dictionary["zh-CN"] = {}
                                    if "zh-TW" not in dictionary:
                                        dictionary["zh-TW"] = {}
                                    if "th" not in dictionary:
                                        dictionary["th"] = {}
                                    
                                    # Add terms (if target language has corresponding value)
                                    if en_term:
                                        dictionary["en"][jp_term] = en_term
                                    if zh_cn_term:
                                        dictionary["zh-CN"][jp_term] = zh_cn_term
                                    if zh_tw_term:
                                        dictionary["zh-TW"][jp_term] = zh_tw_term
                                    if th_term:
                                        dictionary["th"][jp_term] = th_term
                    
                    # Supported dict formats:
                    # 1) { jaTerm: targetTerm }
                    # 2) { jaTerm: { "zh-CN": "...", "zh-TW": "...", "en": "...", "th": "..." } }
                    elif isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, dict):
                                # Initialize per-language dictionaries
                                if "en" not in dictionary:
                                    dictionary["en"] = {}
                                if "zh-CN" not in dictionary:
                                    dictionary["zh-CN"] = {}
                                if "zh-TW" not in dictionary:
                                    dictionary["zh-TW"] = {}
                                if "th" not in dictionary:
                                    dictionary["th"] = {}
                                
                                # Add terms for each language
                                if "en" in v or "English" in v:
                                    dictionary["en"][k] = v.get("en") or v.get("English", "")
                                if "zh-CN" in v or "Chinese" in v:
                                    dictionary["zh-CN"][k] = v.get("zh-CN") or v.get("Chinese", "")
                                if "zh-TW" in v or "Chinese_Traditional" in v:
                                    dictionary["zh-TW"][k] = v.get("zh-TW") or v.get("Chinese_Traditional", "")
                                if "th" in v or "Thai" in v:
                                    dictionary["th"][k] = v.get("th") or v.get("Thai", "")
                            elif isinstance(v, str):
                                # Single target-language case: use current target language
                                if self.target_label not in dictionary:
                                    dictionary[self.target_label] = {}
                                dictionary[self.target_label][k] = v
                
                print(f"Loaded JSON dictionary: {json_path}")
            elif os.path.exists(self.dictionary_file):
                # TXT dictionary compatibility: key=value per line
                if self.target_label not in dictionary:
                    dictionary[self.target_label] = {}
                with open(self.dictionary_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            dictionary[self.target_label][key.strip()] = value.strip()
                print(f"Loaded TXT dictionary: {self.dictionary_file}")
            else:
                print(f"Warning: dictionary file {self.dictionary_file} or {json_path} not found")
        except Exception as e:
            print(f"Error loading dictionary: {e}")
        
        # Save full multi-language dictionary for later use
        self.full_dictionary = dictionary
        
        # Return the dictionary for current target language (backward compatible)
        return dictionary.get(self.target_label, {})
    
    def _resolve_provider_and_model(self, task_type: str = "translate"):
        """Resolve provider and model from config.ini [models] per task and language.

        Returns (provider, model). provider is lowercased; both may be empty
        strings when nothing is configured.
        """
        kind = "review" if task_type == "review" else "translate"
        lang = (self.target_language or "").lower()
        if lang in ("zh_cn", "zh-cn"):
            suffix = "zh_cn"
        elif lang in ("zh_tw", "zh-tw"):
            suffix = "zh_tw"
        elif lang == "en":
            suffix = "en"
        else:
            suffix = None

        def cfg(key):
            if self.config.has_section("models"):
                return self.config.get("models", key, fallback=None)
            return None

        provider = cfg(f"{kind}_provider_{suffix}") if suffix else None
        model = cfg(f"{kind}_model_{suffix}") if suffix else None
        if not provider:
            provider = cfg(f"{kind}_provider_default")
        if not model:
            model = cfg(f"{kind}_model_default")
        return (provider or "").strip().lower(), (model or "").strip()

    def _provider_ready(self, provider: str) -> bool:
        """Whether the given provider can be used (local server or API key present)."""
        provider = (provider or "").lower()
        if provider == "ollama":
            return True
        if provider == "dashscope":
            return bool(self.dashscope_api_key)
        if provider == "deepseek":
            return bool(self.deepseek_api_key)
        if provider == "gemini":
            return bool(self.google_api_key)
        if provider == "claude":
            return bool(self.claude_api_key)
        if provider == "agnes":
            return bool(self.agnes_api_key)
        # openai or unconfigured -> legacy path relies on OpenAI/Azure
        return bool(self.openai_api_key) or bool(self.azure_openai_endpoint and self.azure_openai_api_key)

    def _build_ollama_llm(self, model: str, task_type: str = "translate"):
        """Construct a ChatOllama with a request timeout and generation caps.

        Without a timeout the streaming read can block forever when a small local
        model enters an endless generation loop or when the Ollama server stalls.
        num_predict caps output length; num_ctx widens the context window so long
        prompts (with many GitBook/HTML placeholders) are not silently truncated.
        All three are tunable via the optional [ollama] config section.
        """
        if ChatOllama is None:
            raise RuntimeError("ChatOllama is not available (langchain-community/ollama not installed).")
        base_url = self._env.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        def _cfg_int(key, default):
            val = self.config.get("ollama", key, fallback=None) if self.config.has_section("ollama") else None
            try:
                return int(val) if val not in (None, "") else default
            except (TypeError, ValueError):
                return default

        return ChatOllama(
            model=model,
            temperature=0,
            base_url=base_url,
            timeout=_cfg_int("timeout", 300),       # read timeout (seconds) per request
            num_predict=_cfg_int("num_predict", 4096),  # cap output tokens to avoid infinite loops
            num_ctx=_cfg_int("num_ctx", 8192),      # context window large enough for masked prompts
        )

    def _get_llm_client(self, task_type: str = "translate"):
        """Get LLM client based on config.ini [models] provider/model settings.

        Honors per-language provider/model config. Returns either a LangChain LLM
        (with .invoke) or a tuple (native_client, model_name) for OpenAI-compatible
        clients. Falls back to the legacy hard-coded per-language behavior only when
        no provider is configured.
        """
        provider, model_name = self._resolve_provider_and_model(task_type)
        try:
            if provider == "ollama":
                model = model_name or "gpt-oss:20b"
                print(f"Using Ollama ({model}) for {task_type}")
                return self._build_ollama_llm(model, task_type)

            if provider == "openai":
                if self.azure_openai_endpoint and self.azure_openai_api_key:
                    client = OpenAI(api_key=self.azure_openai_api_key, base_url=self.azure_openai_endpoint)
                    print("Using Azure OpenAI")
                    return client, (model_name or "gpt-4o")
                return NewChatOpenAI(model_name=model_name or "gpt-4o", temperature=0, api_key=self.openai_api_key)

            if provider == "dashscope":
                if not self.dashscope_api_key:
                    raise ValueError("DashScope API key not set")
                client = OpenAI(api_key=self.dashscope_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
                print("Using DashScope (Qwen)")
                return client, (model_name or "qwen-max")

            if provider == "deepseek":
                if not self.deepseek_api_key:
                    raise ValueError("DeepSeek API key not set")
                client = OpenAI(api_key=self.deepseek_api_key, base_url="https://api.deepseek.com/v1")
                print("Using DeepSeek")
                return client, (model_name or "deepseek-chat")

            if provider == "gemini":
                if not self.google_api_key:
                    raise ValueError("Google API key not set")
                client = OpenAI(api_key=self.google_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
                print("Using Gemini")
                return client, (model_name or "gemini-2.5-pro")

            if provider == "agnes":
                if not self.agnes_api_key:
                    raise ValueError("Agnes API key not set")
                client = OpenAI(api_key=self.agnes_api_key, base_url="https://apihub.agnes-ai.com/v1")
                print("Using Agnes")
                return client, (model_name or "agnes-2.0-flash")

            # provider == "claude" or unknown/empty -> legacy hard-coded behavior
            return self._get_llm_client_legacy(task_type)
        except Exception as e:
            print(f"Error initializing model (provider={provider}): {e}")
            # Do not silently fall back to OpenAI when another provider was requested.
            raise

    def _get_llm_client_legacy(self, task_type: str = "translate"):
        """Legacy hard-coded per-language client selection.

        Used only when no provider is configured in [models]."""
        try:
            if task_type in ("translate", "partial_translation"):
                if self.target_language.lower() == "en":
                    if self.azure_openai_endpoint and self.azure_openai_api_key:
                        client = OpenAI(
                            api_key=self.azure_openai_api_key,
                            base_url=self.azure_openai_endpoint
                        )
                        print("Using Azure OpenAI for English translation")
                        return client, "gpt-4o"
                    return NewChatOpenAI(model_name="gpt-4o", temperature=0, api_key=self.openai_api_key)
                elif self.target_language.lower() == "zh_cn" or self.target_language.lower() == "zh_tw":
                    # Translate to Chinese using Qwen-Max
                    if self.dashscope_api_key:
                        client = OpenAI(
                            api_key=self.dashscope_api_key,
                            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                        )
                        print(f"Using Qwen-Max for translation")
                        return client, "qwen-max"
                    else:
                        raise ValueError("DashScope API key not set")
                elif self.target_language.lower() in ("th", "thai"):
                    return NewChatOpenAI(model_name="gpt-4o", temperature=0, api_key=self.openai_api_key)
                else:
                    return NewChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=self.openai_api_key)
            # Review task
            elif task_type == "review":
                if self.target_language.lower() == "en":
                    # English review using Gemini 2.5 Pro (OpenAI compatible endpoint)
                    if self.google_api_key:
                        client = OpenAI(
                            api_key=self.google_api_key,
                            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                        )
                        print(f"Using Gemini 2.5 Pro for review")
                        return client, "gemini-2.5-pro"
                    else:
                        raise ValueError("Google API key not set")
                elif self.target_language.lower() == "zh_cn" or self.target_language.lower() == "zh_tw":
                    # Chinese review using DeepSeek-V3
                    if self.deepseek_api_key:
                        client = OpenAI(
                            api_key=self.deepseek_api_key,
                            base_url="https://api.deepseek.com/v1"
                        )
                        print(f"Using DeepSeek-V3 for review")
                        return client, "deepseek-chat"
                    else:
                        raise ValueError("DeepSeek API key not set")
                elif self.target_language.lower() in ("th", "thai"):
                    return NewChatOpenAI(model_name="gpt-4o", temperature=0, api_key=self.openai_api_key)
                else:
                    return NewChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=self.openai_api_key)
            else:
                # Other tasks use default model
                return NewChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=self.openai_api_key)
        except Exception as e:
            print(f"Error initializing model: {e}")
            print("Fallback to gpt-3.5-turbo")
            return NewChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=self.openai_api_key)
    
    def find_md_files(self) -> List[str]:
        """Find all MD files in the download directory"""
        md_files = []
        for root, _, files in os.walk(self.download_dir):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))
        return md_files
    
    def _match_files_by_glob(self, files: List[Any], pattern: str) -> List[Any]:
        """Match files by glob pattern
        
        Args:
            files: List of file paths (str) or file info dicts
            pattern: Glob pattern (e.g., docs/**/*.md, README.md)
            
        Returns:
            List of matched files
        """
        import fnmatch

        def _glob_to_regex(pat: str):
            """Convert a glob pattern to a compiled regex.

            Unlike pathlib.Path.match, '**' matches zero or more directory
            segments, so '**/*.md' matches both top-level and nested files.
            """
            i, n, out = 0, len(pat), ''
            while i < n:
                if pat[i:i + 3] == '**/':
                    out += '(?:.*/)?'  # zero or more directories
                    i += 3
                elif pat[i:i + 2] == '**':
                    out += '.*'
                    i += 2
                elif pat[i] == '*':
                    out += '[^/]*'
                    i += 1
                elif pat[i] == '?':
                    out += '[^/]'
                    i += 1
                else:
                    out += re.escape(pat[i])
                    i += 1
            return re.compile('^' + out + '$')

        matched = []
        patterns = [p.strip() for p in pattern.split(',')]
        
        for file_item in files:
            # Handle both string paths and dict file info
            if isinstance(file_item, dict):
                file_path = file_item.get("path", "")
            else:
                file_path = file_item
            
            # Extract relative path for matching (remove download_dir prefix if present)
            if self.download_dir and file_path.startswith(self.download_dir):
                # Get path relative to download_dir
                rel_path = os.path.relpath(file_path, self.download_dir)
                # Remove branch name from path (first component)
                path_parts = rel_path.split(os.sep)
                if len(path_parts) > 1:
                    match_path = os.sep.join(path_parts[1:])  # Skip branch name
                else:
                    match_path = rel_path
            else:
                match_path = file_path

            # Normalize separators to '/' so patterns match on Windows too
            match_path = match_path.replace(os.sep, '/')

            for pat in patterns:
                pat = pat.replace(os.sep, '/')
                # Handle ** patterns (zero or more directories); fall back to
                # fnmatch for simple patterns
                if '**' in pat:
                    if _glob_to_regex(pat).match(match_path):
                        matched.append(file_item)
                        break
                else:
                    if fnmatch.fnmatch(match_path, pat):
                        matched.append(file_item)
                        break
        
        return matched
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 5000) -> List[str]:
        """Split text into translatable chunks
        
        Args:
            text: text to split
            max_chunk_size: max characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Compute end position for current chunk
            end_pos = current_pos + max_chunk_size
            
            if end_pos >= len(text):
                # Last chunk
                chunks.append(text[current_pos:])
                break
            
            # Try to split at suitable positions, priority: headings > paragraphs > sentences
            chunk_end = end_pos
            
            # Find nearest split point from current position forward
            search_start = max(current_pos, end_pos - 500)  # search within last 500 chars
            remaining_text = text[search_start:end_pos + 500]
            
            # 1) Prefer splitting at Markdown headings
            heading_pattern = r'\n#{1,6}\s+'
            heading_matches = list(re.finditer(heading_pattern, remaining_text))
            if heading_matches:
                # Find heading closest to end_pos
                for match in reversed(heading_matches):
                    potential_end = search_start + match.start()
                    if current_pos < potential_end <= end_pos:
                        chunk_end = potential_end
                        break
            
            # 2) If no heading, split at paragraph boundary (double newline)
            if chunk_end == end_pos:
                paragraph_pattern = r'\n\n+'
                paragraph_matches = list(re.finditer(paragraph_pattern, remaining_text))
                if paragraph_matches:
                    for match in reversed(paragraph_matches):
                        potential_end = search_start + match.start()
                        if current_pos < potential_end <= end_pos:
                            chunk_end = potential_end
                            break
            
            # 3) If still not found, split at sentence boundary
            if chunk_end == end_pos:
                sentence_pattern = r'[。！？\n]'
                sentence_matches = list(re.finditer(sentence_pattern, remaining_text))
                if sentence_matches:
                    for match in reversed(sentence_matches):
                        potential_end = search_start + match.start() + 1  # include punctuation
                        if current_pos < potential_end <= end_pos:
                            chunk_end = potential_end
                            break
            
            # 4) Otherwise, split at max_chunk_size
            if chunk_end == end_pos or chunk_end <= current_pos:
                chunk_end = end_pos
            
            # Add current chunk
            chunks.append(text[current_pos:chunk_end])
            current_pos = chunk_end
            
            # Skip leading whitespace
            while current_pos < len(text) and text[current_pos] in ' \n\t':
                current_pos += 1
        
        return chunks
    
    def _mask_gitbook_syntax(self, text: str, placeholders: Dict[str, Dict[str, str]]) -> str:
        """Mask GitBook-specific syntax
        
        Args:
            text: Text to process
            placeholders: Placeholder dictionary
            
        Returns:
            Masked text
        """
        masked = text
        
        # 1. YAML frontmatter (highest priority, ONLY at document start)
        # Anchor to the very beginning (no re.MULTILINE) so that horizontal-rule
        # separators (---) in the body are not mistaken for frontmatter.
        yaml_frontmatter_pattern = re.compile(r'\A---\s*\n(.*?)\n---\s*\n', re.DOTALL)
        def repl_yaml(m):
            idx = len(placeholders["yaml_frontmatter"])
            ph = f"__YAML_FRONTMATTER_{idx}__"
            placeholders["yaml_frontmatter"][ph] = m.group(0)
            return ph
        masked = yaml_frontmatter_pattern.sub(repl_yaml, masked)
        
        # 2. GitBook hint blocks - mask only opening and closing tags, keep content translatable
        # Opening tag: {% hint style="..." %}
        hint_open_pattern = re.compile(r'{%\s*hint\s+style="[^"]*"\s*%}')
        def repl_hint_open(m):
            idx = len(placeholders["gitbook_hint_blocks"])
            ph = f"__GITBOOK_HINT_OPEN_{idx}__"
            placeholders["gitbook_hint_blocks"][ph] = m.group(0)
            return ph
        masked = hint_open_pattern.sub(repl_hint_open, masked)
        
        # Closing tag: {% endhint %}
        hint_close_pattern = re.compile(r'{%\s*endhint\s*%}')
        def repl_hint_close(m):
            idx = len(placeholders["gitbook_hint_blocks"])
            ph = f"__GITBOOK_HINT_CLOSE_{idx}__"
            placeholders["gitbook_hint_blocks"][ph] = m.group(0)
            return ph
        masked = hint_close_pattern.sub(repl_hint_close, masked)
        
        # 3. GitBook tabs blocks - mask only opening and closing tags, keep content translatable
        # Opening tag: {% tabs %}
        tabs_open_pattern = re.compile(r'{%\s*tabs\s*%}')
        def repl_tabs_open(m):
            idx = len(placeholders["gitbook_tabs_blocks"])
            ph = f"__GITBOOK_TABS_OPEN_{idx}__"
            placeholders["gitbook_tabs_blocks"][ph] = m.group(0)
            return ph
        masked = tabs_open_pattern.sub(repl_tabs_open, masked)
        
        # Closing tag: {% endtabs %}
        tabs_close_pattern = re.compile(r'{%\s*endtabs\s*%}')
        def repl_tabs_close(m):
            idx = len(placeholders["gitbook_tabs_blocks"])
            ph = f"__GITBOOK_TABS_CLOSE_{idx}__"
            placeholders["gitbook_tabs_blocks"][ph] = m.group(0)
            return ph
        masked = tabs_close_pattern.sub(repl_tabs_close, masked)
        
        # 4. GitBook tab tags - mask opening and closing tags separately
        # Opening tag: {% tab title="..." %}
        tab_open_pattern = re.compile(r'{%\s*tab\s+[^%]*%}')
        def repl_tab_open(m):
            idx = len(placeholders["gitbook_tags"])
            ph = f"__GITBOOK_TAB_OPEN_{idx}__"
            placeholders["gitbook_tags"][ph] = m.group(0)
            return ph
        masked = tab_open_pattern.sub(repl_tab_open, masked)
        
        # Closing tag: {% endtab %}
        tab_close_pattern = re.compile(r'{%\s*endtab\s*%}')
        def repl_tab_close(m):
            idx = len(placeholders["gitbook_tags"])
            ph = f"__GITBOOK_TAB_CLOSE_{idx}__"
            placeholders["gitbook_tags"][ph] = m.group(0)
            return ph
        masked = tab_close_pattern.sub(repl_tab_close, masked)
        
        # 5. GitBook single tags (include, embed, file) - these don't have content
        gitbook_single_tag_pattern = re.compile(r'{%\s*(?:include|embed|file)\s+[^%]*%}')
        def repl_gitbook_single_tag(m):
            idx = len(placeholders["gitbook_tags"])
            ph = f"__GITBOOK_TAG_{idx}__"
            placeholders["gitbook_tags"][ph] = m.group(0)
            return ph
        masked = gitbook_single_tag_pattern.sub(repl_gitbook_single_tag, masked)

        # 5b. Catch-all for any remaining GitBook tags ({% code %}, {% content-ref %},
        # {% endcode %}, {% stepper %}, etc.) so unhandled tags are never altered.
        gitbook_generic_tag_pattern = re.compile(r'{%[^%]*%}')
        def repl_gitbook_generic_tag(m):
            idx = len(placeholders["gitbook_tags"])
            ph = f"__GITBOOK_TAG_{idx}__"
            placeholders["gitbook_tags"][ph] = m.group(0)
            return ph
        masked = gitbook_generic_tag_pattern.sub(repl_gitbook_generic_tag, masked)
        
        # 6. Template expressions {{ ... }}
        template_expr_pattern = re.compile(r'{{[^}]+}}')
        def repl_template(m):
            idx = len(placeholders["template_expressions"])
            ph = f"__TEMPLATE_EXPR_{idx}__"
            placeholders["template_expressions"][ph] = m.group(0)
            return ph
        masked = template_expr_pattern.sub(repl_template, masked)
        
        # 7. HTML tags - mask only tags, keep content translatable
        # Note: Process in specific order to handle all tag types correctly
        
        # 7.1. Self-closing tags with explicit /: <tag />
        html_self_closing_pattern = re.compile(r'<[a-zA-Z][a-zA-Z0-9]*\b[^>]*/>')
        def repl_html_self_closing(m):
            idx = len(placeholders["html_tags"])
            ph = f"__HTML_TAG_{idx}__"
            placeholders["html_tags"][ph] = m.group(0)
            return ph
        masked = html_self_closing_pattern.sub(repl_html_self_closing, masked)
        
        # 7.2. Void elements (img, br, hr, input, etc.) - no closing tag needed
        # These should be masked completely as they don't contain translatable content
        html_void_pattern = re.compile(r'<(?:img|br|hr|input|meta|link|area|base|col|embed|source|track|wbr)\b[^>]*>')
        def repl_html_void(m):
            idx = len(placeholders["html_tags"])
            ph = f"__HTML_TAG_{idx}__"
            placeholders["html_tags"][ph] = m.group(0)
            return ph
        masked = html_void_pattern.sub(repl_html_void, masked)
        
        # 7.3. Closing tags: </tag>
        html_close_pattern = re.compile(r'</([a-zA-Z][a-zA-Z0-9]*)>')
        def repl_html_close(m):
            idx = len(placeholders["html_tags"])
            ph = f"__HTML_CLOSE_{idx}__"
            placeholders["html_tags"][ph] = m.group(0)
            return ph
        masked = html_close_pattern.sub(repl_html_close, masked)
        
        # 7.4. Opening tags: <tag attr="...">
        # Process after void elements to avoid double-masking
        html_open_pattern = re.compile(r'<([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>')
        def repl_html_open(m):
            idx = len(placeholders["html_tags"])
            ph = f"__HTML_OPEN_{idx}__"
            placeholders["html_tags"][ph] = m.group(0)
            return ph
        masked = html_open_pattern.sub(repl_html_open, masked)
        
        return masked
    
    def _translate_text(self, text: str) -> str:
        """Translate text segment-by-segment (manual-safe).

        Non-translatable content is masked first, then ONLY placeholder-free
        Japanese text is sent to the model, paragraph by paragraph, with the
        relevant terminology dictionary in the prompt. Mask tokens (GitBook tags,
        HTML, code, URLs, ...) are never sent to the model and are spliced back
        verbatim, so weak local models cannot hallucinate around them.
        """
        self.logger.info(f"Starting translation to {self.target_language}")
        self.logger.debug(f"Original text length: {len(text)} characters")

        # 1) Mask non-translatable content. The tokens only mark protected spans;
        #    they are excluded from what we send to the model.
        masked_text, placeholders = self._mask_non_translatable(text)
        self.logger.debug(f"Masked text length: {len(masked_text)} characters")
        self.logger.debug(f"Placeholders created: {sum(len(v) for v in placeholders.values())} items")

        # 2) Translate segment-by-segment (placeholder-free input only)
        try:
            translate_provider, _ = self._resolve_provider_and_model("translate")
            if not self._provider_ready(translate_provider):
                print(f"Warning: translation backend not ready (provider={translate_provider or 'unset'}); using original text as translation")
                translated_text = masked_text
            else:
                llm_result = self._get_llm_client("translate")
                if isinstance(llm_result, tuple):
                    llm, model_name = llm_result
                else:
                    llm, model_name = llm_result, None
                memo: Dict[str, str] = {}
                should_translate = lambda t: bool(JP_CHAR_RE.search(t))
                translate_core = lambda seg: self._invoke_paragraph_translation(seg, llm, model_name)
                translated_text = self._map_masked_segments(masked_text, should_translate, translate_core, memo)
        except Exception as e:
            self.logger.error(f"Error during translation: {e}", exc_info=True)
            print(f"Error during translation: {e}")
            translated_text = masked_text

        # 7) Remove extra tags possibly returned by model
        if '</think>' in translated_text:
            start_pos = translated_text.find('</think>')
            end_pos = translated_text.rfind('</think>') + len('</think>')
            if start_pos != -1 and end_pos != -1:
                translated_text = translated_text[:start_pos] + translated_text[end_pos:]

        # 7.5) Remove markdown code fence wrapping possibly added by the model
        if self.target_language.lower() == "en":
            translated_text = self._remove_markdown_code_fence(translated_text)

        # 8) Restore all masked placeholders
        translated_text = self._restore_placeholders(translated_text, placeholders)

        # 9) Enforce dictionary again after translation (only translatable regions)
        translated_text = self._enforce_dictionary_after_translation(translated_text)

        # 9.5) Validate Chinese translation quality (if target is Chinese)
        if self.target_label in ["zh-CN", "zh-TW"]:
            validation_issues = self._validate_chinese_translation(translated_text)
            if validation_issues:
                self.logger.warning(f"Translation quality issues detected: {len(validation_issues)} issues")
                print(f"⚠️ 翻訳品質の問題を検出: {len(validation_issues)}件")
                for issue in validation_issues:
                    self.logger.warning(f"  [{issue['severity']}] {issue['description']}: '{issue['pattern']}' → '{issue['correction']}'")
                    print(f"  [{issue['severity']}] {issue['description']}: '{issue['pattern']}' → '{issue['correction']}'")
                    if self.logger.level <= logging.DEBUG:
                        self.logger.debug(f"    Context: ...{issue['context']}...")
            else:
                self.logger.info("No obvious quality issues detected in Chinese translation")

        # 10) Apply basic review-based improvements
        translated_text = self._apply_review_suggestions(translated_text, "")

        return translated_text

    def _map_masked_segments(self, masked_text: str, should_process, process_core, memo: Dict[str, str]) -> str:
        """Map plain-text runs between mask placeholders through process_core.

        Placeholders are kept verbatim and NEVER passed to process_core (so they
        are never sent to the model); only the text between them is processed.
        Shared by translation and review-fix so both exclude placeholders.
        """
        parts = PLACEHOLDER_RE.split(masked_text)
        tokens = PLACEHOLDER_RE.findall(masked_text)
        out = []
        for i, part in enumerate(parts):
            out.append(self._map_plain_run(part, should_process, process_core, memo))
            if i < len(tokens):
                out.append(tokens[i])
        return "".join(out)

    def _map_plain_run(self, run: str, should_process, process_core, memo: Dict[str, str]) -> str:
        """Process a placeholder-free run paragraph by paragraph.

        Blank-line separators and pieces that should not be processed are kept
        exactly as-is so document structure is preserved.
        """
        if not run or not should_process(run):
            return run
        pieces = re.split(r'(\n[ \t\u3000]*\n)', run)
        out = []
        for piece in pieces:
            if not piece or not should_process(piece):
                out.append(piece)
            else:
                out.append(self._map_paragraph(piece, should_process, process_core, memo))
        return "".join(out)

    def _map_paragraph(self, paragraph: str, should_process, process_core, memo: Dict[str, str]) -> str:
        """Process one paragraph; fall back to line-by-line if line count changes."""
        if paragraph in memo:
            return memo[paragraph]
        result = process_core(paragraph)
        # Structure guard: output must keep the same number of lines.
        if result.count("\n") != paragraph.count("\n"):
            self.logger.warning(
                f"Line count changed ({paragraph.count(chr(10))} -> {result.count(chr(10))}); "
                f"falling back to line-by-line processing"
            )
            lines = []
            for line in paragraph.split("\n"):
                lines.append(process_core(line) if should_process(line) else line)
            result = "\n".join(lines)
        memo[paragraph] = result
        return result

    def _relevant_dictionary_terms(self, source_text: str) -> List[str]:
        """Return 'jp -> target' glossary lines for dictionary terms present in source."""
        target_dict = {}
        if hasattr(self, 'full_dictionary') and self.full_dictionary:
            target_dict = self.full_dictionary.get(self.target_label, {}) or {}
        if not target_dict and isinstance(getattr(self, 'dictionary', None), dict):
            target_dict = self.dictionary
        terms = []
        for jp in sorted(target_dict.keys(), key=len, reverse=True):
            tgt = target_dict.get(jp)
            if jp and tgt and jp in source_text:
                terms.append(f"{jp} -> {tgt}")
        return terms

    def _invoke_paragraph_translation(self, segment: str, llm, model_name) -> str:
        """Send a single placeholder-free Japanese segment to the model.

        Returns the translation, or the original segment on failure / suspicious
        output (empty or absurdly long), so product-manual text is never replaced
        by hallucinated content.
        """
        if not segment.strip():
            return segment
        # Preserve leading/trailing whitespace; only translate the core text.
        leading = segment[:len(segment) - len(segment.lstrip())]
        trailing = segment[len(segment.rstrip()):]
        core = segment.strip()

        target_lang_instruction = {
            "zh_tw": "Traditional Chinese (Taiwan, zh-TW)",
            "zh-TW": "Traditional Chinese (Taiwan, zh-TW)",
            "zh_cn": "Simplified Chinese (Mainland China, zh-CN)",
            "zh-CN": "Simplified Chinese (Mainland China, zh-CN)",
            "en": "English (US)",
            "th": "Thai",
        }.get(self.target_language, self.target_language)

        glossary = self._relevant_dictionary_terms(core)
        glossary_block = ""
        if glossary:
            glossary_block = "\nUse EXACTLY these terminology translations:\n" + "\n".join(glossary) + "\n"

        prompt = (
            f"You are a professional translator for product manuals. "
            f"Translate the following Japanese text into {target_lang_instruction}.\n"
            f"RULES:\n"
            f"- Output ONLY the translation, nothing else (no notes, no quotes, no explanations).\n"
            f"- Translate the meaning faithfully. Do NOT add, invent, or omit information.\n"
            f"- Keep the same number of lines; do not merge or split lines.\n"
            f"- Keep numbers, symbols, English words, and product names unchanged.\n"
            f"- If the text has no translatable content, output it unchanged.\n"
            f"{glossary_block}"
            f"\nJapanese text:\n{core}"
        )

        result = self._call_model_on_segment(prompt, core, leading, trailing, llm, model_name)
        return result if result is not None else segment

    def _call_model_on_segment(self, prompt: str, core: str, leading: str, trailing: str, llm, model_name):
        """Invoke the model on a single placeholder-free segment.

        Returns the cleaned output wrapped back in its original surrounding
        whitespace, or None when the call fails or the output looks like a
        hallucination (empty, or absurdly long), signalling the caller to keep
        the source text unchanged.
        """
        try:
            if hasattr(llm, 'invoke'):
                result = llm.invoke(prompt)
                content = result.content if hasattr(result, 'content') else str(result)
            elif hasattr(llm, 'chat') and hasattr(llm.chat, 'completions'):
                response = llm.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                content = response.choices[0].message.content
            else:
                return None
        except Exception as e:
            self.logger.warning(f"Segment model call failed, keeping source: {e}")
            return None

        if content is None:
            return None
        # Strip stray reasoning / code-fence wrappers some models add.
        if '</think>' in content:
            content = content[content.rfind('</think>') + len('</think>'):]
        content = self._remove_markdown_code_fence(content).strip()

        # Anti-hallucination guard: empty or absurdly long output -> keep source.
        if not content:
            return None
        if len(content) > max(80, len(core) * 6):
            self.logger.warning(
                f"Suspiciously long output ({len(content)} vs core {len(core)}); keeping source"
            )
            return None
        return f"{leading}{content}{trailing}"

    def _invoke_paragraph_fix(self, segment: str, issues_text: str, llm, model_name) -> str:
        """Apply review issues to a single placeholder-free target-language segment.

        Placeholders are never included here (the caller strips them). The model
        is told to fix ONLY the listed issues and otherwise return the text
        unchanged, so good translations are not degraded.
        """
        if not segment.strip():
            return segment
        leading = segment[:len(segment) - len(segment.lstrip())]
        trailing = segment[len(segment.rstrip()):]
        core = segment.strip()

        target_lang_instruction = {
            "zh_tw": "Traditional Chinese (Taiwan, zh-TW)",
            "zh-TW": "Traditional Chinese (Taiwan, zh-TW)",
            "zh_cn": "Simplified Chinese (Mainland China, zh-CN)",
            "zh-CN": "Simplified Chinese (Mainland China, zh-CN)",
            "en": "English (US)",
            "th": "Thai",
        }.get(self.target_language, self.target_language)

        prompt = (
            f"You are proofreading a {target_lang_instruction} product-manual segment.\n"
            f"Apply ONLY the review issues below that clearly apply to THIS segment.\n"
            f"RULES:\n"
            f"- Output ONLY the corrected text, nothing else (no notes, no quotes, no explanations).\n"
            f"- If no listed issue applies, output the segment UNCHANGED, exactly as given.\n"
            f"- Do NOT add, invent, translate away, or omit information.\n"
            f"- Keep the same number of lines; do not merge or split lines.\n"
            f"- Keep numbers, symbols, English words, and product names unchanged.\n"
            f"\nReview issues:\n{issues_text}\n"
            f"\nSegment to proofread:\n{core}"
        )

        result = self._call_model_on_segment(prompt, core, leading, trailing, llm, model_name)
        return result if result is not None else segment

    def _mask_non_translatable(self, text: str):
        """Mask non-translatable content, return (masked_text, placeholders)"""
        masked = text
        placeholders: Dict[str, Dict[str, str]] = {
            "code_blocks": {},
            "inline_code": {},
            "raw_urls": {},
            "emails": {},
            "placeholders": {},
            "filenames": {},
            "image_urls": {},
            "link_urls": {},
            "yaml_frontmatter": {},
            "gitbook_hint_blocks": {},
            "gitbook_tabs_blocks": {},
            "gitbook_tags": {},
            "template_expressions": {},
            "html_tags": {}
        }
        
        # Mask GitBook-specific syntax first (before other masking)
        masked = self._mask_gitbook_syntax(masked, placeholders)

        # Code blocks ```...```
        code_block_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)
        def repl_code_block(m):
            idx = len(placeholders["code_blocks"])  # 0-based
            ph = f"__CODE_BLOCK_{idx}__"
            placeholders["code_blocks"][ph] = m.group(0)
            return ph
        masked = code_block_pattern.sub(repl_code_block, masked)

        # Inline code `...`
        inline_code_pattern = re.compile(r"`[^`]+`")
        def repl_inline_code(m):
            idx = len(placeholders["inline_code"])  # 0-based
            ph = f"__INLINE_CODE_{idx}__"
            placeholders["inline_code"][ph] = m.group(0)
            return ph
        masked = inline_code_pattern.sub(repl_inline_code, masked)

        # Images: mask URL only, keep translatable alt text
        def repl_image(m):
            alt = m.group(1)
            url = m.group(2)
            idx = len(placeholders["image_urls"])  # 0-based
            ph = f"__IMAGE_URL_{idx}__"
            placeholders["image_urls"][ph] = url
            return f"![{alt}]({ph})"
        masked = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl_image, masked)

        # Links: mask URL only, keep translatable link text
        def repl_link(m):
            text_part = m.group(1)
            url = m.group(2)
            idx = len(placeholders["link_urls"])  # 0-based
            ph = f"__LINK_URL_{idx}__"
            placeholders["link_urls"][ph] = url
            return f"[{text_part}]({ph})"
        masked = re.sub(r"(?<!!)\[([^\]]*)\]\(([^)]+)\)", repl_link, masked)

        # Raw URLs (outside markdown link syntax)
        raw_url_pattern = re.compile(r"https?://[^\s)]+")
        def repl_raw_url(m):
            idx = len(placeholders["raw_urls"])  # 0-based
            ph = f"__RAW_URL_{idx}__"
            placeholders["raw_urls"][ph] = m.group(0)
            return ph
        masked = raw_url_pattern.sub(repl_raw_url, masked)

        # Emails
        email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
        def repl_email(m):
            idx = len(placeholders["emails"])  # 0-based
            ph = f"__EMAIL_{idx}__"
            placeholders["emails"][ph] = m.group(0)
            return ph
        masked = email_pattern.sub(repl_email, masked)

        # 花括号占位符 {serial_number}
        curly_pattern = re.compile(r"\{[^}]+\}")
        def repl_curly(m):
            idx = len(placeholders["placeholders"])  # 0-based
            ph = f"__PLACEHOLDER_{idx}__"
            placeholders["placeholders"][ph] = m.group(0)
            return ph
        masked = curly_pattern.sub(repl_curly, masked)

        # Common filenames (with extensions)
        filename_pattern = re.compile(r"\b[\w\-/\.]+\.(?:md|txt|pdf|jpg|jpeg|png|gif|zip|exe|dll|ini|cfg)\b", re.IGNORECASE)
        def repl_filename(m):
            idx = len(placeholders["filenames"])  # 0-based
            ph = f"__FILENAME_{idx}__"
            placeholders["filenames"][ph] = m.group(0)
            return ph
        masked = filename_pattern.sub(repl_filename, masked)

        return masked, placeholders

    def _apply_dictionary_on_text(self, text: str, dictionary: Dict[str, str]) -> str:
        """对可翻译文本应用词典替换：
        - 按键长度降序替换，避免短键覆盖长键
        - 只进行简单的子串替换，不改变空白与换行
        """
        if not dictionary:
            return text
        for key in sorted(dictionary.keys(), key=len, reverse=True):
            value = dictionary.get(key, "")
            if not value:
                continue
            text = text.replace(key, value)
        return text

    def _enforce_dictionary_after_translation(self, text: str) -> str:
        """Run dictionary replacement again on translation to clean leftover Japanese terms.
        Operates only on translatable areas; avoids modifying code/URLs/placeholders, etc.
        """
        # Detect Japanese characters: Hiragana, Katakana, common Kanji
        if not re.search(r"[\u3040-\u30FF\u4E00-\u9FFF]", text or ""):
            return text
        masked, placeholders = self._mask_non_translatable(text)
        applied = self._apply_dictionary_on_text(masked, self.dictionary)
        restored = self._restore_placeholders(applied, placeholders)
        return restored
    
    def _validate_chinese_translation(self, text: str) -> List[Dict[str, str]]:
        """中国語翻訳の品質チェック
        
        明らかな誤字、不自然な表現、一般的な誤訳パターンを検出します。
        
        Args:
            text: 翻訳されたテキスト
            
        Returns:
            検出された問題のリスト [{"pattern": str, "correction": str, "context": str, "severity": str}]
        """
        issues = []
        
        # 一般的な誤字・誤訳パターン（正規表現パターン: 修正候補）
        common_errors = {
            # 「割り当て」関連の誤訳
            r'分钟配': ('分配', 'MAJOR', '「割り当て」の誤訳'),
            r'没有分钟配': ('未分配', 'MAJOR', '「割り当てられていない」の誤訳'),
            r'已分钟配': ('已分配', 'MAJOR', '「割り当て済み」の誤訳'),
            
            # その他の一般的な同音異義語エラー
            r'在先': ('在线', 'MINOR', '「オンライン」の誤訳の可能性'),
            r'文件夹加': ('文件夹', 'MINOR', '余分な文字'),
            r'设定值': ('设置', 'MINOR', '「設定」の不自然な表現の可能性'),
            
            # 不自然な文字列の組み合わせ
            r'功能区区': ('功能区', 'MINOR', '重複文字'),
            r'报表表': ('报表', 'MINOR', '重複文字'),
            r'许可可': ('许可', 'MINOR', '重複文字'),
            
            # スペースの問題
            r'([a-zA-Z]+)([一-龥])': (r'\1 \2', 'MINOR', '英数字と中国語の間にスペースが必要'),
            r'([一-龥])([a-zA-Z]+)': (r'\1 \2', 'MINOR', '中国語と英数字の間にスペースが必要'),
        }
        
        # マスク処理: コードブロック、インラインコード、URLなどを除外
        masked_text, placeholders = self._mask_non_translatable(text)
        
        # 各パターンをチェック
        for pattern, (correction, severity, description) in common_errors.items():
            matches = list(re.finditer(pattern, masked_text))
            for match in matches:
                # コンテキストを取得（前後20文字）
                start = max(0, match.start() - 20)
                end = min(len(masked_text), match.end() + 20)
                context = masked_text[start:end]
                
                # プレースホルダー内のマッチは無視
                if '__' in match.group(0) or match.group(0).startswith('__'):
                    continue
                
                issues.append({
                    "pattern": match.group(0),
                    "correction": correction,
                    "context": context,
                    "severity": severity,
                    "description": description,
                    "position": match.start()
                })
        
        # 重複を削除（同じパターンと位置）
        unique_issues = []
        seen = set()
        for issue in issues:
            key = (issue['pattern'], issue['position'])
            if key not in seen:
                seen.add(key)
                unique_issues.append(issue)
        
        return unique_issues

    def _restore_placeholders(self, text: str, placeholders: Dict[str, Dict[str, str]]) -> str:
        """Restore all placeholder content back into the text
        
        復元順序を制御して、辞書プレースホルダーを最初に復元し、
        その後でGitBook要素などの他のプレースホルダーを復元します。
        """
        restored = text
        
        # 優先順位付き復元（辞書プレースホルダーを最初に復元）
        priority_order = [
            "dict_terms",  # 辞書用語プレースホルダーを最初に復元
            "yaml_frontmatter",
            "gitbook_hint_blocks",
            "gitbook_tabs_blocks",
            "gitbook_tags",
            "template_expressions",
            "html_tags",
            "code_blocks",
            "inline_code",
            "image_urls",
            "link_urls",
            "raw_urls",
            "emails",
            "placeholders",
            "filenames"
        ]
        
        # 優先順位に従って復元
        restored_count = 0
        for category in priority_order:
            if category in placeholders:
                category_count = 0
                for ph, original in placeholders[category].items():
                    if ph in restored:
                        restored = restored.replace(ph, original)
                        category_count += 1
                        restored_count += 1
                if category_count > 0:
                    self.logger.debug(f"Restored {category_count} placeholders from category: {category}")
        
        # 残りのカテゴリーも復元（念のため）
        for category, mapping in placeholders.items():
            if category not in priority_order:
                category_count = 0
                for ph, original in mapping.items():
                    if ph in restored:
                        restored = restored.replace(ph, original)
                        category_count += 1
                        restored_count += 1
                if category_count > 0:
                    self.logger.debug(f"Restored {category_count} placeholders from category: {category}")
        
        self.logger.info(f"Total restored placeholders: {restored_count}")
        return restored
    
    def _remove_markdown_code_fence(self, text: str) -> str:
        """Remove markdown code fence wrappers possibly added by GPT-4o
        
        Handles patterns:
        1. Leading ```markdown or ``` (possibly multiple backticks)
        2. Trailing ``` (possibly multiple backticks)
        
        Args:
            text: text to clean
            
        Returns:
            cleaned text
        """
        if not text or not isinstance(text, str):
            return text
        
        # Remove opening code fence
        # Pattern: line-start backticks (3 or more) + optional 'markdown' + newline
        import re
        
        # Check opening fence
        start_pattern = r'^`{3,}(?:markdown)?\s*\n'
        text = re.sub(start_pattern, '', text, count=1)
        
        # Check trailing fence
        # Pattern: newline + backticks (3 or more) + optional spaces + end of line
        end_pattern = r'\n`{3,}\s*$'
        text = re.sub(end_pattern, '', text, count=1)
        
        return text
    
    def _execute_llm_request(self, *args, **kwargs):
        """执行LLM请求的统一方法，支持传统和增量翻译两种调用方式"""
        try:
            # 判断是传统翻译调用还是增量翻译调用
            if len(args) == 4 and not kwargs:
                # 传统翻译调用方式: llm, prompt, text, model_name=None
                llm, prompt, text, model_name = args[0], args[1], args[2], args[3] if len(args) > 3 else None
                
                if hasattr(llm, 'invoke'):
                    # Format prompt based on input_variables
                    if hasattr(prompt, 'input_variables') and "target_language" in prompt.input_variables:
                        formatted_prompt = prompt.format(text=text, target_language=self.target_language)
                    else:
                        formatted_prompt = prompt.format(text=text)
                    
                    # Log the request (without dictionary content)
                    self.logger.info(f"Sending translation request via invoke() to {model_name or 'LLM'}")
                    self.logger.debug(f"Prompt length: {len(formatted_prompt)} characters")
                    self.logger.info(f"Translation input text length: {len(text)} characters")
                    self.logger.debug(f"Translation input text (first 2000 chars):\n{text[:2000]}")
                    if len(text) > 2000:
                        self.logger.debug(f"... (truncated, total {len(text)} chars)")
                    
                    result = llm.invoke(formatted_prompt)
                    
                    # Extract content
                    if hasattr(result, 'content'):
                        result_content = result.content
                    else:
                        result_content = str(result)
                    
                    # Log the response
                    self.logger.info(f"Received response via invoke() (length: {len(result_content)} chars)")
                    self.logger.debug(f"Translation response (first 2000 chars):\n{result_content[:2000]}")
                    if len(result_content) > 2000:
                        self.logger.debug(f"... (truncated, total {len(result_content)} chars)")
                    
                    return result_content
                elif hasattr(llm, 'chat') and hasattr(llm.chat, 'completions'):
                    # Use OpenAI-compatible client: get formatted prompt from template
                    if hasattr(prompt, 'format'):
                        if hasattr(prompt, 'input_variables') and "target_language" in prompt.input_variables:
                            formatted_prompt = prompt.format(text=text, target_language=self.target_language)
                        else:
                            formatted_prompt = prompt.format(text=text)
                    else:
                        formatted_prompt = str(prompt) + "\n\n" + text
                    
                    # Log the request (without dictionary content)
                    self.logger.info(f"Sending translation request to {model_name}")
                    self.logger.debug(f"Prompt length: {len(formatted_prompt)} characters")
                    self.logger.info(f"Translation input text length: {len(text)} characters")
                    self.logger.debug(f"Translation input text (first 2000 chars):\n{text[:2000]}")
                    if len(text) > 2000:
                        self.logger.debug(f"... (truncated, total {len(text)} chars)")
                    
                    # Debug: Print first 500 chars of prompt
                    print(f"\n[DEBUG] Sending prompt to {model_name}:")
                    print(f"[DEBUG] Prompt preview (first 500 chars): {formatted_prompt[:500]}...")
                    print(f"[DEBUG] Text length: {len(text)} chars\n")
                    
                    response = llm.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "user", "content": formatted_prompt}
                        ],
                        temperature=0
                    )
                    result = response.choices[0].message.content
                    
                    # Log the response
                    self.logger.info(f"Received response from {model_name} (length: {len(result) if result else 0} chars)")
                    self.logger.debug(f"Translation response (first 2000 chars):\n{result[:2000] if result else 'None'}")
                    if result and len(result) > 2000:
                        self.logger.debug(f"... (truncated, total {len(result)} chars)")
                    
                    # Debug: Print first 300 chars of result
                    print(f"\n[DEBUG] Received response from {model_name}:")
                    print(f"[DEBUG] Response preview (first 300 chars): {result[:300] if result else 'None'}...\n")
                    
                    return result
                else:
                    # 兼容旧版本
                    if LLMChain:
                        chain = LLMChain(llm=llm, prompt=prompt)
                        return chain.run(text=text, target_language=self.target_language)
                    else:
                        # Fallback: use invoke directly
                        if hasattr(prompt, 'input_variables') and "target_language" in prompt.input_variables:
                            result = llm.invoke(prompt.format(text=text, target_language=self.target_language))
                        else:
                            result = llm.invoke(prompt.format(text=text))
                        if hasattr(result, 'content'):
                            return result.content
                        else:
                            return str(result)
            elif len(args) == 3 and not kwargs:
                # 增量翻译调用方式: translation_prompt, target_language, task_type="partial_translation"
                translation_prompt, target_language, task_type = args[0], args[1], args[2]
                
                # 获取LLM客户端
                llm_result = self._get_llm_client(task_type)
                if isinstance(llm_result, tuple):
                    llm, model_name = llm_result
                else:
                    llm = llm_result
                    model_name = None
                
                if hasattr(llm, 'invoke'):
                    result = llm.invoke(translation_prompt)
                    if hasattr(result, 'content'):
                        return result.content
                    else:
                        return str(result)
                elif hasattr(llm, 'chat') and hasattr(llm.chat, 'completions'):
                    response = llm.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "根据提供的上下文和变更内容，进行增量翻译："},
                            {"role": "user", "content": translation_prompt}
                        ],
                        temperature=0
                    )
                    return response.choices[0].message.content
                else:
                    # 兼容旧版本
                    if LLMChain:
                        chain = LLMChain(llm=llm, prompt=PromptTemplate(
                            input_variables=["content"],
                            template="{content}"
                        ))
                        return chain.run(content=translation_prompt)
                    else:
                        # Fallback: use invoke directly
                        result = llm.invoke(translation_prompt)
                        if hasattr(result, 'content'):
                            return result.content
                        else:
                            return str(result)
        except Exception as e:
            self.logger.error(f"LLM request failed: {e}", exc_info=True)
            print(f"LLM请求失败: {e}")
            print(f"错误详情: {type(e).__name__}: {str(e)}")
            # 回退到Ollama（翻译）
            failed_model_name = None
            if len(args) == 4:
                failed_model_name = args[3]
            elif len(args) >= 1 and not isinstance(args[0], str):
                failed_model_name = getattr(args[0], "model_name", None) or getattr(args[0], "model", None)

            try:
                self.logger.warning("Using Ollama fallback for translation")
                if len(args) == 4:
                    # 传统翻译回退
                    prompt = args[1]
                    text = args[2]
                    if hasattr(prompt, "format"):
                        if hasattr(prompt, "input_variables") and "target_language" in prompt.input_variables:
                            formatted_prompt = prompt.format(text=text, target_language=self.target_language)
                        else:
                            formatted_prompt = prompt.format(text=text)
                    else:
                        formatted_prompt = f"{prompt}\n{text}"
                else:
                    # 增量翻译回退（尽可能原样调用）
                    translation_prompt = args[0] if args and isinstance(args[0], str) else str(args[0])
                    formatted_prompt = translation_prompt

                ollama_result = self._ollama_invoke_text(
                    formatted_prompt,
                    failed_model_name=failed_model_name,
                    task_type="translation"
                )
                self.logger.info(f"Ollama fallback translation successful (length: {len(ollama_result)} chars)")
                return ollama_result
            except Exception as ollama_error:
                self.logger.error(f"Ollama fallback translation also failed: {ollama_error}", exc_info=True)
                print(f"Ollama回退翻译也失败: {ollama_error}")
            
            # 返回原始文本作为最后的回退
            if len(args) >= 3 and isinstance(args[2], str):
                self.logger.warning("Returning original text as final fallback")
                return args[2]  # 传统翻译的text参数
            elif len(args) >= 1 and isinstance(args[0], str):
                self.logger.warning("Returning original prompt as final fallback")
                return args[0]  # 增量翻译的translation_prompt参数
            return ""
        
    def _apply_review_suggestions(self, translated_text: str, review_content: str) -> str:
        """根据review中的建议改进翻译结果"""
        try:
            print("正在应用审核建议改进翻译结果...")
            
            # 基础策略：不改变换行与空白、不擅自改动Markdown结构
            improved_text = translated_text
            
            # 如果有审核内容，尝试解析并应用具体的建议
            if review_content:
                # 创建一个更严格的LLM请求：在保持目标语言与格式不变的前提下仅做必要修改
                target_lang_desc_map = {
                    "en": "英语（English）",
                    "zh-CN": "简体中文（zh-CN）",
                    "zh-TW": "繁体中文（台湾，zh-TW）",
                    "th": "泰语（Thai）",
                }
                target_lang_desc = target_lang_desc_map.get(self.target_label, self.target_language)
                prompt = (
                    f"请依据以下审核意见，对翻译文本进行必要的局部改进。\n"
                    f"严格要求：\n"
                    f"- 保持译文语言为{target_lang_desc}（不得改变语言）。\n"
                    f"- 保持原文所有换行与空白字符完全不变。\n"
                    f"- 保持Markdown结构与符号完全不变。\n"
                    f"- 保证Markdown格式语法完全正确，不改变任何Markdown元素的位置或格式。\n"
                    f"- **重要：对于Markdown链接[文本](URL)，圆括号内的URL必须完全保持不变，不得修改、翻译或替换。"
                    f"如果URL是__LINK_URL_N__或__IMAGE_URL_N__形式的占位符，必须原样保留。\n"
                    f"- 不要翻译或改动代码块/行内代码/URL/邮箱/占位符/文件名、已有英文术语。\n\n"
                    f"翻译文本：\n{translated_text}\n\n"
                    f"审核意见：\n{review_content}\n\n"
                    f"请仅返回改动后的完整翻译文本，不要包含任何解释。"
                )
                
                # 使用与翻译相同的模型进行改进
                llm_result = self._get_llm_client("translate")
                if isinstance(llm_result, tuple):
                    llm, model_name = llm_result
                else:
                    llm = llm_result
                    model_name = "gpt-3.5-turbo"
                
                try:
                    if hasattr(llm, 'invoke'):
                        result = llm.invoke(prompt)
                        if hasattr(result, 'content'):
                            improved_text = result.content
                        else:
                            improved_text = str(result)
                    elif hasattr(llm, 'chat') and hasattr(llm.chat, 'completions'):
                        # 直接使用OpenAI客户端兼容的API
                        response = llm.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": (
                                    "你是专业的本地化改进助手。必须：\n"
                                    "- 保持译文语言与目标语言一致，不得改变语言；\n"
                                    "- 保持所有换行与空白字符完全不变；\n"
                                    "- 保持Markdown结构与符号完全不变；\n"
                                    "- 保证Markdown格式语法完全正确，不改变任何Markdown元素的位置或格式；\n"
                                    "- 不要改动不可翻译元素（代码/行内代码/URL/链接/邮箱/占位符/文件名、已有英文术语）；\n"
                                    "仅输出改动后的完整译文，不要包含任何解释。"
                                )},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0
                        )
                        improved_text = response.choices[0].message.content
                    else:
                        # 兼容旧版本
                        if LLMChain:
                            chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=[], template=prompt))
                            improved_text = chain.run({})
                        else:
                            # Fallback: use invoke directly
                            result = llm.invoke(prompt)
                            if hasattr(result, 'content'):
                                improved_text = result.content
                            else:
                                improved_text = str(result)
                    
                    print("成功应用审核建议改进翻译结果")
                except Exception as e:
                    print(f"尝试使用LLM改进翻译结果时出错: {e}")
                    print("将使用基础改进策略")
            
            return improved_text
        except Exception as e:
            print(f"应用审核建议时出错: {e}")
            return translated_text

    def translate_file(self, file_path: str) -> str:
        """翻译单个MD文件，支持差分比较和增量翻译"""
        self.logger.info(f"Starting translation of file: {file_path}")
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                current_content = f.read()
            self.logger.debug(f"File content loaded: {len(current_content)} characters")
            
            # 获取文件当前的SHA值（如果可用）
            current_sha = self.file_sha_map.get(file_path) if hasattr(self, 'file_sha_map') else None
            
            # 检查文件是否有变化
            file_has_changed = self.state_manager.has_file_changed(file_path, current_content, current_sha)
            
            # 如果文件没有变化，直接返回之前的翻译结果
            if not file_has_changed and self.state_manager.has_translation_history(file_path, self.target_language):
                previous_translation_path = self.state_manager.get_previous_translation_path(file_path, self.target_language)
                if os.path.exists(previous_translation_path):
                    self.logger.info(f"File {file_path} unchanged, using previous translation")
                    print(f"文件 {file_path} 未发生变化，使用上次的翻译结果")
                    return previous_translation_path
            
            # 检查是否有上次的翻译结果
            previous_translation_content = None
            previous_translation_path = self.state_manager.get_previous_translation_path(file_path, self.target_language)
            if os.path.exists(previous_translation_path):
                with open(previous_translation_path, 'r', encoding='utf-8') as f:
                    previous_translation_content = f.read()
            
            # 确定是否需要进行增量翻译
            if file_has_changed and previous_translation_content:
                self.logger.info(f"File {file_path} changed, performing incremental translation")
                print(f"文件 {file_path} 发生变化，进行增量翻译")
                
                # 获取上次的文件内容
                previous_content = self.state_manager.get_previous_file_content(file_path)
                if previous_content:
                    # 比较文件差异，获取变化的部分
                    diff_result = self.diff_analyzer.compare_files(previous_content, current_content)
                    
                    if diff_result['has_changes']:
                        # 提取变更的章节
                        changed_sections = self.diff_analyzer.get_changed_sections(previous_content, current_content)
                        
                        # 生成部分翻译提示
                        translation_prompt = self.diff_analyzer.generate_partial_translation_prompt(
                            previous_content, current_content, previous_translation_content, 
                            changed_sections, self.target_language
                        )
                        
                        # 执行部分翻译
                        partial_translation = self._execute_llm_request(
                            translation_prompt,
                            self.target_language,
                            task_type="partial_translation"
                        )
                        
                        # 确保部分翻译结果不为None
                        if partial_translation:
                            translated_content = partial_translation
                        else:
                            # 如果部分翻译失败，回退到全文翻译
                            print(f"部分翻译失败，回退到全文翻译")
                            translated_content = self._translate_text(current_content)
                    else:
                        # 如果没有实际变化，使用上次的翻译
                        translated_content = previous_translation_content
                else:
                    # 如果没有上次的内容记录，进行全文翻译
                    translated_content = self._translate_text(current_content)
            else:
                # 执行全文翻译
                self.logger.info(f"Performing full translation of file: {file_path}")
                print(f"执行文件 {file_path} 的全文翻译")
                translated_content = self._translate_text(current_content)
                self.logger.info(f"Full translation completed for file: {file_path} (length: {len(translated_content)} chars)")
            
            # 生成新文件名
            base_name = os.path.basename(file_path)
            dir_name = os.path.dirname(file_path)
            
            # 确定文件所属的分支
            branch_name = "main"  # 默认分支
            # 从文件路径中提取分支信息
            if self.download_dir in file_path:
                path_parts = os.path.relpath(file_path, self.download_dir).split(os.sep)
                if len(path_parts) > 0 and path_parts[0]:
                    # 假设路径格式为: download_dir/branch_name/...
                    branch_name = path_parts[0]
            
            # 构建相对路径（移除分支名称）
            if self.download_dir in file_path:
                relative_path_with_branch = os.path.relpath(dir_name, self.download_dir)
                # 移除路径中的分支部分
                if branch_name in relative_path_with_branch:
                    relative_path = relative_path_with_branch[len(branch_name) + len(os.sep):] if relative_path_with_branch.startswith(branch_name + os.sep) else relative_path_with_branch
                else:
                    relative_path = relative_path_with_branch
            else:
                relative_path = ""
            
            # Output naming: suffix or directory
            name_without_ext, ext = os.path.splitext(base_name)
            
            if self.output_naming == "suffix":
                # Suffix mode: intro.en.md, intro.zh-CN.md
                new_file_name = f"{name_without_ext}_temp{ext}"
                if relative_path == "":
                    target_dir = os.path.join(self.translated_dir, self.target_language)
                else:
                    target_dir = os.path.join(self.translated_dir, self.target_language, branch_name, relative_path)
            else:
                # Directory mode: /en/intro.md, /zh-CN/intro.md
                new_file_name = f"{base_name.replace(ext, '_temp' + ext)}"
                if relative_path == "":
                    target_dir = os.path.join(self.translated_dir, self.target_label)
                else:
                    target_dir = os.path.join(self.translated_dir, self.target_label, branch_name, relative_path)
            
            temp_target_dir = os.path.join(target_dir, "temp")
            
            # 确保目标目录与temp目录存在
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                self.logger.debug(f"Created target directory: {target_dir}")
            if not os.path.exists(temp_target_dir):
                os.makedirs(temp_target_dir, exist_ok=True)
                self.logger.debug(f"Created temp directory: {temp_target_dir}")
            
            # 确保翻译内容不为None
            if translated_content is None:
                translated_content = ""  # 或使用其他合适的默认值
                self.logger.warning(f"Translation content is None for file {file_path}, using empty string")
                print(f"警告：翻译内容为空，使用默认值")
            
            # 最终检查：移除任何残留的markdown代码块包裹标记
            if self.target_language.lower() == "en":
                translated_content = self._remove_markdown_code_fence(translated_content)
                
            # 写入翻译后的文件到temp目录
            new_file_path = os.path.join(temp_target_dir, new_file_name)
            self.logger.info(f"Saving translated file to: {new_file_path} (length: {len(translated_content)} chars)")
            with open(new_file_path, 'w', encoding='utf-8') as f:
                f.write(translated_content)
            self.logger.info(f"Translated file saved successfully: {new_file_path}")
            
            # 更新翻译状态
            self.state_manager.update_translation_state(
                file_path, current_content, current_sha, 
                new_file_path, translated_content, self.target_language
            )
            self.logger.debug(f"Translation state updated for file: {file_path}")
            
            return new_file_path
        except Exception as e:
            self.logger.error(f"Error translating file {file_path}: {e}", exc_info=True)
            print(f"翻译文件 {file_path} 时出错: {e}")
            return ""
    
    def _extract_structure(self, text: str) -> Dict[str, List[Dict]]:
        """Extract GitBook and HTML structure from text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing structure information
        """
        structure = {
            "gitbook_hints": [],      # {% hint %} ... {% endhint %}
            "gitbook_tabs": [],       # {% tabs %} ... {% endtabs %}
            "gitbook_tab": [],        # {% tab %} ... {% endtab %}
            "gitbook_tags": [],       # {% include %}, {% embed %}, etc.
            "html_tags": [],          # <div>, <p>, etc.
            "yaml_frontmatter": None, # --- ... ---
            "code_blocks": [],        # ``` ... ```
            "template_expressions": [] # {{ ... }}
        }
        
        # 1. YAML frontmatter
        yaml_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.MULTILINE | re.DOTALL)
        for match in yaml_pattern.finditer(text):
            structure["yaml_frontmatter"] = {
                "start": match.start(),
                "end": match.end(),
                "content": match.group(0)
            }
        
        # 2. GitBook hint tags
        hint_open_pattern = re.compile(r'{%\s*hint\s+style="([^"]*)"\s*%}')
        hint_close_pattern = re.compile(r'{%\s*endhint\s*%}')
        
        for match in hint_open_pattern.finditer(text):
            structure["gitbook_hints"].append({
                "type": "open",
                "pos": match.start(),
                "style": match.group(1),
                "content": match.group(0)
            })
        
        for match in hint_close_pattern.finditer(text):
            structure["gitbook_hints"].append({
                "type": "close",
                "pos": match.start(),
                "content": match.group(0)
            })
        
        # 3. GitBook tabs tags
        tabs_open_pattern = re.compile(r'{%\s*tabs\s*%}')
        tabs_close_pattern = re.compile(r'{%\s*endtabs\s*%}')
        
        for match in tabs_open_pattern.finditer(text):
            structure["gitbook_tabs"].append({
                "type": "open",
                "pos": match.start(),
                "content": match.group(0)
            })
        
        for match in tabs_close_pattern.finditer(text):
            structure["gitbook_tabs"].append({
                "type": "close",
                "pos": match.start(),
                "content": match.group(0)
            })
        
        # 4. GitBook tab tags
        tab_open_pattern = re.compile(r'{%\s*tab\s+([^%]*)%}')
        tab_close_pattern = re.compile(r'{%\s*endtab\s*%}')
        
        for match in tab_open_pattern.finditer(text):
            structure["gitbook_tab"].append({
                "type": "open",
                "pos": match.start(),
                "title": match.group(1),
                "content": match.group(0)
            })
        
        for match in tab_close_pattern.finditer(text):
            structure["gitbook_tab"].append({
                "type": "close",
                "pos": match.start(),
                "content": match.group(0)
            })
        
        # 5. Other GitBook tags
        gitbook_single_pattern = re.compile(r'{%\s*(?:include|embed|file)\s+[^%]*%}')
        for match in gitbook_single_pattern.finditer(text):
            structure["gitbook_tags"].append({
                "pos": match.start(),
                "content": match.group(0)
            })
        
        # 6. Template expressions
        template_pattern = re.compile(r'{{[^}]+}}')
        for match in template_pattern.finditer(text):
            structure["template_expressions"].append({
                "pos": match.start(),
                "content": match.group(0)
            })
        
        # 7. Code blocks
        code_block_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        for match in code_block_pattern.finditer(text):
            structure["code_blocks"].append({
                "start": match.start(),
                "end": match.end(),
                "content": match.group(0)
            })
        
        # 8. HTML tags (opening and closing)
        html_open_pattern = re.compile(r'<([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>')
        html_close_pattern = re.compile(r'</([a-zA-Z][a-zA-Z0-9]*)>')
        
        for match in html_open_pattern.finditer(text):
            structure["html_tags"].append({
                "type": "open",
                "tag": match.group(1),
                "pos": match.start(),
                "content": match.group(0)
            })
        
        for match in html_close_pattern.finditer(text):
            structure["html_tags"].append({
                "type": "close",
                "tag": match.group(1),
                "pos": match.start(),
                "content": match.group(0)
            })
        
        return structure
    
    def _compare_structures(self, original_structure: Dict, translated_structure: Dict) -> List[Dict]:
        """Compare two structures and return differences
        
        Args:
            original_structure: Structure from original file
            translated_structure: Structure from translated file
            
        Returns:
            List of differences
        """
        differences = []
        
        # Compare GitBook hints
        orig_hints = original_structure["gitbook_hints"]
        trans_hints = translated_structure["gitbook_hints"]
        
        if len(orig_hints) != len(trans_hints):
            differences.append({
                "type": "count_mismatch",
                "element": "gitbook_hints",
                "original_count": len(orig_hints),
                "translated_count": len(trans_hints)
            })
        
        # Compare GitBook tabs
        orig_tabs = original_structure["gitbook_tabs"]
        trans_tabs = translated_structure["gitbook_tabs"]
        
        if len(orig_tabs) != len(trans_tabs):
            differences.append({
                "type": "count_mismatch",
                "element": "gitbook_tabs",
                "original_count": len(orig_tabs),
                "translated_count": len(trans_tabs)
            })
        
        # Compare GitBook tab
        orig_tab = original_structure["gitbook_tab"]
        trans_tab = translated_structure["gitbook_tab"]
        
        if len(orig_tab) != len(trans_tab):
            differences.append({
                "type": "count_mismatch",
                "element": "gitbook_tab",
                "original_count": len(orig_tab),
                "translated_count": len(trans_tab)
            })
        
        # Compare HTML tags
        orig_html = original_structure["html_tags"]
        trans_html = translated_structure["html_tags"]
        
        if len(orig_html) != len(trans_html):
            differences.append({
                "type": "count_mismatch",
                "element": "html_tags",
                "original_count": len(orig_html),
                "translated_count": len(trans_html)
            })
        
        # Compare YAML frontmatter
        orig_yaml = original_structure["yaml_frontmatter"]
        trans_yaml = translated_structure["yaml_frontmatter"]
        
        if (orig_yaml is None) != (trans_yaml is None):
            differences.append({
                "type": "presence_mismatch",
                "element": "yaml_frontmatter",
                "original_present": orig_yaml is not None,
                "translated_present": trans_yaml is not None
            })
        
        return differences
    
    def _fix_structure_mismatch(self, original_text: str, translated_text: str, 
                               original_structure: Dict) -> str:
        """Fix structure mismatches by reconstructing from original structure
        
        Args:
            original_text: Original text
            translated_text: Translated text with potential structure issues
            original_structure: Structure extracted from original
            
        Returns:
            Fixed translated text
        """
        # Strategy: Extract translated content between tags and reconstruct with original structure
        
        # For now, we'll use a simpler approach:
        # Re-translate with better preservation, or return translated text as-is
        # This is a placeholder for more sophisticated logic
        
        self.logger.warning("Structure mismatch detected, but automatic fix not fully implemented yet")
        self.logger.info("Recommendation: Review the translation manually")
        
        return translated_text
    
    def _validate_translation_structure(self, original_file: str, translated_file: str) -> Dict:
        """Validate and fix translation structure
        
        Args:
            original_file: Path to original file
            translated_file: Path to translated file
            
        Returns:
            Validation result dictionary
        """
        try:
            # Read files
            with open(original_file, 'r', encoding='utf-8') as f:
                original_text = f.read()
            
            with open(translated_file, 'r', encoding='utf-8') as f:
                translated_text = f.read()
            
            # Extract structures
            self.logger.info(f"Extracting structure from original file: {original_file}")
            original_structure = self._extract_structure(original_text)
            
            self.logger.info(f"Extracting structure from translated file: {translated_file}")
            translated_structure = self._extract_structure(translated_text)
            
            # Compare structures
            differences = self._compare_structures(original_structure, translated_structure)
            
            if differences:
                self.logger.warning(f"Structure differences detected: {len(differences)} issues")
                for diff in differences:
                    self.logger.warning(f"  - {diff['type']}: {diff['element']} "
                                      f"(original: {diff.get('original_count', 'N/A')}, "
                                      f"translated: {diff.get('translated_count', 'N/A')})")
                
                # Attempt to fix
                fixed_text = self._fix_structure_mismatch(original_text, translated_text, original_structure)
                
                # Save fixed file if different
                if fixed_text != translated_text:
                    with open(translated_file, 'w', encoding='utf-8') as f:
                        f.write(fixed_text)
                    self.logger.info(f"Fixed structure saved to: {translated_file}")
                    
                    return {
                        "has_differences": True,
                        "differences": differences,
                        "fixed": True,
                        "fixed_file": translated_file
                    }
                else:
                    return {
                        "has_differences": True,
                        "differences": differences,
                        "fixed": False,
                        "fixed_file": None
                    }
            else:
                self.logger.info("Structure validation passed: no differences detected")
                return {
                    "has_differences": False,
                    "differences": [],
                    "fixed": False,
                    "fixed_file": None
                }
        
        except Exception as e:
            self.logger.error(f"Error validating translation structure: {e}", exc_info=True)
            return {
                "has_differences": False,
                "differences": [],
                "fixed": False,
                "fixed_file": None,
                "error": str(e)
            }
    
    def _detect_japanese_in_translatable_text(self, text: str) -> Dict[str, Any]:
        """在可翻译区域检测日语字符（平假名・片假名）
        
        Args:
            text: 需要检查的已翻译文本
            
        Returns:
            {
                "has_japanese": bool,  # 是否检测到日语
                "samples": List[str],  # 检测到的日语样本（最多5个）
                "count": int  # 检测到的日语字符总数
            }
        """
        try:
            # 屏蔽不可翻译部分（代码块、URL等）
            masked_text, _ = self._mask_non_translatable(text)
            
            # 检测平假名（\u3040-\u309F）和片假名（\u30A0-\u30FF）
            hiragana_pattern = re.compile(r'[\u3040-\u309F]+')
            katakana_pattern = re.compile(r'[\u30A0-\u30FF]+')
            
            hiragana_matches = hiragana_pattern.findall(masked_text)
            katakana_matches = katakana_pattern.findall(masked_text)
            
            all_matches = hiragana_matches + katakana_matches
            
            # 返回检测结果
            has_japanese = len(all_matches) > 0
            samples = all_matches[:5] if has_japanese else []  # 最多5个样本
            count = sum(len(match) for match in all_matches)
            
            return {
                "has_japanese": has_japanese,
                "samples": samples,
                "count": count
            }
        except Exception as e:
            print(f"检测日语时发生错误: {e}")
            return {
                "has_japanese": False,
                "samples": [],
                "count": 0
            }
    
    def _review_with_severity(self, translated_content: str, original_content: str, 
                             japanese_detection: Dict[str, Any]) -> Dict[str, Any]:
        """Review with severity classification
        
        Args:
            translated_content: Translated content
            original_content: Original content
            japanese_detection: Japanese detection info
            
        Returns:
            {
                "issues": [{"severity": "BLOCKER"|"MAJOR"|"MINOR", "category": str, "description": str, "suggestion": str}],
                "has_blockers": bool,
                "has_majors": bool,
                "review_text": str
            }
        """
        try:
            # Japanese detection warning
            japanese_warning = ""
            if japanese_detection and japanese_detection.get("has_japanese"):
                japanese_warning = (
                    f"\n\n⚠️ 重要警告：翻訳結果中に{japanese_detection['count']}個の日本語文字（平仮名・片仮名）を検出。"
                    f"検出サンプル: {', '.join(japanese_detection['samples'])}。"
                    f"これらは{self.target_language}に翻訳する必要があります。"
                )
            
            # Create review prompt with severity classification
            review_template = (
                "あなたは資深本地化審査専門家です。以下の翻訳を厳格に審査し、問題を重大度で分類してください：\n\n"
                "【重大度分類】\n"
                "**BLOCKER**（必須修正、構造破壊レベル）：\n"
                "- 構造破壊（見出し、リスト、テーブルの崩れ）\n"
                "- GitBook構文の変更・削除\n"
                "- リンク/URL/パスの破壊\n"
                "- コードブロックの変更\n"
                "- 行の欠落・追加\n"
                "- YAML frontmatterの構造変更\n"
                "\n"
                "**MAJOR**（重要、意味・用語レベル）：\n"
                "- 翻訳漏れ（日本語が残存）\n"
                "- 用語辞書違反\n"
                "- 意味の誤訳\n"
                "- 不自然な表現\n"
                "- **誤字・同音異義語エラー**（例: 中国語で「分钟配」→「分配」、「没有分钟配」→「未分配」など）\n"
                "- 文法エラー\n"
                "- 不自然な文字列の組み合わせ（存在しない単語）\n"
                "\n"
                "**MINOR**（改善提案）：\n"
                "- 表現の改善提案\n"
                "- より自然な言い回し\n"
                "\n"
                "【審査項目】\n"
                "- 技術的正確性と安全警告、操作手順の完全性\n"
                "- 用語一貫性（辞書参照）、{target_language}の常用表現に適合\n"
                "- 原始Markdown構造と記号の厳格保持\n"
                "- 原文のすべての改行文字と空白文字の完全不変\n"
                "- 不翻訳規則の遵守（コードブロック/行内コード/URL/メール/プレースホルダー/ファイル名、既存英語用語等）\n"
                "- **重要チェック項目**：翻訳結果に日本語文字（平仮名・片仮名）が残存していないか\n"
                "- **品質チェック**：誤字、同音異義語エラー、不自然な表現、文法エラーがないか\n"
                "  * 特に中国語の場合: 「分钟配」のような誤字、存在しない単語の組み合わせに注意\n"
                + japanese_warning +
                "\n\n原文（日本語）：\n{original}\n\n訳文（{target_language}）：\n{translated}\n\n"
                "【出力フォーマット】\n"
                "各問題を以下の形式で出力してください：\n"
                "[SEVERITY: BLOCKER/MAJOR/MINOR] [CATEGORY: format/translation/terminology/link] 問題の説明\n"
                "修正案: 具体的な修正内容\n"
                "---\n"
            )
            
            prompt = PromptTemplate(
                input_variables=["original", "translated", "target_language"],
                template=review_template
            )
            
            # Get LLM client
            llm_result = self._get_llm_client("review")
            if isinstance(llm_result, tuple):
                llm, model_name = llm_result
            else:
                llm = llm_result
                model_name = None
            
            # Execute review
            review_text = self._execute_review_request(llm, prompt, original_content[:2000], translated_content[:2000], model_name, japanese_detection)
            
            # Parse review text to extract issues
            issues = []
            has_blockers = False
            has_majors = False
            
            # Simple parsing: look for [SEVERITY: ...] patterns
            severity_pattern = re.compile(r'\[SEVERITY:\s*(BLOCKER|MAJOR|MINOR)\]\s*\[CATEGORY:\s*(\w+)\]\s*([^\n]+)')
            for match in severity_pattern.finditer(review_text):
                severity = match.group(1)
                category = match.group(2)
                description = match.group(3).strip()
                
                # Try to find suggestion (next line after "修正案:")
                suggestion = ""
                pos = match.end()
                remaining = review_text[pos:pos+500]
                if "修正案:" in remaining or "修正案：" in remaining:
                    suggestion_match = re.search(r'修正案[：:]\s*([^\n]+)', remaining)
                    if suggestion_match:
                        suggestion = suggestion_match.group(1).strip()
                
                issues.append({
                    "severity": severity,
                    "category": category,
                    "description": description,
                    "suggestion": suggestion
                })
                
                if severity == "BLOCKER":
                    has_blockers = True
                elif severity == "MAJOR":
                    has_majors = True
            
            return {
                "issues": issues,
                "has_blockers": has_blockers,
                "has_majors": has_majors,
                "review_text": review_text
            }
        except Exception as e:
            print(f"Error in _review_with_severity: {e}")
            return {
                "issues": [],
                "has_blockers": False,
                "has_majors": False,
                "review_text": f"Review failed: {str(e)}"
            }
    
    def _fix_translation_issues(self, translated_content: str, original_content: str, 
                               issues: List[Dict[str, Any]]) -> str:
        """Fix translation based on review issues
        
        Args:
            translated_content: Current translation
            original_content: Original content
            issues: List of issues
            
        Returns:
            Fixed translation
        """
        try:
            print("修正翻訳中...")

            if not issues:
                return translated_content

            # Build the issues list once; passed to each segment.
            issues_text = ""
            for i, issue in enumerate(issues, 1):
                issues_text += f"{i}. [{issue['severity']}] {issue['description']}\n"
                if issue.get('suggestion'):
                    issues_text += f"   修正案: {issue['suggestion']}\n"

            # Backend (use the translation provider/model for fixing)
            fix_provider, _ = self._resolve_provider_and_model("translate")
            if not self._provider_ready(fix_provider):
                self.logger.warning(f"Fix backend not ready (provider={fix_provider or 'unset'}); skipping fix")
                return translated_content
            llm_result = self._get_llm_client("translate")
            if isinstance(llm_result, tuple):
                llm, model_name = llm_result
            else:
                llm, model_name = llm_result, None

            # Segment-based, placeholder-free fix: mask structure, then proofread only
            # the target-language text between placeholders, paragraph by paragraph.
            masked_content, fix_placeholders = self._mask_non_translatable(translated_content)
            # Process segments that contain target-language text (letters or CJK).
            target_text_re = re.compile(r'[A-Za-z぀-ヿ一-鿿]')
            memo: Dict[str, str] = {}
            should_fix = lambda t: bool(target_text_re.search(t))
            fix_core = lambda seg: self._invoke_paragraph_fix(seg, issues_text, llm, model_name)
            fixed_masked = self._map_masked_segments(masked_content, should_fix, fix_core, memo)

            # Restore masked GitBook/Markdown structure
            fixed_content = self._restore_placeholders(fixed_masked, fix_placeholders)

            print("修正完了")
            return fixed_content
        except Exception as e:
            self.logger.error(f"Error in _fix_translation_issues: {e}", exc_info=True)
            print(f"Error in _fix_translation_issues: {e}")
            return translated_content
    
    def review_translation(self, file_path: str, original_file_path: str = None) -> Dict[str, Any]:
        """使用LLM对翻译结果进行审核"""
        try:
            # レビューバックエンドが利用可能か確認（利用不可でも基礎改善は実施）
            review_provider, _ = self._resolve_provider_and_model("review")
            if not self._provider_ready(review_provider):
                print(f"警告：レビューバックエンドが利用できません（provider={review_provider or 'unset'}）。基礎的な改善のみ実施します")

                with open(file_path, 'r', encoding='utf-8') as f:
                    translated_content = f.read()
                
                # 只应用基础改进
                improved_content = self._apply_review_suggestions(translated_content, "")
                
                # 保存改进后的翻译结果
                # 将review后的文件命名为与原文件相同的名称加上翻译语言
                name_without_ext, ext = os.path.splitext(os.path.basename(file_path))
                base_dir = os.path.dirname(file_path)
                parent_dir = os.path.dirname(base_dir) if os.path.basename(base_dir).lower() == 'temp' else base_dir
                improved_file_name = f"{name_without_ext.replace('_temp', '')}{ext}"
                improved_file_path = os.path.join(parent_dir, improved_file_name)
                with open(improved_file_path, 'w', encoding='utf-8') as f:
                    f.write(improved_content)
                
                return {
                    "status": "basic_improved",
                    "reason": "Only basic improvements applied without OpenAI API key",
                    "improved_file": improved_file_path
                }
            
            with open(file_path, 'r', encoding='utf-8') as f:
                translated_content = f.read()
            
            # 执行日语检测
            japanese_detection = self._detect_japanese_in_translatable_text(translated_content)
            if japanese_detection["has_japanese"]:
                print(f"⚠️ 警告: 翻译结果中检测到日语（{japanese_detection['count']}个字符）")
                print(f"   样本: {', '.join(japanese_detection['samples'])}")
            else:
                print("✓ 翻译结果中未检测到日语")
            
            # 查找对应的原始文件
            base_name = os.path.basename(file_path)
            original_file_name = base_name.replace(f"_{self.target_language}", "").replace("_temp", "")
            original_content = self._find_original_file(original_file_name)
            
            # Review with severity classification and fix loop (max 2 iterations)
            max_iterations = 2
            current_iteration = 0
            final_review_result = None
            
            while current_iteration < max_iterations:
                print(f"\n{'='*60}")
                print(f"レビュー反復 {current_iteration + 1}/{max_iterations}")
                print(f"{'='*60}")
                
                review_result = self._review_with_severity(translated_content, original_content, japanese_detection)
                final_review_result = review_result
                
                # Count issues by severity
                blocker_count = len([i for i in review_result['issues'] if i['severity'] == 'BLOCKER'])
                major_count = len([i for i in review_result['issues'] if i['severity'] == 'MAJOR'])
                minor_count = len([i for i in review_result['issues'] if i['severity'] == 'MINOR'])
                
                print(f"検出された問題: BLOCKER={blocker_count}, MAJOR={major_count}, MINOR={minor_count}")
                
                if not review_result["has_blockers"] and not review_result["has_majors"]:
                    print("✓ BLOCKER/MAJOR問題なし、レビュー完了")
                    break
                
                # Need to fix
                print(f"修正が必要な問題を検出、修正を実行中...")
                
                # Filter BLOCKER and MAJOR issues for fixing
                critical_issues = [i for i in review_result['issues'] if i['severity'] in ['BLOCKER', 'MAJOR']]
                
                # Fix translation
                translated_content = self._fix_translation_issues(
                    translated_content,
                    original_content,
                    critical_issues
                )
                
                current_iteration += 1
                
                # Re-detect Japanese after fix
                japanese_detection = self._detect_japanese_in_translatable_text(translated_content)
            
            # 保存审核结果到txt文件
            review_file_path = file_path.replace('.md', '_review.txt')
            with open(review_file_path, 'w', encoding='utf-8') as f:
                f.write(final_review_result["review_text"])
                f.write("\n\n" + "="*60 + "\n")
                f.write("問題サマリー:\n")
                for issue in final_review_result["issues"]:
                    f.write(f"[{issue['severity']}] [{issue['category']}] {issue['description']}\n")
                    if issue['suggestion']:
                        f.write(f"  修正案: {issue['suggestion']}\n")
            print(f"已保存审核结果到: {review_file_path}")
            
            # Use the fixed content as improved content
            improved_content = translated_content
            
            # 保存改进后的翻译结果
            # 将review后的文件命名为与原文件相同的名称加上翻译语言
            name_without_ext, ext = os.path.splitext(os.path.basename(file_path))
            base_dir = os.path.dirname(file_path)
            parent_dir = os.path.dirname(base_dir) if os.path.basename(base_dir).lower() == 'temp' else base_dir
            
            # Generate final filename based on output naming convention
            if self.output_naming == "suffix":
                # Suffix mode: intro.en.md, intro.zh-CN.md
                base_name_clean = name_without_ext.replace('_temp', '')
                # Remove existing language suffix if present
                for lang_suffix in ['.en', '.zh-CN', '.zh-TW', '.zh_cn', '.zh_tw']:
                    if base_name_clean.endswith(lang_suffix):
                        base_name_clean = base_name_clean[:-len(lang_suffix)]
                        break
                improved_file_name = f"{base_name_clean}.{self.target_label}{ext}"
            else:
                # Directory mode: keep original filename
                improved_file_name = f"{name_without_ext.replace('_temp', '')}{ext}"
            
            improved_file_path = os.path.join(parent_dir, improved_file_name)
            with open(improved_file_path, 'w', encoding='utf-8') as f:
                f.write(improved_content)
            
            print(f"Saved improved translation to: {improved_file_path}")

            # Commit improved translation back to original repo path (prefer original mapping)
            # try:
            #     mapped_repo_path = None
            #     mapped_branch = None
            #     if original_file_path:
            #         mapped_repo_path = self.local_to_repo_path.get(original_file_path)
            #         mapped_branch = self.local_to_branch.get(original_file_path)
            #     if not mapped_repo_path or not mapped_branch:
            #         base_dir_for_rel = os.path.join(self.translated_dir, self.target_language)
            #         rel = os.path.relpath(parent_dir, base_dir_for_rel)
            #         parts = rel.split(os.sep) if rel else []
            #         mapped_branch = mapped_branch or (parts[0] if parts else "main")
            #         repo_rel_path = os.sep.join(parts[1:]) if len(parts) > 1 else ""
            #         # when branch-root save (no branch folder), repo_rel_path may be empty; use improved filename only
            #         mapped_repo_path = mapped_repo_path or os.path.join(repo_rel_path, improved_file_name).replace("\\", "/")
            #     else:
            #         mapped_repo_path = os.path.join(os.path.dirname(mapped_repo_path), improved_file_name).replace("\\", "/")
            #     self._update_github_file(mapped_repo_path, mapped_branch, improved_content, f"Update translation ({self.target_language})")
            # except Exception as commit_err:
            #     print(f"Failed to commit improved translation: {commit_err}")

            # # Commit improved translation back to original repo path
            # try:
            #     base_dir_for_rel = os.path.join(self.translated_dir, self.target_language)
            #     rel = os.path.relpath(parent_dir, base_dir_for_rel)
            #     parts = rel.split(os.sep) if rel else []
            #     commit_branch = parts[0] if parts else "main"
            #     repo_rel_path = os.sep.join(parts[1:]) if len(parts) > 1 else ""
            #     repo_path = os.path.join(repo_rel_path, improved_file_name).replace("\\", "/")
            #     self._update_github_file(repo_path, commit_branch, improved_content, f"Update translation ({self.target_language})")
            # except Exception as commit_err:
            #     print(f"Failed to commit improved translation: {commit_err}")
            
            # Validate and fix GitBook structure
            self.logger.info("Validating GitBook structure...")
            if original_file_path and os.path.exists(original_file_path):
                validation_result = self._validate_translation_structure(
                    original_file=original_file_path,
                    translated_file=improved_file_path
                )
                
                if validation_result.get("has_differences"):
                    self.logger.warning(f"Structure validation found {len(validation_result['differences'])} issues")
                else:
                    self.logger.info("Structure validation passed")
            else:
                self.logger.warning("Original file not found, skipping structure validation")
                validation_result = None
            
            return {
                "status": "success",
                "review_file": review_file_path,
                "review_content": final_review_result["review_text"] if final_review_result else "",
                "improved_file": improved_file_path,
                "structure_validation": validation_result
            }
        except Exception as e:
            print(f"Error reviewing translation file {file_path}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _find_original_file(self, original_file_name: str) -> str:
        """Find original file content"""
        original_content = ""
        
        # Look in download directory first
        if os.path.exists(self.download_dir):
            download_original_path = os.path.join(self.download_dir, original_file_name)
            if os.path.exists(download_original_path):
                with open(download_original_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
        
        # If not found, search in temp repo directory
        if not original_content and os.path.exists(self.temp_dir):
            for root, _, files in os.walk(self.temp_dir):
                if original_file_name in files:
                    with open(os.path.join(root, original_file_name), 'r', encoding='utf-8') as f:
                        original_content = f.read()
                    break
        
        return original_content

    def _push_translations_to_github(self, translated_files: List[Dict[str, str]]) -> Dict[str, Any]:
        """Push translations to GitHub
        
        Args:
            translated_files: [{"local_path": "...", "repo_path": "...", "branch": "..."}]
            
        Returns:
            {
                "success": bool,
                "pushed_files": int,
                "failed_files": List[str],
                "message": str
            }
        """
        if self.push_option == "none":
            return {"success": True, "pushed_files": 0, "message": "Push disabled"}
        
        if self.push_option == "push_same_repo_direct":
            # Dangerous operation, show warning
            print("⚠️ WARNING: push_same_repo_direct will directly push to the repository!")
            print("This may overwrite existing files.")
            
            pushed_count = 0
            failed_files = []
            
            for file_info in translated_files:
                try:
                    with open(file_info["local_path"], 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    success = self._update_github_file(
                        file_info["repo_path"],
                        file_info["branch"],
                        content,
                        f"Update translation: {file_info['repo_path']}"
                    )
                    
                    if success:
                        pushed_count += 1
                        print(f"✓ Pushed: {file_info['repo_path']}")
                    else:
                        failed_files.append(file_info["local_path"])
                        print(f"✗ Failed: {file_info['local_path']}")
                except Exception as e:
                    print(f"Failed to push {file_info['local_path']}: {e}")
                    failed_files.append(file_info["local_path"])
            
            return {
                "success": len(failed_files) == 0,
                "pushed_files": pushed_count,
                "failed_files": failed_files,
                "message": f"Pushed {pushed_count} files, {len(failed_files)} failed"
            }
        
        return {"success": False, "pushed_files": 0, "message": f"Unknown push_option: {self.push_option}"}
    
    def _update_github_file(self, repo_path: str, branch_name: str, new_content: str, commit_message: str) -> bool:
        """Update a file in GitHub repository at the same path on the specified branch"""
        try:
            if not self.repo:
                if not self.connect_to_github_repo():
                    print("Error: failed to connect to GitHub for commit")
                    return False
            content_obj = self.repo.get_contents(repo_path, ref=branch_name)
            old_content = content_obj.decoded_content.decode('utf-8', errors='ignore')
            if old_content == new_content:
                print(f"Skip commit (content identical): {repo_path}")
                return True
            print(f"Committing update: {repo_path} (branch: {branch_name})")
            result = self.repo.update_file(
                path=repo_path,
                message=commit_message,
                content=new_content,
                sha=content_obj.sha,
                branch=branch_name
            )
            print(f"Commit succeeded: {repo_path} @ {result['commit'].sha}")
            return True
        except Exception as e:
            print(f"Commit failed for {repo_path}: {e}")
            return False
    
    def _execute_review_request(self, llm, prompt, original_content, translated_content, model_name=None, japanese_detection=None):
        """执行审核请求的统一方法
        
        Args:
            llm: LLM客户端
            prompt: 提示模板
            original_content: 原文内容
            translated_content: 已翻译内容
            model_name: 模型名称
            japanese_detection: 日语检测信息（可选）
        """
        try:
            if hasattr(llm, 'invoke'):
                # 根据提示模板的输入变量决定如何格式化
                if self.target_language.lower() == "zh_tw" and "target_language" not in prompt.input_variables:
                    result = llm.invoke(
                        prompt.format(
                            original=original_content,
                            translated=translated_content
                        )
                    )
                else:
                    result = llm.invoke(
                        prompt.format(
                            original=original_content,
                            translated=translated_content,
                            target_language=self.target_language
                        )
                    )
                if hasattr(result, 'content'):
                    return result.content
                else:
                    return str(result)
            elif hasattr(llm, 'chat') and hasattr(llm.chat, 'completions'):
                # 直接使用OpenAI客户端兼容的API
                # 对于兼容API模型，使用更严格的system prompt约束输出
                # 包含日语检测信息
                japanese_warning = ""
                if japanese_detection and japanese_detection.get("has_japanese"):
                    japanese_warning = (
                        f"\n\n⚠️ 重要警告：翻译结果中检测到{japanese_detection['count']}个日语字符（平假名・片假名）。"
                        f"检测样本: {', '.join(japanese_detection['samples'])}。"
                        f"这些必须翻译为{self.target_language}。审核时必须明确指出。"
                    )
                
                if self.target_language.lower() == "zh_tw":
                    system_content = (
                        "你是资深本地化审核专家。严格审核翻译的技术准确性、术语一致性与Markdown结构保留。"
                        "不得翻译代码块/行内代码/URL/邮箱/占位符/文件名、已有英文术语。"
                        "**重要：对于Markdown链接[文本](URL)，圆括号内的URL必须完全保持不变，不得修改、翻译或替换。"
                        "如果URL是__LINK_URL_N__或__IMAGE_URL_N__形式的占位符，必须原样保留。"
                        "同时要求保持原文所有换行符与空白字符完全不变。"
                        "**重要检查项**：必须检查翻译结果中是否残留日语文字（平假名・片假名），如有残留必须明确指出并提供正确翻译。"
                        "仅输出审核问题与修改建议，不要重写全文，不要添加任何额外说明。"
                        + japanese_warning
                    )
                    user_message = (
                        f"原文(日语):\n{original_content}\n\n"
                        f"翻译(繁体中文):\n{translated_content}\n\n"
                        "请列出问题并给出修改建议。"
                    )
                else:
                    system_content = (
                        "你是资深本地化审核专家。严格审核翻译的技术准确性、术语一致性与Markdown结构保留。"
                        "不得翻译代码块/行内代码/URL/邮箱/占位符/文件名、已有英文术语。"
                        "**重要：对于Markdown链接[文本](URL)，圆括号内的URL必须完全保持不变，不得修改、翻译或替换。"
                        "如果URL是__LINK_URL_N__或__IMAGE_URL_N__形式的占位符，必须原样保留。"
                        "同时要求保持原文所有换行符与空白字符完全不变。"
                        "**重要检查项**：必须检查翻译结果中是否残留日语文字（平假名・片假名），如有残留必须明确指出并提供正确翻译。"
                        "仅输出审核问题与修改建议，不要重写全文，不要添加任何额外说明。"
                        + japanese_warning
                    )
                    user_message = (
                        f"原文(日语):\n{original_content}\n\n"
                        f"翻译({self.target_language}):\n{translated_content}\n\n"
                        "请列出问题并给出修改建议。"
                    )

                if model_name and model_name.startswith("claude"):
                    # Claude等模型的system prompt保持上述严格约束
                    pass
                
                response = llm.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0
                )
                return response.choices[0].message.content
            else:
                # 兼容旧版本
                if LLMChain:
                    chain = LLMChain(llm=llm, prompt=prompt)
                    return chain.run(
                        original=original_content,
                        translated=translated_content,
                        target_language=self.target_language
                    )
                else:
                    # Fallback: use invoke directly
                    result = llm.invoke(
                        prompt.format(
                            original=original_content,
                            translated=translated_content,
                            target_language=self.target_language
                        )
                    )
                    if hasattr(result, 'content'):
                        return result.content
                    else:
                        return str(result)
        except Exception as e:
            print(f"审核请求失败: {e}")
            print(f"错误详情: {type(e).__name__}: {str(e)}")
            # 回退到Ollama（审核）
            try:
                formatted_prompt = None
                if hasattr(prompt, "input_variables") and "target_language" not in getattr(prompt, "input_variables", []):
                    formatted_prompt = prompt.format(
                        original=original_content,
                        translated=translated_content
                    )
                else:
                    formatted_prompt = prompt.format(
                        original=original_content,
                        translated=translated_content,
                        target_language=self.target_language
                    )
                self.logger.warning("Using Ollama fallback for review")
                return self._ollama_invoke_text(
                    formatted_prompt,
                    failed_model_name=model_name,
                    task_type="review"
                )
            except Exception as ollama_error:
                print(f"Ollama回退审核也失败: {ollama_error}")
                return "审核失败，无法提供审核意见。"
            
    def clean_up(self):
        """清理临时文件"""
        def safe_rmtree(path):
            if os.path.exists(path):
                retry_count = 3
                retry_delay = 1
                for i in range(retry_count):
                    try:
                        shutil.rmtree(path)
                        print(f"{path} 已清理")
                        break
                    except PermissionError as e:
                        if i < retry_count - 1:
                            print(f"清理 {path} 时权限错误，{retry_delay}秒后重试 ({i+1}/{retry_count}): {e}")
                            time.sleep(retry_delay)
                        else:
                            print(f"多次尝试后仍无法清理 {path}: {e}")
                    except Exception as e:
                        print(f"清理 {path} 时出错: {e}")
                        break
        
        safe_rmtree(self.temp_dir)
        # safe_rmtree(self.download_dir)

    def run(self):
        """运行整个翻译流程（複数言語対応）"""
        try:
            md_files = []
            
            print("从GitHub获取MD文件")
            all_md_files = self.download_md_files()
            
            if not all_md_files:
                return {"success": False, "message": "从GitHub下载MD文件失败"}
            
            # Glob pattern filtering
            if self.target_paths:
                md_files = self._match_files_by_glob(all_md_files, self.target_paths)
                print(f"Glob pattern '{self.target_paths}' matched {len(md_files)}/{len(all_md_files)} files")
            else:
                md_files = all_md_files
            
            if not md_files:
                return {"success": False, "message": "No files matched the pattern"}
            
            print(f"找到 {len(md_files)} 个MD文件")
            
            all_results = {}
            
            # Multiple languages loop
            for target_lang in self.target_languages:
                print(f"\n{'='*60}")
                print(f"開始翻訳到 {target_lang}")
                print(f"{'='*60}\n")
                
                # Temporarily set current language
                original_target_language = self.target_language
                original_target_label = self.target_label
                original_dictionary = self.dictionary
                
                self.target_language = target_lang
                self.target_label = self._normalize_target_label(target_lang)
                self.dictionary = self.full_dictionary.get(self.target_label, {})
                
                results = []
                translated_files_for_push = []
                
                for file_path in md_files:
                    print(f"Translating to {target_lang}: {file_path}")
                    translated_file = self.translate_file(file_path)
                    if translated_file:
                        review_result = self.review_translation(translated_file, original_file_path=file_path)
                        results.append({
                            "original_file": file_path,
                            "translated_file": translated_file,
                            "review_result": review_result,
                            "language": target_lang
                        })
                        
                        # Collect info for push
                        if review_result.get("improved_file"):
                            repo_path = self.local_to_repo_path.get(file_path, "")
                            branch = self.local_to_branch.get(file_path, "main")
                            
                            # Adjust repo_path based on output naming
                            if self.output_naming == "suffix":
                                base, ext = os.path.splitext(repo_path)
                                repo_path = f"{base}.{self.target_label}{ext}"
                            else:
                                repo_path = f"{self.target_label}/{repo_path}"
                            
                            translated_files_for_push.append({
                                "local_path": review_result["improved_file"],
                                "repo_path": repo_path,
                                "branch": branch
                            })
                
                all_results[target_lang] = results
                
                # GitHub push (if configured)
                if self.push_option != "none" and translated_files_for_push:
                    print(f"\n{'='*60}")
                    print(f"Pushing translations for {target_lang} to GitHub...")
                    print(f"{'='*60}\n")
                    push_result = self._push_translations_to_github(translated_files_for_push)
                    all_results[target_lang + "_push_result"] = push_result
                
                # Restore original settings
                self.target_language = original_target_language
                self.target_label = original_target_label
                self.dictionary = original_dictionary
            
            return {"success": True, "results": all_results}
        finally:
            self.clean_up()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="翻译智能体")
    parser.add_argument("--config-file", type=str, default="./config.ini", help="配置文件路径")
    parser.add_argument("--file", type=str, help="直接翻訳するMarkdownファイルのパス")
    parser.add_argument("--target-lang", type=str, help="翻訳先言語 (en, zh-CN, zh-TW)")
    parser.add_argument("--output", type=str, help="翻訳結果の出力先ディレクトリ")
    
    args = parser.parse_args()
    
    # ローカルファイルを直接翻訳する場合
    if args.file:
        import os
        
        if not os.path.exists(args.file):
            print(f"エラー：ファイルが見つかりません: {args.file}")
            exit(1)
        
        # 翻訳先言語を決定
        target_lang = args.target_lang if args.target_lang else None
        
        # 出力先ディレクトリを決定
        if args.output:
            output_dir = args.output
        else:
            # デフォルト：元ファイルと同じディレクトリ
            output_dir = os.path.dirname(os.path.abspath(args.file))
        
        print(f"翻訳開始: {args.file}")
        print(f"翻訳先言語: {target_lang or 'config.iniから取得'}")
        print(f"出力先: {output_dir}")
        print("="*60)
        
        # TranslationAgentを初期化
        agent = TranslationAgent(
            target_language=target_lang,
            config_file=args.config_file
        )
        
        # 出力ディレクトリを設定
        if args.output:
            agent.translated_dir = output_dir
        
        # 翻訳実行
        print(f"\n翻訳中...")
        translated_file = agent.translate_file(args.file)
        
        if translated_file:
            print(f"✓ 翻訳完了: {translated_file}")
            
            # レビュー実行
            print(f"\nレビュー中...")
            review_result = agent.review_translation(translated_file, args.file)
            
            if review_result.get("status") == "success":
                print(f"✓ レビュー完了: {review_result.get('review_file')}")
                if review_result.get("improved_file"):
                    print(f"✓ 改善版ファイル: {review_result.get('improved_file')}")
                
                # 構造検証結果
                if review_result.get("structure_validation"):
                    validation = review_result["structure_validation"]
                    if validation.get("has_differences"):
                        print(f"⚠ 構造検証: {len(validation.get('differences', []))} 件の差異を検出")
                    else:
                        print(f"✓ 構造検証: 問題なし")
            else:
                print(f"⚠ レビュー状態: {review_result.get('status')}")
            
            print("\n" + "="*60)
            print("翻訳処理が完了しました")
        else:
            print("✗ 翻訳失敗")
            exit(1)
        
        exit(0)
    
    # GitHubリポジトリから翻訳する場合（従来の動作）
    agent = TranslationAgent(
        config_file=args.config_file
    )
    
    # 检查是否提供了必要的GitHub仓库信息
    if not agent.github_repo:
        print("错误：未在命令行参数或配置文件中提供GitHub仓库URL")
        print("\n使用方法:")
        print("  1. GitHubリポジトリから翻訳: config.iniにrepoを設定して実行")
        print("  2. ローカルファイルを翻訳: python main.py --file <file.md> [--target-lang <lang>]")
        parser.print_help()
        exit(1)
    
    result = agent.run()
    
    if result["success"]:
        print("翻译任务完成！")
        # results is keyed by target language: {lang: [items...], lang+"_push_result": {...}}
        for lang, items in result["results"].items():
            if lang.endswith("_push_result"):
                print(f"\n[{lang}] {items.get('message', '')} (pushed: {items.get('pushed_files', 0)})")
                continue
            print(f"\n=== {lang} ===")
            for item in items:
                print(f"- 翻译文件: {item['translated_file']}")
                if item['review_result']['status'] == 'success':
                    print(f"  审核文件: {item['review_result']['review_file']}")
                    if 'improved_file' in item['review_result']:
                        print(f"  改进文件: {item['review_result']['improved_file']}")
                elif item['review_result']['status'] == 'basic_improved':
                    print(f"  基础改进文件: {item['review_result']['improved_file']}")
                    print(f"  状态: {item['review_result']['reason']}")
                else:
                    print(f"  审核状态: {item['review_result']['status']}")
    else:
        print(f"翻译任务失败: {result['message']}")
