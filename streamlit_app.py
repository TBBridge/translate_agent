import streamlit as st
import os
import sys
from main import TranslationAgent
import tempfile
from dotenv import load_dotenv

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 加载 .env 文件中的环境变量
load_dotenv()

# 设置页面配置
st.set_page_config(
    page_title="i-Reporter Translation Agent",
    page_icon="./i-Repo_logo_symbol.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 页面标题
st.title("i-Reporter Translation Agent")

# 创建一个表单来放置所有配置项
with st.form(key="translation_config_form"):
    # 主要配置部分
    st.header("Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        # GitHub配置
        st.subheader("GitHub Configuration")
        # github_token = st.text_input(
        #     "GitHub Access Token",
        #     help="Personal access token for GitHub API",
        #     type="password"
        # )
        
        github_repo = st.text_input(
            "GitHub Repository",
            value="CIMTOPSCORP/user-manual",
            help="Format can be https://github.com/owner/repo or owner/repo"
        )
        
        # 分支配置
        branches = st.selectbox(
            "Branches to Process", 
            options=["1.system-requirements"
                    ,"2.start-guide"
                    ,"3.environment-configuration"
                    ,"4.basicoperation"
                    ,"5.create-a-form"
                    ,"6.clustertype_settingprocedure"
                    ,"7.input-features"
                    ,"8.custom-menu"
                    ,"9.output-the-data"
                    ,"10.administrator-functions"
                    ,"11.server-construction-maintenance"
                    ,"12.connection-with-externaldevices"
                    ,"13.optional-features"
                    ,"14.when-using"
                    ,"15.for-support"],
            help="Comma-separated list of branches to process"
        )
    
    with col2:
        # 翻译配置
        st.subheader("Translation Configuration")
        target_language = st.selectbox(
            "Target Language",
            options=["en", "zh_cn", "zh_tw", "kr", "th", "vi"],
            format_func=lambda x: {"en": "English", "zh_cn": "简体中文", "zh_tw": "繁体中文", "kr": "한국어", "th": "ไทย", "vi": "Tiếng Việt"}[x]
        )
        
        dictionary_file = st.text_input(
            "Dictionary File Path", 
            value="dictionary.txt", 
            help="Absolute or relative path to the dictionary file"
        )
        
        max_file_size = st.number_input(
            "Max File Size Limit (KB)", 
            min_value=100, 
            max_value=10240, 
            value=1024, 
            help="Unit is KB, default is 1024KB (1MB)"
        )
    
    # LLM配置
    st.header("LLM Configuration")
    col3, col4 = st.columns(2)
    
    with col3:
        translation_model_name = st.selectbox(
            "LLM Model Name",
            options=["gpt-4o", "claude-3.7-sonnet-20240620", "qwen-plus", "qwen-max"],
            help="Translation task priority uses Qwen"
        )
        
    with col4:
        review_model_name = st.selectbox(
            "Review LLM Model Name",
            options=["gpt-4o", "claude-3.7-sonnet-20240620", "qwen-plus", "qwen-max"],
            help="Review task priority uses Qwen"
        )
        
        
    
    # 目录配置
    st.header("Directory Configuration")
    default_dir = os.path.abspath(os.path.dirname(__file__))
    download_dir = st.text_input(
        "Download Directory", 
        value=os.path.join(default_dir, "downloaded_files"),
        help="Downloaded Markdown files will be saved in this directory"
    )
    
    translated_dir = st.text_input(
        "Translated Results Directory", 
        value=os.path.join(default_dir, "translated_results"),
        help="Translated files will be saved in this directory"
    )
    
    temp_dir = st.text_input(
        "Temp Directory", 
        value=os.path.join(default_dir, "temp"),
        help="Temporary files will be saved in this directory"
    )
    
    # 执行按钮
    submit_button = st.form_submit_button(
        label="Run Translation", 
        help="Click to start the translation process",
        type="primary"
    )

# 执行区域
if submit_button:
    # 验证必填参数
    if not github_repo:
        st.error("Please enter the GitHub repository address")
    
    else:
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.ini', delete=False, encoding='utf-8') as temp_config:
            # 写入配置
            # 从环境变量中获取隐私值
            github_token = os.getenv('GITHUB_TOKEN', '')
            openai_api_key = os.getenv('OPENAI_API_KEY', '')
            qwen_api_key = os.getenv('QWEN_API_KEY', '')
            
            temp_config.write("[github]\n")
            temp_config.write(f"token={github_token}\n")
            temp_config.write(f"repo={github_repo}\n")
            temp_config.write(f"branch={branches}\n\n")
            
            temp_config.write("[translation]\n")
            temp_config.write(f"target_language={target_language}\n")
            temp_config.write(f"dictionary_file={dictionary_file}\n")
            temp_config.write(f"max_file_size={max_file_size * 1024}\n\n")  # 转换为字节
            
            temp_config.write("[llm]\n")
            temp_config.write(f"openai_api_key={openai_api_key}\n")
            temp_config.write(f"qwen_api_key={qwen_api_key}\n")
            temp_config.write(f"model_name={translation_model_name}\n\n")
            
            temp_config.write("[directories]\n")
            temp_config.write(f"download_dir={download_dir}\n")
            temp_config.write(f"translated_dir={translated_dir}\n")
            temp_config.write(f"temp_dir={temp_dir}\n\n")
            
            temp_config.write("[branches]\n")
            temp_config.write(f"branches={branches}\n")
            
            temp_config_path = temp_config.name
        
        try:
            # 创建进度栏
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 显示执行信息
            status_text.text("Initializing translation agent...")
            
            # 创建翻译代理
            agent = TranslationAgent(config_file=temp_config_path)
            print("***", agent.config)
            
            # 显示执行信息
            status_text.text("Running translation process...")
            progress_bar.progress(10)
            
            # 执行翻译
            result = agent.run()
            
            progress_bar.progress(100)
            status_text.text("Translation process completed")
            
            # 显示结果
            if result["success"]:
                st.success("Translation task completed successfully!")
                
                # 展开的结果详情
                with st.expander("View detailed results"):
                    for item in result["results"]:
                        st.markdown(f"### Translated file: {item['translated_file']}")
                        
                        if item['review_result']['status'] == 'success':
                            st.info(f"Reviewed file: {item['review_result']['review_file']}")
                            if 'improved_file' in item['review_result']:
                                st.info(f"Improved file: {item['review_result']['improved_file']}")
                        elif item['review_result']['status'] == 'basic_improved':
                            st.info(f"Basic improved file: {item['review_result']['improved_file']}")
                            st.info(f"Status: {item['review_result']['reason']}")
                        else:
                            st.warning(f"Review status: {item['review_result']['status']}")
            else:
                st.error(f"Translation task failed: {result['message']}")
                
        except Exception as e:
            st.error(f"An error occurred during the translation process: {str(e)}")
        finally:
            # 清理临时配置文件
            try:
                os.remove(temp_config_path)
            except:
                pass

# 侧边栏说明
with st.sidebar:
    st.header("使用説明")
    st.markdown("この翻訳エージェントは、GitHubリポジトリ内のMarkdownファイルを自動的に指定の言語に翻訳するのをサポートします。")
    st.markdown("### 主な機能:")
    st.markdown("- GitHubリポジトリに接続してMarkdownファイルをダウンロード")
    st.markdown("- 大言語モデル(LLM)を使用して翻訳を実行")
    st.markdown("- 差分翻訳をサポートしているため，変更部分のみ翻訳可能")
    st.markdown("- 翻訳結果をレビューして改善する機能を提供")
    st.markdown("- 翻訳履歴を保存して管理")
    
    st.header("注意事項")
    st.markdown("- GitHubリポジトリのアドレスを正しく入力してください")
    st.markdown("- 大きなファイルを翻訳する場合、時間がかかることがあります。ご了承ください")
    st.markdown("- 専門用語の辞書ファイルの形式は: 用語=翻訳")
