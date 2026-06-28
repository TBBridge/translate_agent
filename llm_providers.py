import os
from typing import Any, Optional

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None  # type: ignore

try:
    from langchain_anthropic import ChatAnthropic
except Exception:
    ChatAnthropic = None  # type: ignore

try:
    from langchain_community.chat_models import ChatOllama
except Exception:
    ChatOllama = None  # type: ignore

# Gemini (Google Generative AI) via LangChain
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None  # type: ignore


def create_langchain_llm(provider: str, model_name: str, api_keys: dict) -> Optional[Any]:
    """
    返回可用于 LangChain 的 LLM 实例。
    - openai/deepseek/dashscope：优先使用 ChatOpenAI（支持 base_url）
    - claude：使用 ChatAnthropic
    - ollama：使用 ChatOllama
    - gemini：使用 ChatGoogleGenerativeAI
    如果缺少依赖或密钥，返回 None。
    """
    provider = (provider or "openai").lower()
    model_name = model_name or "gpt-3.5-turbo"

    if provider in ["openai", "deepseek", "dashscope"]:
        if ChatOpenAI is None:
            return None
        base_url = None
        api_key = None
        if provider == "openai":
            api_key = api_keys.get("openai") or os.environ.get("OPENAI_API_KEY")
        elif provider == "deepseek":
            api_key = api_keys.get("deepseek") or os.environ.get("DEEPSEEK_API_KEY")
            base_url = "https://api.deepseek.com/v1"
        elif provider == "dashscope":
            api_key = api_keys.get("dashscope") or os.environ.get("DASHSCOPE_API_KEY")
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not api_key:
            return None
        # ChatOpenAI 在新版本中支持 base_url
        return ChatOpenAI(model_name=model_name, temperature=0, api_key=api_key, base_url=base_url)

    if provider == "claude":
        if ChatAnthropic is None:
            return None
        api_key = api_keys.get("claude") or os.environ.get("CLAUDE_API_KEY")
        if not api_key:
            return None
        return ChatAnthropic(model_name=model_name, temperature=0, api_key=api_key)

    if provider == "ollama":
        if ChatOllama is None:
            return None
        # 本地 Ollama 无需密钥，允许自定义 base_url
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, temperature=0, base_url=base_url)

    if provider == "gemini":
        if ChatGoogleGenerativeAI is None:
            return None
        api_key = api_keys.get("gemini") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return None
        # ChatGoogleGenerativeAI 接受 model 和 api_key
        return ChatGoogleGenerativeAI(model=model_name, api_key=api_key, temperature=0)

    return None


def create_native_client(provider: str, api_keys: dict):
    """
    返回兼容 OpenAI SDK 的原生客户端（用于 dashscope/deepseek 的兼容模式）。
    其余提供商返回 None。
    """
    from openai import OpenAI

    provider = (provider or "openai").lower()
    if provider == "dashscope":
        api_key = api_keys.get("dashscope") or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            return None, None
        client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        return client, None
    if provider == "deepseek":
        api_key = api_keys.get("deepseek") or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            return None, None
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        return client, None
    return None, None