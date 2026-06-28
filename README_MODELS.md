# 翻译代理模型配置指南

本文档介绍如何配置TranslationAgent使用不同的AI模型进行翻译和审核。

## 支持的模型配置

TranslationAgent现在支持以下模型组合：

| 功能 | 目标语言 | 使用模型 | API提供商 |
|------|----------|----------|----------|
| 翻译 | 中文 | Qwen-Max | DashScope (阿里云) |
| 审核 | 中文 | DeepSeek-V3 | DeepSeek |
| 翻译 | 英文 | GPT-4o | OpenAI |
| 审核 | 英文 | Claude 3.7 Sonnet | Anthropic |

## 配置API密钥

### 环境变量配置

在`.env`文件中添加以下配置（如果文件不存在，请创建一个）：

```ini
# OpenAI API密钥
# 用于英文翻译功能
OPENAI_API_KEY=your_openai_api_key

# GitHub令牌（可选，但建议提供）
# 用于访问私有仓库或提高API速率限制
GITHUB_TOKEN=your_github_token

# DashScope API密钥
# 用于中文翻译（Qwen-Max模型）
DASHSCOPE_API_KEY=your_dashscope_api_key

# DeepSeek API密钥
# 用于中文翻译结果的审核（DeepSeek-V3模型）
DEEPSEEK_API_KEY=your_deepseek_api_key

# Claude API密钥
# 用于英文翻译结果的审核（Claude 3.7 Sonnet模型）
CLAUDE_API_KEY=your_claude_api_key
```

### 配置文件配置

你也可以在`config.ini`文件中配置这些API密钥：

```ini
[api]
# API密钥配置
# 从环境变量读取优先于此处设置
# github_token = ghp_your_github_token
# openai_api_key = your_openai_api_key
# dashscope_api_key = your_dashscope_api_key
# deepseek_api_key = your_deepseek_api_key
# claude_api_key = your_claude_api_key
```

**注意：** 环境变量中的配置优先级高于`config.ini`文件中的配置。

## 使用方法

### 基本使用

当你设置了正确的API密钥后，TranslationAgent会根据目标语言自动选择合适的模型：

```python
from main import TranslationAgent

# 中文翻译（自动使用Qwen-Max模型）
zh_agent = TranslationAgent(
    github_repo="octocat/Spoon-Knife",
    target_language="zh"
)
zh_agent.run()

# 英文翻译（自动使用GPT-4o模型）
en_agent = TranslationAgent(
    github_repo="octocat/Spoon-Knife",
    target_language="en"
)
en_agent.run()
```

### 命令行使用

你也可以通过命令行指定目标语言：

```bash
# 中文翻译
python main.py --github-repo octocat/Spoon-Knife --target-language zh

# 英文翻译
python main.py --github-repo octocat/Spoon-Knife --target-language en
```

## API密钥获取

### OpenAI API密钥

1. 访问 [OpenAI官网](https://platform.openai.com/)
2. 注册或登录账号
3. 在API设置页面创建API密钥

### DashScope API密钥

1. 访问 [阿里云DashScope官网](https://dashscope.aliyun.com/)
2. 注册或登录阿里云账号
3. 在控制台创建API密钥

### DeepSeek API密钥

1. 访问 [DeepSeek官网](https://www.deepseek.com/)
2. 注册或登录账号
3. 在API设置页面创建API密钥

### Claude API密钥

1. 访问 [Anthropic官网](https://www.anthropic.com/)
2. 注册或登录账号
3. 在API设置页面创建API密钥

## 故障排除

### 常见错误及解决方案

1. **未设置API密钥错误**
   - 确保在`.env`文件中设置了所有必要的API密钥
   - 检查环境变量是否正确加载

2. **模型初始化失败**
   - 检查API密钥是否有效
   - 确认你的API账号有足够的额度或权限访问相应的模型
   - 检查网络连接是否正常

3. **审核功能跳过**
   - 如果未设置相应的审核模型API密钥，系统会自动跳过审核步骤
   - 查看控制台输出以获取详细信息

## 模型回退机制

当首选模型无法使用时，系统会自动回退到GPT-3.5-turbo模型。这确保了即使在部分API不可用的情况下，翻译功能仍然可以工作。

## 版本更新说明

- 添加了对Qwen-Max、DeepSeek-V3和Claude 3.7 Sonnet模型的支持
- 实现了基于目标语言和任务类型的自动模型选择
- 更新了API密钥配置方式，支持环境变量和配置文件
- 添加了更完善的错误处理和模型回退机制