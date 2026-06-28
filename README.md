# 翻译智能体

这是一个基于LangChain的翻译智能体，可以自动从GitHub仓库获取Markdown文件或处理本地压缩文件，使用AI模型翻译日语内容为其他语言，同时保持原始格式和图片URL，并使用另一模型审核翻译结果。支持多提供商与智能回退：优先本地 Ollama，不可用时按语言/任务回退（英文翻译→OpenAI，英文审核→Gemini；中文翻译→DashScope(Qwen)，中文审核→DeepSeek）。

## 功能特点

- 优先直接下载GitHub Markdown文件以避免克隆权限问题
- 支持GitHub认证，可以访问私有仓库或提高API速率限制
- 支持处理本地压缩文件（ZIP、TAR、TAR.GZ等格式）
- **使用AI模型智能翻译日语内容，保持原始格式和图片URL不变**
- 使用自定义词典替换专用词语
- 生成带语言标识的文件名
- 利用多提供商审核翻译结果：优先 Ollama，不可用时按语言/任务回退
- 提供详细的错误处理和重试机制

## 安装步骤

1. 克隆或下载此项目

2. 安装依赖包
   ```bash
   pip install -r requirements.txt
   ```

3. 配置环境变量
   - 复制 `.env.example` 文件为 `.env`
   - 推荐配置以下密钥以获得最佳效果：
     - `OPENAI_API_KEY`（英文翻译/兜底）
     - `GOOGLE_API_KEY`（英文审核-Gemini）
     - `DASHSCOPE_API_KEY`（中文翻译-Qwen）
     - `DEEPSEEK_API_KEY`（中文审核-DeepSeek）
     - （可选）`CLAUDE_API_KEY`（备用审核）
   - （可选但推荐）填入GitHub令牌以访问私有仓库或提高API限制

## 配置要求

1. **LLM提供商与密钥**
   - 默认优先使用本地 `Ollama`（无需密钥）。若不可用，按以下优先级回退：
     - 英文：翻译→`OpenAI`，审核→`Gemini`
     - 中文（简体/繁体）：翻译→`DashScope(Qwen)`，审核→`DeepSeek`
   - 对应需要的环境变量：`OPENAI_API_KEY`、`GOOGLE_API_KEY`、`DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY`（可选：`CLAUDE_API_KEY`）

2. **GitHub令牌（可选但推荐）**
   - 用于访问私有仓库或提高API速率限制
   - 获取方法：
     1. 访问 https://github.com/settings/tokens
     2. 点击 "Generate new token" -> "Generate new token (classic)"
     3. 设置描述名称，选择有效期
     4. 选择权限：对于公开仓库，只需勾选 `public_repo`；对于私有仓库，勾选 `repo`
     5. 点击 "Generate token" 并复制保存
   - 设置 `GITHUB_TOKEN` 环境变量

3. **词典文件**
   - 默认使用 `dictionary.txt`
   - 格式：`日语词语=目标语言翻译`，每行一个

## 词典格式

词典文件使用简单的键值对格式：

```
AI=人工智能
機械学習=机器学习
データサイエンス=数据科学
```

## 使用方法

### 1. 使用配置文件

最简单的使用方式是通过配置文件设置参数。默认配置文件为项目根目录下的`config.ini`：

```bash
python main.py
```

也可以指定自定义配置文件：

```bash
python main.py --config-file ./custom_config.ini
```

配置文件的详细说明请参考[README_CONFIG.md](README_CONFIG.md)。

### 2. 命令行参数

也可以直接通过命令行参数设置：

```bash
python main.py --file <Markdown文件路径> --target-lang <目标语言> --output <输出目录>
```

参数说明：
- `--file`: 直接翻译的 Markdown 文件路径（本地文件）
- `--target-lang`: 翻译目标语言（`en` / `zh-CN` / `zh-TW`）
- `--output`: 翻译结果输出目录（省略时使用原文件同目录）
- `--config-file`: 配置文件路径，默认为'./config.ini'

> GitHub 仓库的 `repo/branch/target_paths` 目前主要从 `config.ini` 读取；如果只需要翻译单个本地 MD，请使用 `--file`。

## 工作流程

1. 程序启动后，首先检查是否提供了压缩文件
   - 如果提供了压缩文件，解压并查找所有MD文件
2. 如果没有提供压缩文件，优先尝试直接通过GitHub API下载MD文件
   - 如果设置了GitHub令牌，会使用令牌进行认证
   - 如果遇到API速率限制，会自动重试
   - 如果下载失败，会尝试克隆整个仓库
3. 查找所有Markdown文件
4. **对每个文件使用AI模型进行翻译，保持格式和图片URL不变**
5. 使用配置的多提供商审核翻译结果（英文默认Gemini，中文默认DeepSeek）
6. 清理临时文件

## 支持的压缩文件格式

- ZIP格式（.zip文件）
- TAR格式（.tar文件）
- GZIP压缩的TAR格式（.tar.gz或.tgz文件）

## 注意事项

1. 直接下载功能适用于公开仓库，对于私有仓库，需要提供GitHub令牌
2. 翻译/审核会优先使用配置的 LLM 提供商；如果 `OpenAI/Gemini/DeepSeek/DashScope` 侧出现 `quota不足(429)` 或其它错误，**将尽可能回退到本地 Ollama**（需要 `ollama` 已启动且对应模型可用）。
3. 程序会在当前目录生成翻译后的文件和审核结果文件
4. 确保有足够的磁盘空间存储临时文件和翻译结果
5. 对于大型仓库或压缩文件，可能需要较长时间完成翻译过程
6. 如果遇到GitHub API速率限制问题，请设置GitHub令牌或等待一段时间后重试
7. 压缩文件中的所有.md文件都会被翻译，无论其所在目录位置
8. **使用AI模型进行翻译时，输入内容会被限制在合理长度以避免API限制**

## Ollama 回退模型映射
当主提供商的翻译/审核请求失败时，Ollama 将回退到以下模型：

- `gpt-5.1` -> `gpt-oss:120b`
- `gpt-3.5-turbo` -> `gpt-oss:20b`
- `qwen3-max` -> `qwen3:30b`
- `qwen-max` -> `qwen3:30b`
- `deepseek-chat` -> `deepseek-r1:14b`

## 故障排除

### GitHub 401 “Bad credentials”

- 常见原因：
  - 未设置 `GITHUB_TOKEN` 或值为空/拼写错误
  - 令牌已过期、被撤销或未授权组织 SSO
  - 权限不足（例如访问私有仓库或细粒度令牌未授予必要权限）
- 解决步骤：
  - 生成令牌：
    - Classic Token：勾选 `public_repo`（公开仓库）或 `repo`（私有）
    - Fine-grained Token：为目标仓库授予 `Contents: Read-only` 与 `Metadata: Read-only`
  - 配置方式（二选一或同时）：
    - PowerShell 临时设置：`$env:GITHUB_TOKEN="<your_token>"`
    - 永久设置：`setx GITHUB_TOKEN "<your_token>"`（重启终端生效）
    - `.env` 文件：添加 `GITHUB_TOKEN=<your_token>` 并确保已启用 `python-dotenv`
  - 再次运行：`python main.py`（确保 `config.ini` 中 `[github]` 的 `repo/branch/target_paths` 已正确配置）

## 许可证

[MIT](LICENSE)