# 词典功能使用说明

## 概述

翻译智能体现在支持多种词典数据源：
- **Notion数据库**：团队协作管理翻译词典
- **本地JSON文件**：结构化词典数据存储
- **本地TXT文件**：传统格式词典文件（向后兼容）

## 词典格式支持

### JSON格式

JSON词典文件格式：
```json
[
  {
    "Key": "术语关键词",
    "Japanese": "日语原文",
    "English": "英文翻译",
    "Chinese": "简体中文翻译",
    "Chinese_Traditional": "繁体中文翻译"
  }
]
```

## 配置方法

### 1. 配置文件方式

在 `config.ini` 文件中配置：

```ini
[translation]
# 专用词语文件路径（支持JSON和TXT格式）
dictionary_file = ./dictionary.json

[notion]
# 词典数据来源：notion（从Notion数据库读取）或 file（从本地文件读取）
dictionary_source = file
# Notion API Token（也可以通过环境变量NOTION_TOKEN设置）
notion_token = your_notion_token_here
# Notion数据库ID（也可以通过环境变量NOTION_DATABASE_ID设置）
notion_database_id = your_database_id_here
# API调用延迟（秒）
api_delay = 0.1
```

### 2. 环境变量方式

设置以下环境变量：

```bash
export NOTION_TOKEN="your_notion_token_here"
export NOTION_DATABASE_ID="your_database_id_here"
```

### 3. 命令行方式

使用 `--use-notion-dictionary` 参数强制使用Notion词典：

```bash
python main.py --use-notion-dictionary --file example.md
```

## Notion数据库结构

Notion数据库需要包含以下属性：

- **Key** (Title): 日语术语或关键词
- **Japanese** (Rich Text): 日语原文
- **English** (Rich Text): 英文翻译
- **Chinese** (Rich Text): 简体中文翻译
- **Chinese_Traditional** (Rich Text): 繁体中文翻译

## 功能特性

### 1. 混合模式支持

- 优先使用Notion词典
- Notion不可用时自动回退到本地文件
- 确保翻译过程的稳定性

### 2. 多语言支持

根据目标语言自动选择对应的翻译字段：
- `en` → English字段
- `zh_cn` → Chinese字段  
- `zh_tw` → Chinese_Traditional字段

### 3. 错误处理

- Notion客户端初始化失败时自动回退
- API调用失败时的重试机制
- 详细的错误日志和状态提示

## 使用示例

### 基本使用

```bash
# 使用本地JSON词典文件（默认）
python main.py --file document.md

# 使用本地TXT词典文件
python main.py --dictionary-file dictionary.txt --file document.md

# 使用Notion词典
python main.py --use-notion-dictionary --file document.md

# GitHub仓库翻译，使用JSON词典
python main.py --github-repo user/repo

# GitHub仓库翻译，使用Notion词典
python main.py --use-notion-dictionary --github-repo user/repo
```

### 配置优先级

1. 命令行参数 `--use-notion-dictionary`
2. 配置文件 `dictionary_source` 设置
3. 环境变量设置
4. 默认使用本地文件

## 安装依赖

确保安装了notion-client库：

```bash
pip install notion-client
```

或使用requirements.txt：

```bash
pip install -r requirements.txt
```

## 故障排除

### 常见问题

1. **notion-client库未安装**
   - 错误：`notion-client库未安装，Notion词典功能不可用`
   - 解决：`pip install notion-client`

2. **API Token无效**
   - 错误：`Notion客户端初始化失败`
   - 解决：检查NOTION_TOKEN是否正确设置

3. **数据库ID无效**
   - 错误：`Notion客户端初始化失败`
   - 解决：检查NOTION_DATABASE_ID是否正确

4. **数据库结构不匹配**
   - 错误：词典数据为空
   - 解决：确保数据库包含所需的属性字段

5. **JSON文件格式错误**
   - 错误：`JSON文件解析错误`
   - 解决：检查JSON文件格式是否正确，确保是数组格式

6. **词典文件不存在**
   - 错误：`词典文件不存在`
   - 解决：确保dictionary.json文件存在且路径正确

7. **词典数据为空**
   - 错误：词典加载成功但条目数为0
   - 解决：检查词典文件中是否有有效的Key和翻译字段

### 调试模式

启用详细日志输出：

```bash
python main.py --use-notion-dictionary --file test.md 2>&1 | tee debug.log
```

## 性能考虑

- API调用有延迟限制（默认0.1秒）
- 大量词典数据时首次加载可能较慢
- 建议在Notion中合理组织词典数据

## 迁移指南

### 从TXT格式迁移到JSON格式

1. 将现有的TXT词典文件转换为JSON格式
2. 在 `config.ini` 中更新 `dictionary_file = ./dictionary.json`
3. 测试翻译功能确保正常工作
4. 保留TXT文件作为备份

### 从本地词典文件迁移到Notion

1. 使用 `xml_translation_merger.py` 将现有词典数据上传到Notion
2. 在 `config.ini` 中设置 `dictionary_source = notion`
3. 配置Notion API Token和数据库ID
4. 测试翻译功能确保正常工作
5. 保留本地词典文件作为备份

### JSON格式转换工具

可以使用以下Python脚本将TXT词典转换为JSON格式：

```python
import json

def convert_txt_to_json(txt_file, json_file):
    dictionary = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                dictionary.append({
                    "Key": key.strip(),
                    "Japanese": "",
                    "English": value.strip(),
                    "Chinese": "",
                    "Chinese_Traditional": ""
                })
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)
```
