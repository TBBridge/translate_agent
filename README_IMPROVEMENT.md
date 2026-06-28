# 翻译结果自动改进功能指南

本文档介绍TranslationAgent的翻译结果自动改进功能，该功能能够根据审核模型的建议自动优化翻译文本质量。

## 功能概述

TranslationAgent现在可以：
1. **自动应用基础改进**：对所有翻译结果应用常见的语言优化
2. **基于审核意见动态改进**：利用AI模型分析审核意见，针对性地优化翻译文本
3. **生成改进版本文件**：保存改进后的翻译结果为单独的文件

## 工作原理

### 1. 基础改进

系统会自动对所有翻译结果应用以下基础改进：

- 将"章节"替换为更符合中文表达习惯的"部分"
- 简化冗余表达，如将"人工智能（人工智能）"简化为"人工智能"
- 将"引用文的示例。"改进为"这是引用文的示例。"
- 移除残留的日语词汇，如将"セクション"替换为"部分"

这些改进在没有API密钥的情况下也能工作。

### 2. 基于审核意见的动态改进

当提供了有效的API密钥时，系统会：

1. 使用相应的审核模型（中文用DeepSeek-V3，英文用Claude 3.7 Sonnet）对翻译结果进行审核
2. 解析审核意见，提取具体的改进建议
3. 调用原始翻译模型（中文用Qwen-Max，英文用GPT-4o）根据审核建议重新优化翻译文本
4. 保存优化后的结果为新文件

## 文件结构

翻译流程完成后，会生成以下文件：

- `原始文件名_目标语言.md`：初始翻译结果
- `原始文件名_目标语言_review.txt`：审核结果文件，包含详细的审核意见
- `原始文件名_目标语言_improved.md`：根据审核意见改进后的翻译结果

## 使用方法

### 基本使用

翻译改进功能会自动运行，无需额外配置：

```python
from main import TranslationAgent

# 创建翻译代理实例
agent = TranslationAgent(
    github_repo="octocat/Spoon-Knife",
    target_language="zh"  # 或 "en"
)

# 运行翻译流程（会自动进行审核和改进）
result = agent.run()
```

### 命令行使用

通过命令行运行时，系统也会自动进行审核和改进：

```bash
# 中文翻译（会自动使用Qwen-Max翻译和DeepSeek-V3审核）
python main.py --github-repo octocat/Spoon-Knife --target-language zh

# 英文翻译（会自动使用GPT-4o翻译和Claude 3.7 Sonnet审核）
python main.py --github-repo octocat/Spoon-Knife --target-language en
```

## 结果查看

翻译任务完成后，系统会输出所有生成的文件路径：

```
翻译任务完成！
- 翻译文件: translated_results/README_zh.md
  审核文件: translated_results/README_zh_review.txt
  改进文件: translated_results/README_zh_improved.md
```

## 配置要求

### 基础改进功能

- 无需特殊配置，任何情况下都会自动应用

### 基于审核意见的动态改进

需要配置以下API密钥：

- **中文翻译改进**：需要配置`DASHSCOPE_API_KEY`（用于Qwen-Max模型）
- **英文翻译改进**：需要配置`OPENAI_API_KEY`（用于GPT-4o模型）
- **中文审核**：需要配置`DEEPSEEK_API_KEY`（用于DeepSeek-V3模型）
- **英文审核**：需要配置`CLAUDE_API_KEY`（用于Claude 3.7 Sonnet模型）

这些API密钥可以在`.env`文件或`config.ini`文件中配置。

## 故障排除

### 常见问题及解决方案

1. **没有生成改进文件**
   - 检查是否配置了正确的API密钥
   - 查看控制台输出是否有错误信息
   - 确认审核功能是否正常运行

2. **改进后的翻译质量不佳**
   - 检查审核模型是否返回了有意义的建议
   - 尝试更新API密钥或使用不同的模型组合

3. **程序运行缓慢**
   - 翻译改进功能需要额外的API调用，可能会增加整体运行时间
   - 对于大型项目，可以考虑分批处理文件

## 版本更新说明

- 添加了自动翻译改进功能
- 实现了基于审核意见的动态优化
- 增加了基础改进策略，无需API密钥也能提升翻译质量
- 更新了文件命名规则，生成单独的改进版本文件
- 优化了输出信息，显示所有生成的文件路径