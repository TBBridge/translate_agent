import difflib
import re
from typing import List, Dict, Tuple, Any

class FileDiffAnalyzer:
    def __init__(self):
        """初始化文件差分分析器"""
        self.markdown_patterns = {
            "header": r'^(#{1,6})\s+(.*)',
            "code_block": r'^(```(?:\w+)?)(.*?)(```)',
            "list_item": r'^(\s*)([-\*\d+\.])\s+',
            "blockquote": r'^>\s+',
            "table": r'^\|.*\|'
        }
    
    def compare_files(self, old_content: str, new_content: str) -> Dict[str, Any]:
        """比较两个版本的文件内容，返回差分信息"""
        old_lines = old_content.split('\n')
        new_lines = new_content.split('\n')
        
        # 使用difflib生成行级差分
        differ = difflib.Differ()
        diff_result = list(differ.compare(old_lines, new_lines))
        
        # 分析变更类型和位置
        changes = {
            "added": [],  # 新增的行
            "modified": [],  # 修改的行
            "deleted": [],  # 删除的行
            "unchanged": []  # 未变更的行
        }
        
        line_mapping = {}
        
        old_line_index = 0
        new_line_index = 0
        
        for line in diff_result:
            line_type = line[:2]
            content = line[2:]
            
            if line_type == '  ':
                # 未变更的行
                changes["unchanged"].append((old_line_index, new_line_index, content))
                line_mapping[new_line_index] = old_line_index  # 建立新旧行号映射
                old_line_index += 1
                new_line_index += 1
            elif line_type == '+ ':
                # 新增的行
                changes["added"].append((new_line_index, content))
                new_line_index += 1
            elif line_type == '- ':
                # 删除的行
                changes["deleted"].append((old_line_index, content))
                old_line_index += 1
            elif line_type == '? ':
                # 行内变更标记，跳过
                continue
        
        # 分析变更的块
        change_blocks = self._identify_change_blocks(new_lines, changes)
        
        return {
            "has_changes": len(changes["added"]) > 0 or len(changes["modified"]) > 0 or len(changes["deleted"]) > 0,
            "changes": changes,
            "change_blocks": change_blocks,
            "line_mapping": line_mapping
        }
    
    def _identify_change_blocks(self, new_lines: List[str], changes: Dict[str, List]) -> List[Dict[str, Any]]:
        """识别变更的内容块"""
        change_blocks = []
        
        # 获取所有变更行的行号
        changed_line_numbers = set()
        for added_line in changes["added"]:
            changed_line_numbers.add(added_line[0])
        
        # 简单的块识别：将连续的变更行合并为一个块
        if not changed_line_numbers:
            return change_blocks
        
        sorted_line_numbers = sorted(changed_line_numbers)
        start_line = sorted_line_numbers[0]
        prev_line = start_line
        
        for line_num in sorted_line_numbers[1:] + [float('inf')]:
            if line_num > prev_line + 1:
                # 块结束
                change_blocks.append({
                    "start_line": start_line,
                    "end_line": prev_line,
                    "content": "\n".join(new_lines[start_line:prev_line + 1]) if start_line <= prev_line else ""
                })
                start_line = line_num
            prev_line = line_num
        
        return change_blocks
    
    def extract_markdown_sections(self, content: str) -> List[Dict[str, Any]]:
        """提取Markdown文件中的各个章节"""
        sections = []
        lines = content.split('\n')
        current_section = None
        
        for i, line in enumerate(lines):
            header_match = re.match(self.markdown_patterns["header"], line)
            
            if header_match:
                # 找到新的章节
                if current_section:
                    current_section["end_line"] = i - 1
                    sections.append(current_section)
                
                # 开始新章节
                level = len(header_match.group(1))
                title = header_match.group(2)
                
                current_section = {
                    "level": level,
                    "title": title,
                    "start_line": i,
                    "end_line": len(lines) - 1,  # 临时设置为文件末尾
                    "content_lines": [i]
                }
            elif current_section:
                # 添加到当前章节的内容行
                current_section["content_lines"].append(i)
        
        # 添加最后一个章节
        if current_section:
            sections.append(current_section)
        
        # 为每个章节提取内容
        for section in sections:
            section_content_lines = [lines[i] for i in section["content_lines"]]
            section["content"] = "\n".join(section_content_lines)
        
        return sections
    
    def get_changed_sections(self, old_content: str, new_content: str) -> List[Dict[str, Any]]:
        """获取发生变更的Markdown章节"""
        # 提取新旧内容的章节
        old_sections = self.extract_markdown_sections(old_content)
        new_sections = self.extract_markdown_sections(new_content)
        
        # 简单比较章节标题来识别变更的章节
        # 注意：这是一个简化的实现，更复杂的场景需要更复杂的匹配逻辑
        old_titles = {section["title"]: section for section in old_sections}
        changed_sections = []
        
        for new_section in new_sections:
            if new_section["title"] not in old_titles:
                # 新增章节
                new_section["change_type"] = "added"
                changed_sections.append(new_section)
            else:
                # 检查章节内容是否变更
                old_section = old_titles[new_section["title"]]
                if old_section["content"] != new_section["content"]:
                    new_section["change_type"] = "modified"
                    changed_sections.append(new_section)
        
        # 检查是否有删除的章节
        new_titles = {section["title"]: section for section in new_sections}
        for old_section in old_sections:
            if old_section["title"] not in new_titles:
                old_section["change_type"] = "deleted"
                changed_sections.append(old_section)
        
        return changed_sections
    
    def generate_partial_translation_prompt(self, previous_content: str, current_content: str, 
                                          previous_translation_content: str, changed_sections: List[Dict[str, Any]], 
                                          target_language: str) -> str:
        """生成部分翻译的提示模板"""
        prompt = f"以下是一个已翻译的Markdown文档，但原文有一些部分已经更新。请仅翻译新增或修改的部分，保持整体风格一致，并将翻译结果返回为{target_language}。\n\n"
        
        # 添加未变更的上下文
        context_lines = []
        for section in changed_sections:
            # 添加变更块前后的几行作为上下文
            start_context = max(0, section["start_line"] - 3)
            end_context = min(len(current_content.split('\n')) - 1, section["end_line"] + 3)
            
            prompt += f"\n【变更区域上下文】\n"
            for i in range(start_context, end_context + 1):
                current_lines = current_content.split('\n')
                previous_translated_lines = previous_translation_content.split('\n') if previous_translation_content else []
                
                line_text = current_lines[i] if i < len(current_lines) else ""
                prompt += f"原文[{i+1}]: {line_text}\n"
                
                if i < len(previous_translated_lines):
                    prompt += f"译文[{i+1}]: {previous_translated_lines[i]}\n"
            
            # 添加需要翻译的新内容
            prompt += f"\n【需要翻译的新内容】\n{section['content']}\n"
            prompt += "\n请提供这段新内容的翻译：\n"
        
        return prompt