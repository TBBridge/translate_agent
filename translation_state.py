import os
import json
import hashlib
import time
from typing import Dict, List, Any

class TranslationStateManager:
    def __init__(self, state_file_path: str = "./translation_state.json"):
        """初始化翻译状态管理器"""
        self.state_file_path = state_file_path
        self.state_data = self._load_state()
        
    def _load_state(self) -> Dict[str, Any]:
        """从文件加载翻译状态"""
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载翻译状态失败: {e}")
        return {
            "last_translation_time": 0,
            "files": {}
        }
    
    def _save_state(self) -> None:
        """保存翻译状态到文件"""
        try:
            with open(self.state_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.state_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存翻译状态失败: {e}")
    
    def get_file_state(self, file_path: str) -> Dict[str, Any]:
        """获取指定文件的翻译状态"""
        return self.state_data["files"].get(file_path, {})
    
    def update_translation_state(self, file_path: str, original_content: str, file_sha: str, 
                                translated_path: str, translated_content: str, target_language: str) -> None:
        """更新文件的翻译状态"""
        # 计算内容哈希作为备选标识
        content_hash = self._calculate_content_hash(original_content)
        
        # 确保文件状态存在
        if file_path not in self.state_data["files"]:
            self.state_data["files"][file_path] = {}
        
        # 确保语言状态存在
        if target_language not in self.state_data["files"][file_path]:
            self.state_data["files"][file_path][target_language] = {}
        
        # 更新状态
        self.state_data["files"][file_path][target_language] = {
            "sha": file_sha,
            "content_hash": content_hash,
            "translated_path": translated_path,
            "translation_time": time.time(),
            "original_content": original_content,
            "translated_content": translated_content
        }
        
        self.state_data["last_translation_time"] = time.time()
        self._save_state()
    
    def has_file_changed(self, file_path: str, current_content: str, current_sha: str = None) -> bool:
        """检查文件是否有变化"""
        # 获取所有语言的状态
        file_state = self.get_file_state(file_path)
        if not file_state:
            return True  # 文件从未翻译过，视为变更
        
        # 检查是否有任何语言的翻译记录
        for lang_state in file_state.values():
            # 优先使用SHA比较
            if current_sha and lang_state.get("sha") == current_sha:
                return False  # 文件未变更
            
            # 使用内容哈希比较
            current_content_hash = self._calculate_content_hash(current_content)
            if lang_state.get("content_hash") == current_content_hash:
                return False  # 文件未变更
        
        return True  # 文件已变更
    
    def has_translation_history(self, file_path: str, target_language: str) -> bool:
        """检查是否有上次的翻译历史"""
        file_state = self.get_file_state(file_path)
        return target_language in file_state
    
    def get_previous_translation_path(self, file_path: str, target_language: str) -> str:
        """获取上次翻译的文件路径"""
        file_state = self.get_file_state(file_path)
        if target_language in file_state:
            return file_state[target_language].get("translated_path", "")
        return ""
    
    def get_previous_file_content(self, file_path: str) -> str:
        """获取上次翻译的原始文件内容"""
        file_state = self.get_file_state(file_path)
        # 返回第一个语言的原始内容（通常所有语言的原始内容是相同的）
        for lang_state in file_state.values():
            if "original_content" in lang_state:
                return lang_state["original_content"]
        return ""
    
    def _calculate_content_hash(self, content: str) -> str:
        """计算内容的哈希值"""
        if not content:
            return ""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_all_file_paths(self) -> List[str]:
        """获取所有有记录的文件路径"""
        return list(self.state_data["files"].keys())
    
    def clear_state(self) -> None:
        """清除所有翻译状态"""
        self.state_data = {
            "last_translation_time": 0,
            "files": {}
        }
        self._save_state()

    def compare_file_content(self, file_path: str, new_content: str) -> Dict[str, Any]:
        """比较文件内容变更，返回变更信息"""
        file_state = self.get_file_state(file_path)
        translated_path = file_state.get("translated_path", "")
        
        # 检查是否有上次的翻译文件
        if not os.path.exists(translated_path):
            return {
                "has_changes": True,
                "needs_full_translation": True,
                "diff": None
            }
        
        try:
            # 简单的行级差分分析
            new_lines = new_content.split('\n')
            
            # 返回变更信息
            return {
                "has_changes": True,  # 我们假设只要调用此方法，就可能有变更
                "needs_full_translation": False,
                "new_lines": new_lines,
                "translated_path": translated_path
            }
        except Exception as e:
            print(f"比较文件内容时出错: {e}")
            return {
                "has_changes": True,
                "needs_full_translation": True,
                "diff": None
            }