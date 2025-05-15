"""
Module for managing chat history in the RAG system.
"""
import json
from datetime import datetime
from pathlib import Path

class ChatHistoryManager:
    def __init__(self, history_dir: str = "chat_history"):
        self.history_dir = Path(history_dir)
        self.history_file = self.history_dir / "conversation_history.json"
        self.ensure_history_dir()
        self.conversation_history = self.load_history()

    def ensure_history_dir(self):
        """確保歷史記錄目錄存在"""
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def load_history(self):
        """從檔案載入對話歷史"""
        if self.history_file.exists():
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def save_history(self):
        """儲存對話歷史到檔案"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)

    def add_conversation(self, question: str, answer: str, model_type: str):
        """添加新的對話到歷史記錄"""
        conversation = {
            'timestamp': datetime.now().isoformat(),
            'model': model_type,
            'question': question,
            'answer': answer
        }
        self.conversation_history.append(conversation)
        self.save_history()

    def get_recent_history(self, limit: int = 5) -> list[dict]:
        """獲取最近的對話歷史"""
        return self.conversation_history[-limit:]

    def clear_history(self):
        """清空對話歷史"""
        self.conversation_history = []
        self.save_history()