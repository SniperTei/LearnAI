"""
用户认证和权限管理
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
import hashlib
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class UserManager:
    """用户管理器"""

    def __init__(self):
        self.users_file = Path("data/users.json")
        self.history_file = Path("data/upload_history.json")
        self._ensure_files()
        self.users = self._load_users()
        self.history = self._load_history()

    def _ensure_files(self):
        """确保必要的文件存在"""
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        # 如果用户文件不存在，创建默认管理员
        if not self.users_file.exists():
            default_users = {
                "admin": {
                    "username": "admin",
                    "password": self._hash_password("admin666"),
                    "role": "admin",
                    "created_at": datetime.now().isoformat()
                }
            }
            self._save_json(self.users_file, default_users)
            logger.info("创建默认管理员账号: admin/admin666")

        # 如果历史文件不存在，创建空历史
        if not self.history_file.exists():
            self._save_json(self.history_file, [])

    def _load_users(self) -> Dict:
        """加载用户数据"""
        return self._load_json(self.users_file, {})

    def _load_history(self) -> List[Dict]:
        """加载上传历史"""
        return self._load_json(self.history_file, [])

    def _load_json(self, file_path: Path, default):
        """加载JSON文件"""
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
        return default

    def _save_json(self, file_path: Path, data):
        """保存JSON文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存文件失败 {file_path}: {e}")

    def _hash_password(self, password: str) -> str:
        """哈希密码"""
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """验证用户"""
        if username not in self.users:
            return None

        user = self.users[username]
        password_hash = self._hash_password(password)

        if user["password"] == password_hash:
            return {
                "username": user["username"],
                "role": user["role"]
            }
        return None

    def get_user_role(self, username: str) -> Optional[str]:
        """获取用户角色"""
        if username in self.users:
            return self.users[username]["role"]
        return None

    def can_upload(self, username: str) -> bool:
        """检查用户是否可以上传"""
        role = self.get_user_role(username)
        return role == "admin"

    def add_upload_history(
        self,
        username: str,
        filename: str,
        file_size: int,
        doc_count: int,
        success: bool
    ):
        """添加上传历史记录"""
        record = {
            "id": len(self.history) + 1,
            "username": username,
            "filename": filename,
            "file_size": file_size,
            "doc_count": doc_count,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(record)
        self._save_json(self.history_file, self.history)
        logger.info(f"记录上传历史: {username} 上传了 {filename}")

    def get_upload_history(
        self,
        username: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取上传历史"""
        history = self.history

        # 按用户筛选
        if username:
            history = [h for h in history if h["username"] == username]

        # 按时间倒序
        history = reversed(history)

        # 限制数量
        return list(history)[:limit]

    def add_user(self, username: str, password: str, role: str = "user") -> bool:
        """添加新用户"""
        if username in self.users:
            return False

        self.users[username] = {
            "username": username,
            "password": self._hash_password(password),
            "role": role,
            "created_at": datetime.now().isoformat()
        }
        self._save_json(self.users_file, self.users)
        logger.info(f"添加新用户: {username} (角色: {role})")
        return True


# 全局用户管理器实例
user_manager = UserManager()
