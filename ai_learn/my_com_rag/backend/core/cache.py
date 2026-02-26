"""
答案缓存模块
用于缓存RAG生成的答案，避免重复消耗token
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AnswerCache:
    """答案缓存管理器"""

    def __init__(self, cache_dir: str = "data/cache", ttl_hours: int = 24):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录
            ttl_hours: 缓存有效期（小时）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.stats_file = self.cache_dir / "stats.json"

        # 加载统计信息
        self.stats = self._load_stats()

    def _load_stats(self) -> Dict[str, Any]:
        """加载缓存统计"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载缓存统计失败: {e}")
        return {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "tokens_saved": 0
        }

    def _save_stats(self):
        """保存缓存统计"""
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存缓存统计失败: {e}")

    def _generate_key(self, question: str) -> str:
        """
        生成缓存key

        Args:
            question: 用户问题

        Returns:
            缓存key（MD5 hash）
        """
        # 使用MD5生成唯一key
        return hashlib.md5(question.encode('utf-8')).hexdigest()

    def get(self, question: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存的答案

        Args:
            question: 用户问题

        Returns:
            缓存的答案，如果不存在或已过期则返回None
        """
        self.stats["total_queries"] += 1

        key = self._generate_key(question)
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            self.stats["cache_misses"] += 1
            self._save_stats()
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # 检查是否过期
            cache_time = datetime.fromisoformat(cache_data["timestamp"])
            if datetime.now() - cache_time > self.ttl:
                # 过期，删除缓存
                cache_file.unlink()
                self.stats["cache_misses"] += 1
                self._save_stats()
                return None

            # 缓存命中
            self.stats["cache_hits"] += 1
            self._save_stats()

            logger.info(f"缓存命中: {question[:30]}...")
            return cache_data["answer"]

        except Exception as e:
            logger.error(f"读取缓存失败: {e}")
            self.stats["cache_misses"] += 1
            self._save_stats()
            return None

    def set(self, question: str, answer: Dict[str, Any]):
        """
        保存答案到缓存

        Args:
            question: 用户问题
            answer: AI生成的答案
        """
        key = self._generate_key(question)
        cache_file = self.cache_dir / f"{key}.json"

        cache_data = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"答案已缓存: {question[:30]}...")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def clear(self):
        """清空所有缓存"""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                if cache_file.name != "stats.json":
                    cache_file.unlink()
            logger.info("缓存已清空")
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")

    def clear_expired(self):
        """清理过期的缓存"""
        try:
            now = datetime.now()
            expired_count = 0

            for cache_file in self.cache_dir.glob("*.json"):
                if cache_file.name == "stats.json":
                    continue

                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)

                    cache_time = datetime.fromisoformat(cache_data["timestamp"])
                    if now - cache_time > self.ttl:
                        cache_file.unlink()
                        expired_count += 1
                except Exception:
                    # 损坏的缓存文件，删除
                    cache_file.unlink()
                    expired_count += 1

            logger.info(f"清理了 {expired_count} 个过期缓存")
        except Exception as e:
            logger.error(f"清理过期缓存失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        stats = self.stats.copy()

        # 计算命中率
        if stats["total_queries"] > 0:
            stats["hit_rate"] = round(stats["cache_hits"] / stats["total_queries"] * 100, 2)
        else:
            stats["hit_rate"] = 0.0

        # 估算节省的token数（假设每次问答消耗1000 tokens）
        stats["estimated_tokens_saved"] = stats["cache_hits"] * 1000

        # 获取缓存文件数量
        cache_files = list(self.cache_dir.glob("*.json"))
        stats["cache_count"] = len([f for f in cache_files if f.name != "stats.json"])

        return stats

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "tokens_saved": 0
        }
        self._save_stats()
        logger.info("缓存统计已重置")


# 全局缓存实例
answer_cache = AnswerCache()
