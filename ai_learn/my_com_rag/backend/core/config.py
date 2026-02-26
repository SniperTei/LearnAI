"""
配置管理模块
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """应用配置"""

    # API配置
    ZHIPUAI_API_KEY: str
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True

    # 向量数据库配置
    VECTOR_DB_TYPE: str = "chromadb"
    VECTOR_DB_PATH: str = "./data/vector_db"

    # 文档路径
    DOCUMENTS_PATH: str = "./data/documents"
    UPLOAD_PATH: str = "./data/uploads"

    # LangChain配置
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TEMPERATURE: float = 0.7

    # RAG优化配置
    USE_QUERY_REWRITE: bool = True  # 是否使用查询重写
    USE_RERANKER: bool = True  # 是否使用重排序
    USE_HYBRID_SEARCH: bool = True  # 是否使用混合检索
    HYBRID_ALPHA: float = 0.5  # 混合检索中向量检索的权重
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"  # 重排序模型

    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_PATH: str = "./logs"

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # 忽略额外的字段
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保必要的目录存在
        self._create_directories()

    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.DOCUMENTS_PATH,
            self.UPLOAD_PATH,
            self.VECTOR_DB_PATH,
            self.LOG_PATH,
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# 全局配置实例
settings = Settings()
