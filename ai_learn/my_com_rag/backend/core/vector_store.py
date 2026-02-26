"""
向量数据库管理模块
"""
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS
from backend.core.embeddings import get_embeddings
from backend.core.config import settings
import logging

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """向量数据库管理器"""

    def __init__(self, collection_name: str = "knowledge_base"):
        self.collection_name = collection_name
        self.embeddings = get_embeddings()
        self.vector_store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """初始化向量数据库"""
        try:
            if settings.VECTOR_DB_TYPE.lower() == "chromadb":
                persist_directory = settings.VECTOR_DB_PATH
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=persist_directory
                )
                logger.info(f"初始化Chroma向量数据库: {persist_directory}")

            elif settings.VECTOR_DB_TYPE.lower() == "faiss":
                import faiss
                index_path = f"{settings.VECTOR_DB_PATH}/faiss_index"
                try:
                    self.vector_store = FAISS.load_local(
                        index_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"加载已存在的FAISS索引: {index_path}")
                except:
                    self.vector_store = None
                    logger.info("FAISS索引不存在，将在首次添加文档时创建")

            else:
                raise ValueError(f"不支持的向量数据库类型: {settings.VECTOR_DB_TYPE}")

        except Exception as e:
            logger.error(f"初始化向量数据库失败: {str(e)}")
            raise

    def add_documents(self, documents: List[Document]) -> bool:
        """
        添加文档到向量数据库

        Args:
            documents: 文档列表

        Returns:
            是否成功
        """
        try:
            if not documents:
                logger.warning("没有文档需要添加")
                return False

            if settings.VECTOR_DB_TYPE.lower() == "faiss" and self.vector_store is None:
                # 为FAISS创建新索引
                from langchain_community.vectorstores import FAISS
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                self._save_faiss()
            elif settings.VECTOR_DB_TYPE.lower() == "faiss":
                # 添加到现有FAISS索引
                self.vector_store.add_documents(documents)
                self._save_faiss()
            else:
                # Chroma直接添加
                self.vector_store.add_documents(documents)

            logger.info(f"成功添加 {len(documents)} 个文档到向量数据库")
            return True

        except Exception as e:
            logger.error(f"添加文档到向量数据库失败: {str(e)}")
            return False

    def _save_faiss(self):
        """保存FAISS索引"""
        if settings.VECTOR_DB_TYPE.lower() == "faiss":
            index_path = f"{settings.VECTOR_DB_PATH}/faiss_index"
            self.vector_store.save_local(index_path)
            logger.info(f"FAISS索引已保存: {index_path}")

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 相似度阈值（可选）

        Returns:
            相关文档列表
        """
        try:
            if self.vector_store is None:
                logger.warning("向量数据库未初始化")
                return []

            if score_threshold:
                results = self.vector_store.similarity_search_with_score(
                    query,
                    k=k
                )
                # 过滤低于阈值的结果
                filtered_results = [
                    doc for doc, score in results
                    if score >= score_threshold
                ]
                return filtered_results
            else:
                results = self.vector_store.similarity_search(
                    query,
                    k=k
                )
                return results

        except Exception as e:
            logger.error(f"相似度搜索失败: {str(e)}")
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[tuple[Document, float]]:
        """
        相似度搜索（带分数）

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            (文档, 相似度分数)列表
        """
        try:
            if self.vector_store is None:
                logger.warning("向量数据库未初始化")
                return []

            results = self.vector_store.similarity_search_with_score(
                query,
                k=k
            )
            return results

        except Exception as e:
            logger.error(f"相似度搜索失败: {str(e)}")
            return []

    def delete_collection(self) -> bool:
        """
        删除整个集合

        Returns:
            是否成功
        """
        try:
            if settings.VECTOR_DB_TYPE.lower() == "chromadb":
                # Chroma需要重新初始化来清除
                self.vector_store.delete_collection()
                logger.info("已删除Chroma集合")

            elif settings.VECTOR_DB_TYPE.lower() == "faiss":
                import shutil
                index_path = f"{settings.VECTOR_DB_PATH}/faiss_index"
                if Path(index_path).exists():
                    shutil.rmtree(index_path)
                    logger.info("已删除FAISS索引")
                self.vector_store = None

            return True

        except Exception as e:
            logger.error(f"删除集合失败: {str(e)}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息

        Returns:
            集合信息字典
        """
        try:
            info = {
                "type": settings.VECTOR_DB_TYPE,
                "collection_name": self.collection_name,
                "initialized": self.vector_store is not None,
            }

            if settings.VECTOR_DB_TYPE.lower() == "chromadb" and self.vector_store:
                info["count"] = self.vector_store._collection.count()

            return info

        except Exception as e:
            logger.error(f"获取集合信息失败: {str(e)}")
            return {"error": str(e)}
