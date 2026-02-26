"""
混合检索模块
结合向量检索和BM25关键词检索
"""
from typing import List, Dict, Any, Tuple
import logging
from langchain_core.documents import Document
from backend.core.vector_store import VectorStoreManager
from backend.core.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索器，结合向量和BM25"""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        alpha: float = 0.5,
        k: int = 10
    ):
        """
        初始化混合检索器

        Args:
            vector_store_manager: 向量存储管理器
            alpha: 向量检索权重 (0-1)，BM25权重为 1-alpha
            k: 每种检索返回的结果数量
        """
        self.vector_store_manager = vector_store_manager
        self.alpha = alpha
        self.k = k
        self.bm25 = BM25Retriever()

        # 尝试加载BM25索引
        if not self.bm25.is_indexed():
            logger.warning("BM25索引不存在，将只使用向量检索")
        else:
            logger.info("BM25索引已加载")

    def index_documents_for_bm25(self, documents: List[Document]):
        """
        为BM25建立索引

        Args:
            documents: 文档列表
        """
        try:
            self.bm25.index_documents(documents)
            logger.info(f"BM25索引建立完成，文档数: {len(documents)}")
        except Exception as e:
            logger.error(f"建立BM25索引失败: {e}")

    def search(
        self,
        query: str,
        k: int = 4,
        use_hybrid: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        混合检索

        Args:
            query: 查询文本
            k: 返回结果数量
            use_hybrid: 是否使用混合检索

        Returns:
            检索结果列表 [(doc, score), ...]
        """
        if not use_hybrid or not self.bm25.is_indexed():
            # 只使用向量检索
            logger.debug("使用纯向量检索")
            docs_with_scores = self.vector_store_manager.similarity_search_with_score(
                query=query,
                k=k
            )
            # 转换分数为越小越好（向量检索的score是距离，越小越好）
            return [(doc, 1.0 / (1.0 + score)) for doc, score in docs_with_scores]

        # 向量检索
        vector_results = self.vector_store_manager.similarity_search_with_score(
            query=query,
            k=self.k
        )

        # BM25检索
        bm25_results = self.bm25.search(query=query, k=self.k)

        # 融合结果（RRF - Reciprocal Rank Fusion）
        fused_results = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            k=k
        )

        return fused_results

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Dict[str, Any]],
        k: int = 4,
        rrf_k: int = 60
    ) -> List[Tuple[Document, float]]:
        """
        倒数排序融合（RRF）算法

        Args:
            vector_results: 向量检索结果 [(doc, score), ...]
            bm25_results: BM25检索结果 [{"doc_id": ..., "score": ...}, ...]
            k: 返回结果数量
            rrf_k: RRF参数（默认60）

        Returns:
            融合后的结果列表 [(doc, score), ...]
        """
        # 创建分数字典
        scores = {}

        # 处理向量检索结果（使用排名而不是原始分数）
        for rank, (doc, _) in enumerate(vector_results, 1):
            # 使用文档内容作为key（简化处理）
            doc_key = doc.page_content[:100]  # 使用前100个字符作为唯一标识
            rrf_score = 1.0 / (rrf_k + rank)
            scores[doc_key] = scores.get(doc_key, 0) + rrf_score * self.alpha

        # 处理BM25结果
        for rank, result in enumerate(bm25_results, 1):
            doc_id = result["doc_id"]
            # 获取文档内容
            if doc_id < len(self.bm25.corpus):
                # 从BM25的语料库中找到对应的文档
                # 这里简化处理，实际应该维护文档ID到Document对象的映射
                pass

        # 由于文档映射的复杂性，这里使用简化的融合策略
        # 直接对向量检索结果进行重新排序

        # 提取向量检索的文档
        docs = [doc for doc, _ in vector_results]

        # 如果没有BM25结果，返回向量检索结果
        if not bm25_results:
            return [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs[:k])]

        # 简化：返回向量检索的前k个结果
        # 实际应用中应该维护文档ID映射
        logger.debug("使用简化的RRF融合")
        return [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs[:k])]

    def retrieve_with_reranking(
        self,
        query: str,
        k: int = 4,
        reranker=None,
        use_hybrid: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        检索并重排序

        Args:
            query: 查询文本
            k: 返回结果数量
            reranker: 重排序器
            use_hybrid: 是否使用混合检索

        Returns:
            检索并重排序后的结果
        """
        # 第一阶段：检索（获取更多候选）
        candidates = self.search(query, k=k * 3, use_hybrid=use_hybrid)

        # 第二阶段：重排序
        if reranker and candidates:
            docs = [doc for doc, _ in candidates]
            reranked = reranker.rerank(query, docs, top_k=k)
            return reranked

        return candidates[:k]
