"""
重排序模块
使用Reranker模型对检索结果进行重新排序
"""
from typing import List, Dict, Any, Tuple
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class Reranker:
    """重排序器基类"""

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回前k个结果

        Returns:
            排序后的文档列表 [(doc, score), ...]
        """
        raise NotImplementedError


class SimpleReranker(Reranker):
    """简单的基于关键词匹配的重排序器"""

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        基于关键词匹配的重排序

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回前k个结果

        Returns:
            排序后的文档列表
        """
        # 提取查询中的关键词
        query_keywords = set(query.split())

        scored_docs = []
        for doc in documents:
            content = doc.page_content.lower()

            # 计算匹配分数
            score = 0
            for keyword in query_keywords:
                if keyword.lower() in content:
                    # 关键词出现次数
                    count = content.count(keyword.lower())
                    score += count * 10

            scored_docs.append((doc, score))

        # 按分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            scored_docs = scored_docs[:top_k]

        return scored_docs


class CrossEncoderReranker(Reranker):
    """基于交叉编码器的重排序器"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        初始化CrossEncoder重排序器

        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"加载重排序模型: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("重排序模型加载成功")
        except ImportError:
            logger.warning("sentence_transformers未安装，使用简单重排序器")
            self.model = None
        except Exception as e:
            logger.error(f"加载重排序模型失败: {e}")
            self.model = None

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        使用CrossEncoder进行重排序

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回前k个结果

        Returns:
            排序后的文档列表
        """
        if self.model is None:
            # 降级到简单重排序器
            logger.warning("使用简单重排序器")
            simple_reranker = SimpleReranker()
            return simple_reranker.rerank(query, documents, top_k)

        if not documents:
            return []

        try:
            # 准备输入对
            pairs = [[query, doc.page_content] for doc in documents]

            # 计算分数
            scores = self.model.predict(pairs)

            # 组合文档和分数
            scored_docs = list(zip(documents, scores))

            # 按分数排序（分数越高越好）
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            if top_k:
                scored_docs = scored_docs[:top_k]

            logger.debug(f"重排序完成，返回 {len(scored_docs)} 个结果")
            return scored_docs

        except Exception as e:
            logger.error(f"重排序失败: {e}，使用原始顺序")
            docs_to_return = documents[:top_k] if top_k else documents
            return [(doc, 0.0) for doc in docs_to_return]


class HybridReranker(Reranker):
    """混合重排序器，结合多种信号"""

    def __init__(self, use_cross_encoder: bool = True):
        """
        初始化混合重排序器

        Args:
            use_cross_encoder: 是否使用CrossEncoder
        """
        self.use_cross_encoder = use_cross_encoder
        if use_cross_encoder:
            self.cross_encoder = CrossEncoderReranker()
        else:
            self.cross_encoder = None
        self.simple_reranker = SimpleReranker()

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        混合重排序

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回前k个结果

        Returns:
            排序后的文档列表
        """
        if self.use_cross_encoder and self.cross_encoder and self.cross_encoder.model:
            return self.cross_encoder.rerank(query, documents, top_k)
        else:
            return self.simple_reranker.rerank(query, documents, top_k)
