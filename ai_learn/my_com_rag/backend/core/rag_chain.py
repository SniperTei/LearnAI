"""
RAG（检索增强生成）链模块
"""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatZhipuAI
from backend.core.config import settings
from backend.core.vector_store import VectorStoreManager
from backend.core.embeddings import get_embeddings
from backend.core.cache import answer_cache
from backend.core.query_rewriter import QueryRewriter
from backend.core.reranker import HybridReranker
from backend.core.hybrid_retriever import HybridRetriever
import logging

logger = logging.getLogger(__name__)


def format_docs(docs: List[Document]) -> str:
    """格式化文档为字符串"""
    return "\n\n".join(doc.page_content for doc in docs)


class RAGChain:
    """RAG问答链"""

    # 自定义提示模板
    QA_TEMPLATE = """你是一个专业的企业知识库助手。请基于以下上下文信息回答用户的问题。

上下文信息：
{context}

用户问题：{question}

请遵循以下要求：
1. 如果上下文中有相关信息，请**完整地**基于上下文回答
2. 如果上下文中没有相关信息，请明确告知用户知识库中没有找到相关内容
3. 回答要准确、详细、专业
4. **重要**：如果上下文包含公司信息（如公司名称、税号、地址、开户行、账号等），必须在答案中完整列出这些信息
5. 如果上下文包含操作步骤、流程说明等，要详细列出具体步骤
6. 如果问题涉及多个方面，请分点回答，每个方面都要详细说明
7. 不要遗漏上下文中的重要信息

回答："""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        temperature: float = None,
        k: int = 4,
        use_query_rewrite: bool = True,
        use_reranker: bool = True,
        use_hybrid_search: bool = True,
        alpha: float = 0.5
    ):
        """
        初始化RAG链

        Args:
            vector_store_manager: 向量数据库管理器
            temperature: 温度参数
            k: 检索的文档数量
            use_query_rewrite: 是否使用查询重写
            use_reranker: 是否使用重排序
            use_hybrid_search: 是否使用混合检索
            alpha: 向量检索权重（混合检索时使用）
        """
        self.vector_store_manager = vector_store_manager
        self.temperature = temperature or settings.TEMPERATURE
        self.k = k
        self.use_query_rewrite = use_query_rewrite
        self.use_reranker = use_reranker
        self.use_hybrid_search = use_hybrid_search
        self.alpha = alpha

        # 初始化LLM
        self.llm = ChatZhipuAI(
            api_key=settings.ZHIPUAI_API_KEY,
            model="glm-4",  # 使用智谱GLM-4模型
            temperature=self.temperature,
        )

        # 创建提示模板
        self.prompt = PromptTemplate.from_template(self.QA_TEMPLATE)

        # 创建检索器
        self.retriever = self.vector_store_manager.vector_store.as_retriever(
            search_kwargs={"k": self.k}
        )

        # 使用 LCEL 创建 RAG 链
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # 初始化优化组件
        self.query_rewriter = None
        self.reranker = None
        self.hybrid_retriever = None

        if use_query_rewrite:
            self.query_rewriter = QueryRewriter(use_llm=True)
            logger.info("查询重写器已启用")

        if use_reranker:
            self.reranker = HybridReranker(use_cross_encoder=True)
            logger.info("重排序器已启用")

        if use_hybrid_search:
            self.hybrid_retriever = HybridRetriever(
                vector_store_manager=vector_store_manager,
                alpha=alpha,
                k=k
            )
            logger.info(f"混合检索已启用（alpha={alpha}）")

        logger.info("RAG链初始化完成（含优化组件）")

    def ask(
        self,
        question: str,
        return_source: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        提问并获取答案（使用所有优化技术）

        Args:
            question: 用户问题
            return_source: 是否返回来源文档
            use_cache: 是否使用缓存

        Returns:
            包含答案和来源的字典
        """
        try:
            # 检查缓存
            if use_cache:
                cached_answer = answer_cache.get(question)
                if cached_answer:
                    logger.info(f"从缓存返回答案: {question[:50]}...")
                    cached_answer["from_cache"] = True
                    return cached_answer

            # 步骤1: 查询重写
            original_question = question
            if self.query_rewriter:
                question = self.query_rewriter.rewrite(question)
                logger.info(f"查询重写: '{original_question}' -> '{question}'")

            # 步骤2: 检索（使用混合检索或纯向量检索）
            retrieved_docs = []

            if self.hybrid_retriever:
                # 混合检索 + 重排序
                retrieved_docs_with_scores = self.hybrid_retriever.retrieve_with_reranking(
                    query=question,
                    k=self.k,
                    reranker=self.reranker if self.use_reranker else None,
                    use_hybrid=self.use_hybrid_search
                )
                retrieved_docs = [doc for doc, _ in retrieved_docs_with_scores]
                logger.info(f"混合检索+重排序返回 {len(retrieved_docs)} 个文档")
            else:
                # 纯向量检索
                retrieved_docs = self.vector_store_manager.similarity_search(
                    query=question,
                    k=self.k
                )

                # 重排序
                if self.reranker:
                    retrieved_docs_with_scores = self.reranker.rerank(
                        question,
                        retrieved_docs,
                        top_k=self.k
                    )
                    retrieved_docs = [doc for doc, _ in retrieved_docs_with_scores]
                    logger.info(f"重排序后返回 {len(retrieved_docs)} 个文档")

            if not retrieved_docs:
                return {
                    "question": original_question,
                    "answer": "抱歉，知识库中没有找到相关内容。请尝试更换问题或联系管理员添加相关文档。",
                    "success": False,
                    "from_cache": False
                }

            # 步骤3: 生成答案
            answer = self.rag_chain.invoke(question)

            response = {
                "question": original_question,  # 返回原始问题
                "rewritten_question": question if question != original_question else None,
                "answer": answer,
                "success": True,
                "from_cache": False,
                "optimization_used": {
                    "query_rewrite": self.query_rewriter is not None and question != original_question,
                    "hybrid_search": self.use_hybrid_search,
                    "reranking": self.use_reranker
                }
            }

            # 添加来源文档
            if return_source:
                sources = []
                for doc in retrieved_docs:
                    source_info = {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)
                response["sources"] = sources

            # 保存到缓存
            if use_cache:
                answer_cache.set(original_question, response)

            logger.info(f"成功回答问题: {original_question[:50]}...")
            return response

        except Exception as e:
            logger.error(f"问答失败: {str(e)}")
            return {
                "question": question,
                "answer": f"抱歉，处理您的问题时出错: {str(e)}",
                "success": False,
                "from_cache": False
            }

    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        仅进行相似度搜索（不生成答案）

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            相似文档列表
        """
        try:
            results = self.vector_store_manager.similarity_search_with_score(
                query=query,
                k=k
            )

            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })

            return formatted_results

        except Exception as e:
            logger.error(f"相似度搜索失败: {str(e)}")
            return []

    def chat(
        self,
        question: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        对话模式（带历史记录）

        Args:
            question: 用户问题
            chat_history: 对话历史

        Returns:
            回答结果
        """
        # 目前简单实现，可以扩展为带历史的对话
        return self.ask(question)


class KnowledgeBase:
    """知识库主类"""

    def __init__(self):
        self.vector_store_manager = VectorStoreManager()

        # 从配置读取优化选项
        use_query_rewrite = settings.USE_QUERY_REWRITE
        use_reranker = settings.USE_RERANKER
        use_hybrid_search = settings.USE_HYBRID_SEARCH
        alpha = settings.HYBRID_ALPHA

        self.rag_chain = RAGChain(
            self.vector_store_manager,
            use_query_rewrite=use_query_rewrite,
            use_reranker=use_reranker,
            use_hybrid_search=use_hybrid_search,
            alpha=alpha
        )
        logger.info("知识库初始化完成")

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        return self.rag_chain.similarity_search(query, k)

    def ask(self, question: str) -> Dict[str, Any]:
        """提问"""
        return self.rag_chain.ask(question)

    def get_info(self) -> Dict[str, Any]:
        """获取知识库信息"""
        return self.vector_store_manager.get_collection_info()
