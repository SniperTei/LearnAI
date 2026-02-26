"""
查询重写模块
优化用户查询以提升检索效果
"""
from typing import List, Dict, Any
import logging
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatZhipuAI
from backend.core.config import settings

logger = logging.getLogger(__name__)


class QueryRewriter:
    """查询重写器"""

    def __init__(self, use_llm: bool = True):
        """
        初始化查询重写器

        Args:
            use_llm: 是否使用LLM进行重写
        """
        self.use_llm = use_llm
        self.llm = None

        if use_llm:
            try:
                self.llm = ChatZhipuAI(
                    api_key=settings.ZHIPUAI_API_KEY,
                    model="glm-4",
                    temperature=0.1  # 低温度保证稳定性
                )
                logger.info("查询重写器初始化成功（使用LLM）")
            except Exception as e:
                logger.error(f"初始化LLM失败: {e}")
                self.use_llm = False
        else:
            logger.info("查询重写器初始化成功（基于规则）")

    def rewrite(self, query: str, context: str = None) -> str:
        """
        重写查询

        Args:
            query: 原始查询
            context: 可选的上下文信息

        Returns:
            重写后的查询
        """
        if not self.use_llm or not self.llm:
            return self._rule_based_rewrite(query)

        return self._llm_based_rewrite(query, context)

    def _rule_based_rewrite(self, query: str) -> str:
        """
        基于规则的查询重写

        Args:
            query: 原始查询

        Returns:
            重写后的查询
        """
        # 简单的规则扩展
        query_lower = query.lower()

        # 添加常见同义词
        expansions = {
            "怎么": "如何 方法 步骤",
            "如何": "怎么 方法 步骤",
            "什么是": "定义 概念 介绍",
            "报销": "费用 发票 财务",
            "请假": "休假 申请 流程",
            "公司": "企业 单位 集团",
            "系统": "平台 软件 应用"
        }

        # 检查是否需要扩展
        for keyword, synonyms in expansions.items():
            if keyword in query:
                # 添加同义词（最多添加一次）
                added = False
                for synonym in synonyms.split():
                    if synonym not in query_lower and not added:
                        query = f"{query} {synonym}"
                        added = True
                        break  # 只添加一个同义词

        return query

    def _llm_based_rewrite(self, query: str, context: str = None) -> str:
        """
        基于LLM的查询重写

        Args:
            query: 原始查询
            context: 可选的上下文

        Returns:
            重写后的查询
        """
        try:
            prompt = PromptTemplate.from_template("""
你是一个查询优化专家。请重写用户的查询，使其更适合在知识库中检索相关信息。

**原始查询：** {query}

**重写要求：**
1. 保持原意，但使查询更完整、更具体
2. 添加相关的关键词（如公司信息、流程步骤等）
3. 如果查询过于简短，补充可能的上下文
4. 如果查询过于口语化，转换为更专业的表达
5. 只返回重写后的查询，不要解释

**重写后的查询：**
""")

            result = self.llm.invoke(prompt.format(query=query))

            if isinstance(result, str):
                rewritten = result.strip()
            else:
                rewritten = result.content.strip()

            logger.info(f"查询重写: '{query}' -> '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.error(f"LLM查询重写失败: {e}")
            return query

    def rewrite_with_history(
        self,
        query: str,
        history: List[Dict[str, str]] = None
    ) -> str:
        """
        结合对话历史的查询重写

        Args:
            query: 当前查询
            history: 对话历史 [{"role": "user", "content": "..."}]

        Returns:
            重写后的查询
        """
        if not history or not self.use_llm:
            return self.rewrite(query)

        try:
            # 构建历史上下文
            history_text = "\n".join([
                f"{item['role']}: {item['content']}"
                for item in history[-3:]  # 只使用最近3轮对话
            ])

            prompt = PromptTemplate.from_template("""
你是一个查询优化专家。请根据对话历史重写用户的查询，使其更完整、更明确。

**对话历史：**
{history}

**当前查询：** {query}

**重写要求：**
1. 理解对话上下文，将代词（如"它"、"那个"）替换为具体指代
2. 如果当前查询不完整，结合历史信息补充完整
3. 保持原意，使查询更清晰
4. 只返回重写后的查询，不要解释

**重写后的查询：**
""")

            result = self.llm.invoke(prompt.format(
                history=history_text,
                query=query
            ))

            if isinstance(result, str):
                rewritten = result.strip()
            else:
                rewritten = result.content.strip()

            logger.info(f"基于历史的查询重写: '{query}' -> '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.error(f"基于历史的查询重写失败: {e}")
            return self.rewrite(query)


class QueryExpansion:
    """查询扩展器，添加同义词和相关词"""

    def __init__(self):
        """初始化查询扩展器"""
        # 预定义的同义词词典
        self.synonyms = {
            "报销": ["费用", "发票", "财务", "会计", "审批"],
            "请假": ["休假", "假期", "申请", "批准"],
            "公司": ["企业", "单位", "集团", "组织"],
            "流程": ["步骤", "方法", "途径", "操作"],
            "系统": ["平台", "软件", "应用", "工具"],
            "怎么": ["如何", "方法", "方式"],
            "什么": ["哪些", "如何", "怎样"],
            "为什么": ["原因", "理由"],
            "增值税": ["发票", "税收", "税务"],
            "发票": ["票据", "凭证", "收据"],
        }

    def expand(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        扩展查询，生成多个变体

        Args:
            query: 原始查询
            max_expansions: 最大扩展数量

        Returns:
            扩展后的查询列表（包含原始查询）
        """
        queries = [query]

        for word, synonyms in self.synonyms.items():
            if word in query:
                for synonym in synonyms[:max_expansions]:
                    expanded = query.replace(word, synonym, 1)
                    if expanded not in queries:
                        queries.append(expanded)

                # 只扩展第一个匹配的词
                break

        return queries
