"""
嵌入模型配置
使用智谱AI的embedding模型
"""
from langchain_community.embeddings import ZhipuAIEmbeddings
from backend.core.config import settings


def get_embeddings():
    """获取智谱AI嵌入模型"""
    return ZhipuAIEmbeddings(
        api_key=settings.ZHIPUAI_API_KEY,
        model="embedding-2"  # 智谱AI的embedding模型
    )
