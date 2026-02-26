"""
API数据模型
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class LoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(..., description="用户名", min_length=1)
    password: str = Field(..., description="密码", min_length=1)


class LoginResponse(BaseModel):
    """登录响应"""
    success: bool
    message: str
    username: Optional[str] = None
    role: Optional[str] = None
    token: Optional[str] = None


class QuestionRequest(BaseModel):
    """问题请求"""
    question: str = Field(..., description="用户问题", min_length=1)
    use_rag: bool = Field(True, description="是否使用RAG生成答案")


class SearchResult(BaseModel):
    """搜索结果"""
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None
    images: Optional[List[str]] = None  # 添加图片字段


class AnswerResponse(BaseModel):
    """回答响应"""
    question: str
    answer: str
    success: bool
    sources: Optional[List[SearchResult]] = None


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(..., description="搜索查询", min_length=1)
    k: int = Field(4, description="返回结果数量", ge=1, le=10)


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    success: bool
    message: str
    document_count: int = 0
    file_name: Optional[str] = None


class UploadHistoryItem(BaseModel):
    """上传历史记录项"""
    id: int
    username: str
    filename: str
    file_size: int
    doc_count: int
    success: bool
    timestamp: str


class UploadHistoryResponse(BaseModel):
    """上传历史响应"""
    total: int
    records: List[UploadHistoryItem]


class KnowledgeBaseInfo(BaseModel):
    """知识库信息"""
    type: str
    collection_name: str
    initialized: bool
    count: Optional[int] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    message: str
