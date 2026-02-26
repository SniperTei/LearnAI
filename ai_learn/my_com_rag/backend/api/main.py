"""
FastAPI主应用
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
from typing import Optional
import uvicorn
import logging
from pathlib import Path
import shutil
import aiofiles

from backend.api.models import (
    LoginRequest,
    LoginResponse,
    QuestionRequest,
    AnswerResponse,
    SearchRequest,
    DocumentUploadResponse,
    UploadHistoryResponse,
    KnowledgeBaseInfo,
    HealthResponse
)
from backend.core.config import settings
from backend.core.document_processor import DocumentProcessor
from backend.core.vector_store import VectorStoreManager
from backend.core.rag_chain import KnowledgeBase
from backend.core.auth import user_manager
from backend.core.cache import answer_cache

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局知识库实例
kb_instance = None


def get_current_user(username: Optional[str] = Header(None)) -> Optional[str]:
    """从请求头获取当前用户"""
    if not username:
        return None
    return username


def require_auth(username: Optional[str] = Depends(get_current_user)) -> str:
    """要求认证"""
    if not username:
        raise HTTPException(status_code=401, detail="未登录，请先登录")
    return username


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global kb_instance

    # 启动时初始化
    logger.info("正在初始化知识库...")
    kb_instance = KnowledgeBase()
    logger.info("知识库初始化完成")

    yield

    # 关闭时清理
    logger.info("正在关闭应用...")


# 创建FastAPI应用
app = FastAPI(
    title="企业知识库 API",
    description="基于RAG的企业智能知识库系统",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """根路径"""
    return HealthResponse(
        status="healthy",
        message="企业知识库API正在运行"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy",
        message="服务正常"
    )


@app.get("/api/images/{image_name}")
async def get_image(image_name: str):
    """
    获取提取的图片
    """
    image_path = Path("data/images") / image_name

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="图片不存在")

    return FileResponse(image_path)


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    提问接口

    使用RAG生成答案
    """
    try:
        if request.use_rag:
            # 使用RAG生成答案
            result = kb_instance.ask(request.question)

            # 处理sources，添加图片信息
            if "sources" in result and result["sources"]:
                for source in result["sources"]:
                    if isinstance(source, dict):
                        # 如果metadata中有images，添加到结果中
                        if "metadata" in source and "images" in source["metadata"]:
                            source["images"] = source["metadata"]["images"]

            return AnswerResponse(**result)
        else:
            # 仅搜索
            results = kb_instance.search(request.question, k=4)

            # 处理搜索结果，添加图片信息
            formatted_results = []
            for result in results:
                result_dict = {
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("score")
                }
                # 如果metadata中有images，添加到结果中
                if "images" in result.get("metadata", {}):
                    result_dict["images"] = result["metadata"]["images"]
                formatted_results.append(result_dict)

            return AnswerResponse(
                question=request.question,
                answer="找到以下相关文档：",
                success=True,
                sources=formatted_results
            )

    except Exception as e:
        logger.error(f"处理问题失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
async def search(request: SearchRequest):
    """
    搜索接口

    仅进行语义搜索，不生成答案
    """
    try:
        results = kb_instance.search(request.query, k=request.k)

        # 处理搜索结果，添加图片信息
        formatted_results = []
        for result in results:
            result_dict = {
                "content": result.get("content", ""),
                "metadata": result.get("metadata", {}),
                "score": result.get("score")
            }
            # 如果metadata中有images，添加到结果中
            if "images" in result.get("metadata", {}):
                result_dict["images"] = result["metadata"]["images"]
            formatted_results.append(result_dict)

        return {
            "query": request.query,
            "results": formatted_results,
            "count": len(formatted_results)
        }
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    用户登录接口
    """
    try:
        user = user_manager.verify_user(request.username, request.password)
        if user:
            return LoginResponse(
                success=True,
                message="登录成功",
                username=user["username"],
                role=user["role"],
                token=user["username"]  # 简化：使用用户名作为token
            )
        else:
            return LoginResponse(
                success=False,
                message="用户名或密码错误"
            )
    except Exception as e:
        logger.error(f"登录失败: {str(e)}")
        return LoginResponse(
            success=False,
            message="登录失败"
        )


@app.post("/api/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    username: str = Depends(require_auth)
):
    """
    上传文档接口（需要管理员权限）

    支持的格式: PDF, DOCX, TXT, MD, XLSX
    """
    try:
        # 检查权限
        if not user_manager.can_upload(username):
            raise HTTPException(status_code=403, detail="只有管理员才能上传文件")

        # 保存上传的文件
        file_path = Path(settings.UPLOAD_PATH) / file.filename

        # 保存文件
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            file_size = len(content)
            await f.write(content)

        logger.info(f"用户 {username} 上传文件: {file_path}")

        # 处理文档
        processor = DocumentProcessor()
        documents = processor.process_file(str(file_path))

        if not documents:
            # 记录失败历史
            user_manager.add_upload_history(
                username=username,
                filename=file.filename,
                file_size=file_size,
                doc_count=0,
                success=False
            )
            return DocumentUploadResponse(
                success=False,
                message=f"无法处理文件: {file.filename}",
                file_name=file.filename
            )

        # 添加到向量数据库
        vector_manager = VectorStoreManager()
        success = vector_manager.add_documents(documents)

        # 记录上传历史
        user_manager.add_upload_history(
            username=username,
            filename=file.filename,
            file_size=file_size,
            doc_count=len(documents),
            success=success
        )

        if success:
            return DocumentUploadResponse(
                success=True,
                message=f"成功上传并处理文档: {file.filename}",
                document_count=len(documents),
                file_name=file.filename
            )
        else:
            return DocumentUploadResponse(
                success=False,
                message="文档处理失败",
                file_name=file.filename
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/upload/history", response_model=UploadHistoryResponse)
async def get_upload_history(
    username: str = Depends(require_auth),
    limit: int = 100
):
    """
    获取上传历史记录

    管理员可以查看所有记录，普通用户只能查看自己的记录
    """
    try:
        # 检查是否为管理员
        is_admin = user_manager.get_user_role(username) == "admin"

        # 获取历史记录
        if is_admin:
            # 管理员查看所有记录
            records = user_manager.get_upload_history(limit=limit)
        else:
            # 普通用户只看自己的记录
            records = user_manager.get_upload_history(username=username, limit=limit)

        return UploadHistoryResponse(
            total=len(records),
            records=records
        )
    except Exception as e:
        logger.error(f"获取上传历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/load-directory")
async def load_directory(directory: str = None):
    """
    从目录加载所有文档

    如果不指定目录，使用配置文件中的默认目录
    """
    try:
        target_dir = directory or settings.DOCUMENTS_PATH

        if not Path(target_dir).exists():
            raise HTTPException(
                status_code=404,
                detail=f"目录不存在: {target_dir}"
            )

        # 处理目录中的所有文档
        processor = DocumentProcessor()
        documents = processor.process_directory(target_dir)

        if not documents:
            return {
                "success": True,
                "message": "目录中没有找到可处理的文档",
                "document_count": 0,
                "directory": target_dir
            }

        # 添加到向量数据库
        vector_manager = VectorStoreManager()
        success = vector_manager.add_documents(documents)

        if success:
            return {
                "success": True,
                "message": f"成功从目录加载文档",
                "document_count": len(documents),
                "directory": target_dir
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="添加文档到向量数据库失败"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"加载目录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/info", response_model=KnowledgeBaseInfo)
async def get_kb_info():
    """
    获取知识库信息
    """
    try:
        info = kb_instance.get_info()
        return KnowledgeBaseInfo(**info)
    except Exception as e:
        logger.error(f"获取知识库信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/clear")
async def clear_knowledge_base():
    """
    清空知识库

    删除所有文档和向量索引
    """
    try:
        vector_manager = VectorStoreManager()
        success = vector_manager.delete_collection()

        if success:
            return {
                "success": True,
                "message": "知识库已清空"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="清空知识库失败"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"清空知识库失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cache/stats")
async def get_cache_stats():
    """
    获取缓存统计信息
    """
    try:
        stats = answer_cache.get_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"获取缓存统计失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cache/clear")
async def clear_cache():
    """
    清空所有缓存
    """
    try:
        answer_cache.clear()
        return {
            "success": True,
            "message": "缓存已清空"
        }
    except Exception as e:
        logger.error(f"清空缓存失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cache/clear-expired")
async def clear_expired_cache():
    """
    清理过期缓存
    """
    try:
        answer_cache.clear_expired()
        return {
            "success": True,
            "message": "过期缓存已清理"
        }
    except Exception as e:
        logger.error(f"清理过期缓存失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """运行应用"""
    uvicorn.run(
        "backend.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
