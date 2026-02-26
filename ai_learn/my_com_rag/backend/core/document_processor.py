"""
文档处理模块
负责解析、分块各种类型的文档
"""
import os
from pathlib import Path
from typing import List, Optional, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
)
from langchain_core.documents import Document
from backend.core.config import settings
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """文档处理器"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )

        # 图片存储目录
        self.images_dir = Path("data/images")
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # 支持的文件类型及其加载器
        self.loaders = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".doc": Docx2txtLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".markdown": UnstructuredMarkdownLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".xls": UnstructuredExcelLoader,
        }

    def extract_images_from_pdf(self, pdf_path: str) -> List[str]:
        """
        从PDF文件中提取图片

        Args:
            pdf_path: PDF文件路径

        Returns:
            图片文件路径列表
        """
        image_paths = []
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            pdf_name = Path(pdf_path).stem

            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)

                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)

                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        image_filename = f"{pdf_name}_page{page_num+1}_img{img_index+1}.{image_ext}"
                        image_path = self.images_dir / image_filename

                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        # 返回相对路径用于API访问
                        image_paths.append(f"/api/images/{image_filename}")
                        logger.info(f"提取图片: {image_filename}")

            doc.close()

        except ImportError:
            logger.warning("PyMuPDF未安装，无法从PDF提取图片。运行: pip install PyMuPDF")
        except Exception as e:
            logger.error(f"从PDF提取图片失败: {str(e)}")

        return image_paths

    def extract_images_from_docx(self, docx_path: str) -> List[str]:
        """
        从DOCX文件中提取图片

        Args:
            docx_path: DOCX文件路径

        Returns:
            图片文件路径列表
        """
        image_paths = []
        try:
            from docx import Document
            from docx.oxml.inline import CT_Picture

            doc = Document(docx_path)
            docx_name = Path(docx_path).stem
            img_index = 0

            # 从文档关系中提取图片
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        image_data = rel.target_part.blob
                        image_ext = rel.target_ref.split('.')[-1]
                        image_filename = f"{docx_name}_img{img_index+1}.{image_ext}"
                        image_path = self.images_dir / image_filename

                        with open(image_path, "wb") as img_file:
                            img_file.write(image_data)

                        image_paths.append(f"/api/images/{image_filename}")
                        img_index += 1
                        logger.info(f"提取图片: {image_filename}")

                    except Exception as e:
                        logger.warning(f"提取单个图片失败: {str(e)}")

        except ImportError:
            logger.warning("python-docx未安装，无法从DOCX提取图片。运行: pip install python-docx")
        except Exception as e:
            logger.error(f"从DOCX提取图片失败: {str(e)}")

        return image_paths

    def load_document(self, file_path: str) -> List[Document]:
        """
        加载单个文档

        Args:
            file_path: 文档路径

        Returns:
            文档列表
        """
        try:
            file_path_obj = Path(file_path)
            file_ext = file_path_obj.suffix.lower()

            if file_ext not in self.loaders:
                logger.warning(f"不支持的文件类型: {file_ext}")
                return []

            # 提取图片
            images = []
            if file_ext == ".pdf":
                images = self.extract_images_from_pdf(str(file_path_obj))
            elif file_ext in [".docx", ".doc"]:
                images = self.extract_images_from_docx(str(file_path_obj))

            loader_class = self.loaders[file_ext]
            loader = loader_class(str(file_path_obj))
            documents = loader.load()

            # 添加元数据
            for doc in documents:
                doc.metadata["source"] = str(file_path_obj)
                doc.metadata["file_type"] = file_ext
                doc.metadata["file_name"] = file_path_obj.name
                # 添加图片信息到元数据
                if images:
                    doc.metadata["images"] = images

            if images:
                logger.info(f"从文档中提取了 {len(images)} 张图片")

            logger.info(f"成功加载文档: {file_path}")
            return documents

        except Exception as e:
            logger.error(f"加载文档失败 {file_path}: {str(e)}")
            return []

    def load_documents_from_directory(
        self,
        directory: str
    ) -> List[Document]:
        """
        从目录加载所有文档

        Args:
            directory: 目录路径

        Returns:
            所有文档列表
        """
        all_documents = []
        directory_path = Path(directory)

        if not directory_path.exists():
            logger.warning(f"目录不存在: {directory}")
            return all_documents

        # 遍历目录中的所有文件
        for file_path in directory_path.rglob("*"):
            if file_path.is_file():
                documents = self.load_document(str(file_path))
                all_documents.extend(documents)

        logger.info(f"从目录 {directory} 加载了 {len(all_documents)} 个文档")
        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分块文档

        Args:
            documents: 原始文档列表

        Returns:
            分块后的文档列表
        """
        if not documents:
            return []

        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"文档分块完成: {len(documents)} -> {len(split_docs)} 个块")
        return split_docs

    def process_file(self, file_path: str) -> List[Document]:
        """
        处理单个文件：加载并分块

        Args:
            file_path: 文件路径

        Returns:
            处理后的文档块列表
        """
        documents = self.load_document(file_path)
        if not documents:
            return []

        return self.split_documents(documents)

    def process_directory(self, directory: str) -> List[Document]:
        """
        处理整个目录：加载并分块

        Args:
            directory: 目录路径

        Returns:
            处理后的文档块列表
        """
        documents = self.load_documents_from_directory(directory)
        if not documents:
            return []

        return self.split_documents(documents)
