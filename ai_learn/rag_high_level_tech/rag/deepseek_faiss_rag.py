"""
DeepSeek + Faiss 本地知识库检索系统
=====================================

功能：
1. 加载本地文档（txt, pdf, docx）
2. 使用 DeepSeek API 生成 Embedding
3. 使用 Faiss 构建向量索引
4. 检索相关文档并用 DeepSeek 生成答案

作者: Claude Code Assistant
日期: 2026-01-27
"""

import os
import json
import pickle
from typing import List, Dict, Tuple
from pathlib import Path
import re

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()


# ============================================================================
# 1. 文档加载和预处理
# ============================================================================

class DocumentLoader:
    """文档加载器 - 支持多种格式"""

    @staticmethod
    def load_txt(file_path: str) -> str:
        """加载 TXT 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def load_pdf(file_path: str) -> str:
        """加载 PDF 文件"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except ImportError:
            print("⚠️  请安装 PyPDF2: pip install PyPDF2")
            return ""

    @staticmethod
    def load_docx(file_path: str) -> str:
        """加载 DOCX 文件"""
        try:
            import docx
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except ImportError:
            print("⚠️  请安装 python-docx: pip install python-docx")
            return ""

    @staticmethod
    def load_file(file_path: str) -> str:
        """根据文件扩展名自动加载"""
        ext = Path(file_path).suffix.lower()

        loaders = {
            '.txt': DocumentLoader.load_txt,
            '.pdf': DocumentLoader.load_pdf,
            '.docx': DocumentLoader.load_docx,
        }

        loader = loaders.get(ext)
        if loader:
            return loader(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    @staticmethod
    def load_directory(directory: str) -> List[Tuple[str, str]]:
        """
        加载目录下的所有文档

        返回: [(文件名, 文档内容), ...]
        """
        documents = []
        path = Path(directory)

        supported_exts = ['.txt', '.pdf', '.docx']

        for file_path in path.rglob('*'):
            if file_path.suffix.lower() in supported_exts:
                try:
                    content = DocumentLoader.load_file(str(file_path))
                    if content.strip():
                        documents.append((file_path.name, content))
                        print(f"✅ 已加载: {file_path.name}")
                except Exception as e:
                    print(f"❌ 加载失败 {file_path.name}: {e}")

        return documents


class TextSplitter:
    """文本分割器 - 将文档切分成小块"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        参数:
            chunk_size: 每块的字符数
            chunk_overlap: 块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, metadata: str = "") -> List[Dict]:
        """
        分割文本

        返回: [{"content": 文本块, "metadata": 元数据}, ...]
        """
        chunks = []

        # 清理文本
        text = re.sub(r'\n+', '\n', text)  # 合并多余换行
        text = text.strip()

        # 按段落分割（如果段落太长再按字符分割）
        paragraphs = text.split('\n\n')

        current_chunk = ""
        chunk_id = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前块加上新段落不超过限制
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                # 保存当前块
                if current_chunk.strip():
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": metadata,
                        "chunk_id": chunk_id
                    })
                    chunk_id += 1

                # 开始新块（如果有重叠，保留部分内容）
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + para + "\n\n"
                else:
                    current_chunk = para + "\n\n"

        # 保存最后一块
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": metadata,
                "chunk_id": chunk_id
            })

        return chunks

    def split_documents(self, documents: List[Tuple[str, str]]) -> List[Dict]:
        """
        分割多个文档

        参数:
            documents: [(文件名, 内容), ...]

        返回: [文本块字典, ...]
        """
        all_chunks = []

        for filename, content in documents:
            chunks = self.split_text(content, metadata=filename)
            all_chunks.extend(chunks)

        return all_chunks


# ============================================================================
# 2. DeepSeek Embedding
# ============================================================================

class DeepSeekEmbedding:
    """DeepSeek Embedding 生成器"""

    def __init__(self, api_key: str = None, base_url: str = None):
        """
        初始化 DeepSeek 客户端

        DeepSeek API 兼容 OpenAI SDK
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本的 Embedding

        注意：DeepSeek 可能需要通过 chat 模型生成 embedding
        这里提供兼容接口
        """
        try:
            # 尝试使用 embeddings endpoint（如果 DeepSeek 支持）
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",  # 或 DeepSeek 的 embedding 模型
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"⚠️  DeepSeek embedding 调用失败: {e}")
            print("💡 建议：使用 sentence-transformers 作为替代")
            # 返回零向量作为 fallback
            return [0.0] * 1536

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        批量获取 Embedding
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                emb = self.get_embedding(text)
                embeddings.append(emb)
                print(f"✅ 生成 embedding {len(embeddings)}/{len(texts)}")

        return embeddings


# ============================================================================
# 3. Faiss 索引
# ============================================================================

class FaissIndex:
    """Faiss 向量索引管理器"""

    def __init__(self, dimension: int = 1536):
        """
        参数:
            dimension: 向量维度（1536 for OpenAI ada-002）
        """
        self.dimension = dimension
        self.index = None
        self.chunks = []  # 存储文本块

    def build_index(self, chunks: List[Dict], embeddings: List[List[float]]):
        """
        构建 Faiss 索引

        参数:
            chunks: 文本块列表
            embeddings: 对应的 embedding 列表
        """
        self.chunks = chunks

        # 转换为 numpy 数组
        embeddings_array = np.array(embeddings, dtype='float32')

        # 确保维度正确
        if embeddings_array.shape[1] != self.dimension:
            print(f"⚠️  向量维度不匹配: 期望 {self.dimension}, 实际 {embeddings_array.shape[1]}")
            self.dimension = embeddings_array.shape[1]

        # 创建索引（使用 L2 距离）
        self.index = faiss.IndexFlatL2(self.dimension)

        # 添加向量
        self.index.add(embeddings_array)

        print(f"✅ Faiss 索引构建完成: {len(chunks)} 个文档块")

    def save(self, index_path: str = "faiss_index.bin", data_path: str = "chunks.pkl"):
        """
        保存索引和数据
        """
        if self.index is None:
            raise ValueError("索引未构建，无法保存")

        # 保存 Faiss 索引
        faiss.write_index(self.index, index_path)

        # 保存文本块数据
        with open(data_path, 'wb') as f:
            pickle.dump(self.chunks, f)

        print(f"✅ 索引已保存: {index_path}, {data_path}")

    def load(self, index_path: str = "faiss_index.bin", data_path: str = "chunks.pkl"):
        """
        加载索引和数据
        """
        # 加载 Faiss 索引
        self.index = faiss.read_index(index_path)

        # 加载文本块数据
        with open(data_path, 'rb') as f:
            self.chunks = pickle.load(f)

        self.dimension = self.index.d

        print(f"✅ 索引已加载: {len(self.chunks)} 个文档块")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        搜索最相似的文档块

        返回: [{"content": 文本, "metadata": 元数据, "score": 相似度分数}, ...]
        """
        if self.index is None:
            raise ValueError("索引未构建，请先构建或加载索引")

        # 转换查询向量
        query_array = np.array([query_embedding], dtype='float32')

        # 搜索
        distances, indices = self.index.search(query_array, top_k)

        # 整理结果
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(dist)
                results.append(chunk)

        return results


# ============================================================================
# 4. DeepSeek 问答生成
# ============================================================================

class DeepSeekChat:
    """DeepSeek 问答生成器"""

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def generate_answer(self, question: str, context: str) -> str:
        """
        基于上下文生成答案
        """
        prompt = f"""你是一个专业的问答助手。请基于以下参考文档回答用户的问题。

参考文档：
{context}

问题：{question}

要求：
1. 答案必须基于参考文档中的信息
2. 如果文档中没有相关信息，请明确说明
3. 保持答案简洁准确

答案："""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的问答助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"⚠️  生成答案时出错: {e}"


# ============================================================================
# 5. 完整的 RAG 系统
# ============================================================================

class RAGSystem:
    """完整的 RAG 系统"""

    def __init__(self):
        self.embedder = DeepSeekEmbedding()
        self.chat = DeepSeekChat()
        self.index = FaissIndex()

    def build_knowledge_base(self, documents_dir: str):
        """
        构建知识库

        参数:
            documents_dir: 文档目录路径
        """
        print("\n" + "=" * 80)
        print("📚 第一步：加载文档")
        print("=" * 80)

        # 1. 加载文档
        loader = DocumentLoader()
        documents = loader.load_directory(documents_dir)

        if not documents:
            print("❌ 未找到任何文档")
            return

        print(f"\n✅ 共加载 {len(documents)} 个文档")

        # 2. 分割文本
        print("\n" + "=" * 80)
        print("✂️  第二步：分割文本")
        print("=" * 80)

        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        print(f"✅ 共分割成 {len(chunks)} 个文本块")

        # 3. 生成 Embedding
        print("\n" + "=" * 80)
        print("🔄 第三步：生成 Embedding")
        print("=" * 80)

        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedder.get_embeddings_batch(texts, batch_size=10)

        # 4. 构建 Faiss 索引
        print("\n" + "=" * 80)
        print("🔨 第四步：构建 Faiss 索引")
        print("=" * 80)

        self.index.build_index(chunks, embeddings)

        # 5. 保存索引
        print("\n" + "=" * 80)
        print("💾 第五步：保存索引")
        print("=" * 80)

        self.index.save()
        print("\n✨ 知识库构建完成！")

    def load_knowledge_base(self):
        """加载已构建的知识库"""
        self.index.load()
        print("✅ 知识库已加载")

    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        查询知识库

        返回: {"answer": 答案, "sources": [来源文档], "query": 问题}
        """
        print(f"\n🔍 查询问题: {question}")

        # 1. 生成问题 embedding
        query_embedding = self.embedder.get_embedding(question)

        # 2. 检索相关文档
        results = self.index.search(query_embedding, top_k=top_k)

        print(f"📚 找到 {len(results)} 个相关文档:")

        for i, result in enumerate(results):
            print(f"\n  [{i+1}] {result['metadata']} (分数: {result['score']:.4f})")
            print(f"      {result['content'][:100]}...")

        # 3. 组装上下文
        context = "\n\n".join([
            f"【来源: {r['metadata']}】\n{r['content']}"
            for r in results
        ])

        # 4. 生成答案
        print("\n💭 正在生成答案...")
        answer = self.chat.generate_answer(question, context)

        return {
            "answer": answer,
            "sources": results,
            "query": question
        }


# ============================================================================
# 6. 主程序
# ============================================================================

def main():
    """主程序演示"""

    print("=" * 80)
    print("🤖 DeepSeek + Faiss 本地知识库检索系统")
    print("=" * 80)

    # 检查 API Key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("\n❌ 错误：未设置 DEEPSEEK_API_KEY")
        print("\n请按以下步骤配置：")
        print("1. 复制 .env.example 为 .env")
        print("2. 在 .env 中设置你的 DeepSeek API Key")
        print("\n获取 API Key: https://platform.deepseek.com/")
        return

    # 创建 RAG 系统
    rag = RAGSystem()

    # 创建示例文档目录
    docs_dir = "knowledge_base"
    Path(docs_dir).mkdir(exist_ok=True)

    # 检查是否已有文档
    if not list(Path(docs_dir).rglob('*')):
        print(f"\n📝 在 {docs_dir}/ 目录中添加你的文档（txt, pdf, docx）")
        print("然后重新运行程序")
        return

    # 构建或加载知识库
    if Path("faiss_index.bin").exists():
        print("\n检测到已有索引，是否重新构建？(y/n): ", end="")
        choice = input().strip().lower()

        if choice == 'y':
            rag.build_knowledge_base(docs_dir)
        else:
            rag.load_knowledge_base()
    else:
        rag.build_knowledge_base(docs_dir)

    # 交互式问答
    print("\n" + "=" * 80)
    print("💬 开始问答（输入 'quit' 退出）")
    print("=" * 80)

    while True:
        print("\n" + "─" * 80)
        question = input("❓ 你的问题: ").strip()

        if not question:
            continue

        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 再见！")
            break

        # 查询
        result = rag.query(question, top_k=3)

        # 显示答案
        print("\n" + "─" * 80)
        print("📖 答案:")
        print("─" * 80)
        print(result['answer'])
        print("─" * 80)


if __name__ == "__main__":
    main()
