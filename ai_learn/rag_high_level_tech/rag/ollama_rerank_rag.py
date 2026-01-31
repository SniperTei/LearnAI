"""
Ollama + TF-IDF + Rerank 本地知识库检索系统
=============================================

使用本地 Ollama 模型 + TF-IDF + Rerank 构建的本地知识库问答系统。
演示 RAG 高效召回方法：重排序（Reranking）

功能：
1. 使用 TF-IDF 进行粗排（快速召回）
2. 使用 Rerank 模型进行精排（提高准确度）
3. 检索相关文档并用 Ollama 生成答案

作者: Claude Code Assistant
日期: 2026-01-27
"""

import os
import json
import pickle
from typing import List, Dict, Tuple
from pathlib import Path
import re
import requests

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# 配置
# ============================================================================

OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_MODEL = "deepseek-r1:1.5b"  # 你的本地模型

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
        """加载目录下的所有文档"""
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
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, metadata: str = "") -> List[Dict]:
        """分割文本"""
        chunks = []
        text = re.sub(r'\n+', '\n', text)
        text = text.strip()

        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_id = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": metadata,
                        "chunk_id": chunk_id
                    })
                    chunk_id += 1

                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + para + "\n\n"
                else:
                    current_chunk = para + "\n\n"

        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": metadata,
                "chunk_id": chunk_id
            })

        return chunks

    def split_documents(self, documents: List[Tuple[str, str]]) -> List[Dict]:
        """分割多个文档"""
        all_chunks = []
        for filename, content in documents:
            chunks = self.split_text(content, metadata=filename)
            all_chunks.extend(chunks)
        return all_chunks


# ============================================================================
# 2. TF-IDF 检索器（粗排）
# ============================================================================

class TFIDFRetriever:
    """TF-IDF 检索器 - 用于粗排"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=None,
            ngram_range=(1, 2)
        )
        self.chunks = []
        self.tfidf_matrix = None

    def build_index(self, chunks: List[Dict]):
        """构建 TF-IDF 索引"""
        self.chunks = chunks
        texts = [chunk['content'] for chunk in chunks]

        print("🔄 正在构建 TF-IDF 索引...")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        print(f"✅ TF-IDF 索引构建完成: {len(chunks)} 个文档块")
        print(f"   特征维度: {self.tfidf_matrix.shape[1]}")

    def save(self, matrix_path: str = "tfidf_matrix.pkl", data_path: str = "chunks.pkl"):
        """保存索引和数据"""
        with open(matrix_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }, f)

        with open(data_path, 'wb') as f:
            pickle.dump(self.chunks, f)

        print(f"✅ 索引已保存: {matrix_path}, {data_path}")

    def load(self, matrix_path: str = "tfidf_matrix.pkl", data_path: str = "chunks.pkl"):
        """加载索引和数据"""
        with open(matrix_path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.tfidf_matrix = data['tfidf_matrix']

        with open(data_path, 'rb') as f:
            self.chunks = pickle.load(f)

        print(f"✅ 索引已加载: {len(self.chunks)} 个文档块")

    def search(self, query: str, top_k: int = 100) -> List[Dict]:
        """
        粗排：快速检索 top-k 个候选文档

        注意：这里召回更多文档（如100个），为 rerank 做准备
        """
        if self.tfidf_matrix is None:
            raise ValueError("索引未构建，请先构建或加载索引")

        # 将查询转换为 TF-IDF 向量
        query_vector = self.vectorizer.transform([query])

        # 计算余弦相似度
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]

        # 获取 top-k 最相似的（召回更多用于 rerank）
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['coarse_score'] = float(similarities[idx])  # 粗排分数
            results.append(chunk)

        return results


# ============================================================================
# 3. Rerank 模型（精排）⭐ 新增
# ============================================================================

class Reranker:
    """
    重排序器 - 对粗排结果进行精排

    方法1: 关键词匹配增强（快速）
    方法2: 使用模型打分（准确但慢）
    """

    def __init__(self, method: str = "keyword"):
        """
        参数:
            method: rerank 方法
                - "keyword": 基于关键词匹配（快速，推荐）
                - "model": 使用 LLM 打分（准确但慢）
        """
        self.method = method

    def _keyword_rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        基于关键词匹配的 rerank

        原理：
        1. 提取查询中的关键词
        2. 计算关键词在文档中的出现频率
        3. 结合粗排分数重新排序
        """
        # 提取查询关键词（简单的分词）
        query_keywords = set(query.lower().split())

        for doc in documents:
            content = doc['content'].lower()

            # 计算关键词匹配分数
            keyword_matches = 0
            for keyword in query_keywords:
                if keyword in content:
                    keyword_matches += content.count(keyword)

            # 计算 rerank 分数：粗排分数 * (1 + 关键词匹配权重)
            doc['rerank_score'] = doc['coarse_score'] * (1 + keyword_matches * 0.1)
            doc['keyword_matches'] = keyword_matches

        # 按 rerank 分数排序
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

        return reranked

    def _model_rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        使用 LLM 进行 rerank

        原理：
        让模型对每个文档的相关性打分（0-10）

        注意：这种方法速度较慢，但更准确
        """
        print(f"🔄 使用模型进行 rerank ({len(documents)} 个文档)...")

        for i, doc in enumerate(documents):
            # 构建打分 prompt
            prompt = f"""请评分查询和文档的相关性（0-10分）：

查询：{query}

文档：{doc['content'][:200]}...

请只输出一个0-10的数字分数："""

            try:
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": CHAT_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_predict": 5}
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    score_text = result.get("response", "5").strip()

                    # 尝试提取数字
                    import re
                    score_match = re.search(r'\d+(\.\d+)?', score_text)
                    if score_match:
                        score = float(score_match.group())
                    else:
                        score = 5.0  # 默认分数

                    doc['rerank_score'] = score / 10.0  # 归一化到 0-1
                else:
                    doc['rerank_score'] = doc['coarse_score']  # fallback

                print(f"  [{i+1}/{len(documents)}] 打分完成: {doc['rerank_score']:.2f}")

            except Exception as e:
                print(f"  ⚠️  文档 {i+1} 打分失败: {e}")
                doc['rerank_score'] = doc['coarse_score']  # fallback

        # 按 rerank 分数排序
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

        return reranked

    def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        对文档进行重排序

        参数:
            query: 用户查询
            documents: 粗排结果列表

        返回:
            重排序后的文档列表
        """
        if self.method == "keyword":
            print("🔄 使用关键词匹配进行 rerank...")
            return self._keyword_rerank(query, documents)
        elif self.method == "model":
            return self._model_rerank(query, documents)
        else:
            raise ValueError(f"未知的 rerank 方法: {self.method}")


# ============================================================================
# 4. Ollama 问答生成
# ============================================================================

class OllamaChat:
    """Ollama 问答生成器"""

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = CHAT_MODEL):
        self.base_url = base_url
        self.model = model

        # 检查模型是否可用
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                print(f"✅ Ollama 可用模型: {', '.join(model_names)}")
        except Exception as e:
            print(f"⚠️  无法连接到 Ollama: {e}")

    def generate_answer(self, question: str, context: str) -> str:
        """基于上下文生成答案"""
        prompt = f"""你是一个专业的问答助手。请基于以下参考文档回答用户的问题。

参考文档：
{context}

问题：{question}

要求：
1. 答案必须基于参考文档中的信息
2. 如果文档中没有相关信息，请明确说明"提供的文档中没有包含该问题的答案"
3. 保持答案简洁准确

答案："""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1000
                    }
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "⚠️  无法生成答案")
            else:
                return f"⚠️  生成答案时出错: {response.status_code}"

        except Exception as e:
            return f"⚠️  生成答案时出错: {e}"


# ============================================================================
# 5. 完整的 RAG 系统（带 Rerank）
# ============================================================================

class RAGSystemWithRerank:
    """带 Rerank 的 RAG 系统"""

    def __init__(self, rerank_method: str = "keyword"):
        self.retriever = TFIDFRetriever()
        self.reranker = Reranker(method=rerank_method)
        self.chat = OllamaChat()

    def build_knowledge_base(self, documents_dir: str):
        """构建知识库"""
        print("\n" + "=" * 80)
        print("📚 第一步：加载文档")
        print("=" * 80)

        loader = DocumentLoader()
        documents = loader.load_directory(documents_dir)

        if not documents:
            print("❌ 未找到任何文档")
            return

        print(f"\n✅ 共加载 {len(documents)} 个文档")

        print("\n" + "=" * 80)
        print("✂️  第二步：分割文本")
        print("=" * 80)

        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        print(f"✅ 共分割成 {len(chunks)} 个文本块")

        print("\n" + "=" * 80)
        print("🔨 第三步：构建 TF-IDF 索引")
        print("=" * 80)

        self.retriever.build_index(chunks)

        print("\n" + "=" * 80)
        print("💾 第四步：保存索引")
        print("=" * 80)

        self.retriever.save()
        print("\n✨ 知识库构建完成！")

    def load_knowledge_base(self):
        """加载已构建的知识库"""
        self.retriever.load()
        print("✅ 知识库已加载")

    def query(self, question: str, coarse_top_k: int = 50, final_top_k: int = 3) -> Dict:
        """
        查询知识库（带 Rerank）

        参数:
            question: 用户问题
            coarse_top_k: 粗排召回数量（默认50）
            final_top_k: 最终返回数量（默认3）
        """
        print(f"\n🔍 查询问题: {question}")

        # 1️⃣ 粗排：快速召回
        print(f"\n📊 第一步：粗排（召回 top-{coarse_top_k}）")
        coarse_results = self.retriever.search(question, top_k=coarse_top_k)
        print(f"✅ 粗排完成，召回 {len(coarse_results)} 个候选文档")

        # 显示粗排 top-3
        print("\n粗排 Top-3:")
        for i, result in enumerate(coarse_results[:3]):
            print(f"  [{i+1}] {result['metadata']} (分数: {result['coarse_score']:.4f})")

        # 2️⃣ 精排：Rerank
        print(f"\n🎯 第二步：精排（Rerank）")
        reranked_results = self.reranker.rerank(question, coarse_results)
        print(f"✅ 精排完成")

        # 显示精排后 top-3
        print("\n精排 Top-3:")
        for i, result in enumerate(reranked_results[:3]):
            print(f"  [{i+1}] {result['metadata']} (rerank分数: {result.get('rerank_score', 0):.4f})")

        # 3️⃣ 取最终 top-k
        final_results = reranked_results[:final_top_k]

        # 4️⃣ 组装上下文
        print(f"\n📚 最终选中的 {len(final_results)} 个文档:")
        for i, result in enumerate(final_results):
            print(f"\n  [{i+1}] {result['metadata']}")
            print(f"      粗排分数: {result['coarse_score']:.4f}")
            print(f"      精排分数: {result.get('rerank_score', 0):.4f}")
            if 'keyword_matches' in result:
                print(f"      关键词匹配: {result['keyword_matches']}")
            print(f"      内容: {result['content'][:100]}...")

        context = "\n\n".join([
            f"【来源: {r['metadata']}】\n{r['content']}"
            for r in final_results
        ])

        # 5️⃣ 生成答案
        print("\n💭 正在生成答案...")
        answer = self.chat.generate_answer(question, context)

        return {
            "answer": answer,
            "sources": final_results,
            "coarse_results": coarse_results,  # 用于对比
            "query": question
        }


# ============================================================================
# 6. 主程序
# ============================================================================

def main():
    """主程序演示"""

    print("=" * 80)
    print("🤖 Ollama + TF-IDF + Rerank 本地知识库检索系统")
    print("   演示 RAG 高效召回方法：重排序（Reranking）")
    print("=" * 80)

    # 检查 Ollama 是否运行
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            print("\n❌ 错误：Ollama 未运行")
            print("\n请先启动 Ollama：")
            print("  - macOS: 打开 Ollama 应用")
            print("  - 或运行: ollama serve")
            return
    except Exception as e:
        print(f"\n❌ 错误：无法连接到 Ollama")
        print(f"💡 请确保 Ollama 正在运行: {OLLAMA_BASE_URL}")
        return

    # 选择 Rerank 方法
    print("\n请选择 Rerank 方法:")
    print("1. 关键词匹配（快速，推荐）")
    print("2. 模型打分（准确但慢）")
    print("\n输入 1 或 2（默认 1）: ", end="")

    try:
        choice = input().strip()
        if choice == "2":
            rerank_method = "model"
            print("\n✅ 使用模型打分方法")
        else:
            rerank_method = "keyword"
            print("\n✅ 使用关键词匹配方法")
    except:
        rerank_method = "keyword"
        print("\n✅ 使用关键词匹配方法（默认）")

    # 创建 RAG 系统
    rag = RAGSystemWithRerank(rerank_method=rerank_method)

    # 创建示例文档目录
    docs_dir = "knowledge_base"
    Path(docs_dir).mkdir(exist_ok=True)

    # 检查是否已有文档
    if not list(Path(docs_dir).rglob('*')):
        print(f"\n📝 在 {docs_dir}/ 目录中添加你的文档（txt, pdf, docx）")
        print("然后重新运行程序")
        return

    # 构建或加载知识库
    if Path("tfidf_matrix.pkl").exists():
        print("\n检测到已有索引，是否重新构建？(y/n): ", end="")
        try:
            choice = input().strip().lower()
            if choice == 'y':
                rag.build_knowledge_base(docs_dir)
            else:
                rag.load_knowledge_base()
        except:
            rag.load_knowledge_base()
    else:
        rag.build_knowledge_base(docs_dir)

    # 交互式问答
    print("\n" + "=" * 80)
    print("💬 开始问答（输入 'quit' 退出）")
    print("=" * 80)

    while True:
        print("\n" + "─" * 80)
        try:
            question = input("❓ 你的问题: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break

            # 查询（带 rerank）
            result = rag.query(question, coarse_top_k=50, final_top_k=3)

            # 显示答案
            if "error" in result:
                print(f"\n❌ {result['error']}")
                continue

            print("\n" + "─" * 80)
            print("📖 答案:")
            print("─" * 80)
            print(result['answer'])
            print("─" * 80)

            # 显示对比
            print("\n📊 粗排 vs 精排对比（Top-3）:")
            coarse_top3 = result['coarse_results'][:3]
            final_top3 = result['sources']

            print("\n粗排 Top-3:")
            for i, doc in enumerate(coarse_top3):
                print(f"  {i+1}. {doc['metadata']} (分数: {doc['coarse_score']:.4f})")

            print("\n精排 Top-3:")
            for i, doc in enumerate(final_top3):
                print(f"  {i+1}. {doc['metadata']} (rerank: {doc.get('rerank_score', 0):.4f})")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
