"""
Ollama + Faiss + Rerank 三国知识库检索系统
========================================

使用本地 Ollama 模型 + Faiss + Rerank 构建的三国知识库问答系统。
演示 RAG 高效召回方法：重排序（Reranking）在大型数据集上的效果

特点：
- 专门用于三国演义（1.7MB，大量人物和事件）
- 优化的参数设置
- 清晰展示 Rerank 的效果

作者: Claude Code Assistant
日期: 2026-01-27
"""

import os
import sys
import json
import pickle
from typing import List, Dict
from pathlib import Path
import re
import requests

import numpy as np
import faiss

# ============================================================================
# 配置
# ============================================================================

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "deepseek-r1:1.5b"

# 三国知识库专用配置
DOCS_DIR = "knowledge_threekingdoms"
INDEX_PREFIX = "threekingdoms"  # 索引文件前缀
CHUNK_SIZE = 800          # 大文档用更大的块
CHUNK_OVERLAP = 100       # 更大的重叠
COARSE_TOP_K = 100        # 粗排召回100个（数据量大）
FINAL_TOP_K = 5           # 最终返回5个

# ============================================================================
# 导入原来的类（这里简化重写，避免代码太长）
# ============================================================================

class DocumentLoader:
    """文档加载器"""
    @staticmethod
    def load_txt(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def load_directory(directory: str) -> List[tuple]:
        documents = []
        path = Path(directory)
        for file_path in path.rglob('*.txt'):
            try:
                content = DocumentLoader.load_txt(str(file_path))
                if content.strip():
                    documents.append((file_path.name, content))
                    print(f"✅ 已加载: {file_path.name} ({len(content):,} 字符)")
            except Exception as e:
                print(f"❌ 加载失败 {file_path.name}: {e}")
        return documents

class TextSplitter:
    """文本分割器"""
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, metadata: str = "") -> List[Dict]:
        chunks = []
        text = re.sub(r'\n+', '\n', text).strip()
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

    def split_documents(self, documents: List[tuple]) -> List[Dict]:
        all_chunks = []
        for filename, content in documents:
            chunks = self.split_text(content, metadata=filename)
            all_chunks.extend(chunks)
        return all_chunks

class OllamaEmbedding:
    """Ollama Embedding 生成器"""
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = EMBEDDING_MODEL
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception("Ollama 未运行")
            print(f"✅ Ollama 连接成功")
        except Exception as e:
            raise Exception(f"无法连接到 Ollama: {e}")

    def get_embedding(self, text: str) -> List[float]:
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("embedding", [])
            return []
        except:
            return []

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """批量获取 Embedding"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                emb = self.get_embedding(text)
                if emb:
                    embeddings.append(emb)
                    print(f"✅ [{len(embeddings)}/{len(texts)}] 生成 embedding ({len(text):,} 字符)")
                else:
                    embeddings.append([0.0] * 768)
        return embeddings

class FaissIndex:
    """Faiss 向量索引"""
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        self.chunks = []
        self.embeddings = None

    def build_index(self, chunks: List[Dict], embeddings: List[List[float]]):
        self.chunks = chunks
        self.embeddings = np.array(embeddings, dtype='float32')

        if self.embeddings.shape[1] != self.dimension:
            self.dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings)

        print(f"✅ Faiss 索引构建完成: {len(chunks)} 个文档块")

    def save(self):
        """保存索引"""
        faiss.write_index(self.index, f"{INDEX_PREFIX}_faiss.bin")
        with open(f"{INDEX_PREFIX}_chunks.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings
            }, f)
        print(f"✅ 索引已保存: {INDEX_PREFIX}_*.bin")

    def load(self):
        """加载索引"""
        self.index = faiss.read_index(f"{INDEX_PREFIX}_faiss.bin")
        with open(f"{INDEX_PREFIX}_chunks.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.embeddings = data['embeddings']
        self.dimension = self.index.d
        print(f"✅ 索引已加载: {len(self.chunks)} 个文档块")

    def search(self, query_embedding: List[float], top_k: int = COARSE_TOP_K) -> List[Dict]:
        """搜索文档"""
        query_array = np.array([query_embedding], dtype='float32')
        distances, indices = self.index.search(query_array, top_k)

        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['coarse_score'] = float(dist)
                chunk['embedding'] = self.embeddings[idx]
                results.append(chunk)

        return results

class Reranker:
    """Rerank 重排序器"""
    def __init__(self, method: str = "vector"):
        self.method = method

    def rerank(self, query: str, query_embedding: List[float], documents: List[Dict]) -> List[Dict]:
        """重排序"""
        if self.method == "vector":
            print("🔄 使用向量相似度进行 rerank...")
            query_vec = np.array(query_embedding).reshape(1, -1)

            for doc in documents:
                doc_vec = doc['embedding'].reshape(1, -1)
                cosine_sim = np.dot(query_vec, doc_vec.T) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
                )

                l2_dist = doc['coarse_score']
                l2_sim = 1 / (1 + l2_dist)
                doc['rerank_score'] = 0.7 * cosine_sim[0][0] + 0.3 * l2_sim
                doc['cosine_sim'] = cosine_sim[0][0]

            return sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

        elif self.method == "keyword":
            print("🔄 使用关键词匹配进行 rerank...")
            query_keywords = set(query.lower().split())

            for doc in documents:
                content = doc['content'].lower()
                keyword_matches = sum(content.count(kw) for kw in query_keywords)
                l2_sim = 1 / (1 + doc['coarse_score'])
                doc['rerank_score'] = l2_sim * (1 + keyword_matches * 0.1)
                doc['keyword_matches'] = keyword_matches

            return sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

class OllamaChat:
    """Ollama 问答生成器"""
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = CHAT_MODEL

    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""你是一个专业的三国知识问答助手。请基于以下参考文档回答用户的问题。

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
                    "options": {"temperature": 0.3, "num_predict": 1000}
                },
                timeout=120
            )

            if response.status_code == 200:
                return response.json().get("response", "⚠️  无法生成答案")
            return f"⚠️  生成答案时出错: {response.status_code}"
        except Exception as e:
            return f"⚠️  生成答案时出错: {e}"

class RAGSystemWithRerank:
    """带 Rerank 的 RAG 系统"""
    def __init__(self, rerank_method: str = "vector"):
        self.embedder = OllamaEmbedding()
        self.chat = OllamaChat()
        self.index = FaissIndex()
        self.reranker = Reranker(method=rerank_method)

    def build_knowledge_base(self):
        """构建知识库"""
        print("\n" + "=" * 80)
        print("📚 三国知识库构建")
        print("=" * 80)

        # 加载文档
        print("\n第一步：加载文档")
        loader = DocumentLoader()
        documents = loader.load_directory(DOCS_DIR)

        if not documents:
            print(f"❌ 未找到文档，请在 {DOCS_DIR}/ 目录中放入三国文档")
            return

        print(f"\n✅ 共加载 {len(documents)} 个文档")

        # 分割文本
        print("\n第二步：分割文本")
        splitter = TextSplitter()
        chunks = splitter.split_documents(documents)
        print(f"✅ 共分割成 {len(chunks)} 个文本块")

        # 生成 embedding
        print("\n第三步：生成 Embedding")
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedder.get_embeddings_batch(texts)

        # 构建索引
        print("\n第四步：构建 Faiss 索引")
        self.index.build_index(chunks, embeddings)

        # 保存
        print("\n第五步：保存索引")
        self.index.save()
        print("\n✨ 三国知识库构建完成！")

    def load_knowledge_base(self):
        """加载知识库"""
        self.index.load()
        print("✅ 三国知识库已加载")

    def query(self, question: str) -> Dict:
        """查询"""
        print(f"\n{'=' * 80}")
        print(f"🔍 查询: {question}")
        print(f"{'=' * 80}")

        # 生成查询向量
        query_embedding = self.embedder.get_embedding(question)
        if not query_embedding:
            return {"error": "无法生成查询向量"}

        # 粗排
        print(f"\n📊 第一步：粗排（向量检索，召回 top-{COARSE_TOP_K}）")
        coarse_results = self.index.search(query_embedding, top_k=COARSE_TOP_K)
        print(f"✅ 粗排完成")

        # 精排
        print(f"\n🎯 第二步：精排（Rerank）")
        reranked_results = self.reranker.rerank(question, query_embedding, coarse_results)
        print(f"✅ 精排完成")

        # 显示对比
        print(f"\n📊 粗排 vs 精排对比（Top-5）:")
        print("\n" + "─" * 80)
        for i in range(5):
            coarse = coarse_results[i]
            rerank = reranked_results[i]
            print(f"\n  [{i+1}] 粗排: {coarse['metadata'][:30]:30s} (L2: {coarse['coarse_score']:.4f})")
            print(f"      精排: {rerank['metadata'][:30]:30s} (rerank: {rerank.get('rerank_score', 0):.4f})")

            # 如果排名变化，标出来
            if coarse['metadata'] != rerank['metadata']:
                print(f"      ⚠️  排名变化！")

        # 最终结果
        final_results = reranked_results[:FINAL_TOP_K]

        # 组装上下文
        context = "\n\n".join([
            f"【来源: {r['metadata'][:50]}】\n{r['content'][:300]}..."
            for r in final_results
        ])

        # 生成答案
        print(f"\n💭 正在生成答案...")
        answer = self.chat.generate_answer(question, context)

        return {
            "answer": answer,
            "sources": final_results,
            "coarse_results": coarse_results[:FINAL_TOP_K],
            "query": question
        }

# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 80)
    print("🤖 三国知识库检索系统（Ollama + Faiss + Rerank）")
    print("=" * 80)
    print("\n📚 基于《三国演义》1.7MB 文本")
    print("🎯 演示 Rerank 在大数据集上的效果")

    # 检查 Ollama
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            print("\n❌ Ollama 未运行")
            return
    except:
        print("\n❌ 无法连接 Ollama")
        return

    # 选择 Rerank 方法
    print("\n请选择 Rerank 方法:")
    print("1. 向量相似度（最快，推荐）")
    print("2. 关键词匹配")
    print("\n输入 1 或 2（默认 1）: ", end="")

    try:
        choice = input().strip()
        rerank_method = "vector" if choice != "2" else "keyword"
        print(f"\n✅ 使用 {'向量相似度' if rerank_method == 'vector' else '关键词匹配'} 方法")
    except:
        rerank_method = "vector"
        print("\n✅ 使用向量相似度方法（默认）")

    # 创建系统
    try:
        rag = RAGSystemWithRerank(rerank_method=rerank_method)
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        print("\n💡 请确保已下载 embedding 模型: ollama pull nomic-embed-text")
        return

    # 构建或加载知识库
    if Path(f"{INDEX_PREFIX}_faiss.bin").exists():
        print(f"\n检测到已有索引，是否重新构建？(y/n): ", end="")
        try:
            choice = input().strip().lower()
            if choice == 'y':
                rag.build_knowledge_base()
            else:
                rag.load_knowledge_base()
        except:
            rag.load_knowledge_base()
    else:
        rag.build_knowledge_base()

    # 交互式问答
    print("\n" + "=" * 80)
    print("💬 开始问答（输入 'quit' 退出）")
    print("=" * 80)

    # 提示示例问题
    print("\n💡 示例问题:")
    print("  - 刘备的三个兄弟是谁？")
    print("  - 曹操是如何起家的？")
    print("  - 桃园三结义是在哪里？")
    print("  - 吕布的武器是什么？")
    print("  - 董卓是怎么死的？")
    print("  - 赤兔马是谁的？")

    while True:
        print("\n" + "─" * 80)
        try:
            question = input("❓ 你的问题: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break

            result = rag.query(question)

            if "error" in result:
                print(f"\n❌ {result['error']}")
                continue

            print("\n" + "─" * 80)
            print("📖 答案:")
            print("─" * 80)
            print(result['answer'])
            print("─" * 80)

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
