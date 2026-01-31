"""
三国知识库 + Rerank 重排序系统
================================

使用本地 Ollama 模型 + Faiss + Rerank 构建的三国知识库问答系统。
演示 RAG 高效召回方法：重排序（Reranking）

功能：
1. 使用 Ollama Embedding 生成向量
2. 使用 Faiss 进行粗排（快速召回 top-50）
3. 使用 Rerank 进行精排（提高准确度）
4. 对比粗排和精排的效果差异
5. 专门针对三国知识库优化

特点：
- 清晰展示粗排 vs 精排的对比
- 多种重排方法（向量相似度、关键词匹配）
- 可视化展示排名变化

作者: Claude Code Assistant
日期: 2026-01-31
"""

import os
import json
import pickle
from typing import List, Dict, Tuple
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
CHAT_MODEL = "deepseek-r1:7b"

DOCS_DIR = "knowledge_threekingdoms"
INDEX_PREFIX = "threekingdoms_rerank"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
COARSE_TOP_K = 50
FINAL_TOP_K = 5

# ============================================================================
# 文档加载和预处理
# ============================================================================

class DocumentLoader:
    """文档加载器"""

    @staticmethod
    def load_txt(file_path: str) -> str:
        """加载 TXT 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def load_directory(directory: str) -> List[Tuple[str, str]]:
        """加载目录下的所有文档"""
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
        """分割文本"""
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

    def split_documents(self, documents: List[Tuple[str, str]]) -> List[Dict]:
        """分割多个文档"""
        all_chunks = []
        for filename, content in documents:
            chunks = self.split_text(content, metadata=filename)
            all_chunks.extend(chunks)
        return all_chunks


# ============================================================================
# Ollama Embedding
# ============================================================================

class OllamaEmbedding:
    """Ollama Embedding 生成器"""

    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = EMBEDDING_MODEL

        # 检查 Ollama 是否运行
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception("Ollama 未运行")
            print(f"✅ Ollama 连接成功")
        except Exception as e:
            raise Exception(f"无法连接到 Ollama: {e}")

    def get_embedding(self, text: str) -> List[float]:
        """获取文本的 Embedding"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("embedding", [])
            else:
                print(f"⚠️  Embedding 生成失败: {response.status_code}")
                return []

        except Exception as e:
            print(f"⚠️  Embedding 生成出错: {e}")
            return []

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """批量获取 Embedding"""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                emb = self.get_embedding(text)
                if emb:
                    embeddings.append(emb)
                    print(f"✅ 生成 embedding {len(embeddings)}/{len(texts)}")
                else:
                    print(f"⚠️  跳过无法生成 embedding 的文本")
                    # 用零向量填充
                    embeddings.append([0.0] * 768)

        return embeddings


# ============================================================================
# Faiss 索引（粗排）
# ============================================================================

class FaissIndex:
    """Faiss 向量索引管理器 - 用于粗排"""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        self.chunks = []
        self.embeddings = None  # 保存 embeddings 用于 rerank

    def build_index(self, chunks: List[Dict], embeddings: List[List[float]]):
        """构建 Faiss 索引"""
        self.chunks = chunks
        self.embeddings = np.array(embeddings, dtype='float32')
        embeddings_array = self.embeddings

        if embeddings_array.shape[1] != self.dimension:
            print(f"⚠️  向量维度不匹配: 期望 {self.dimension}, 实际 {embeddings_array.shape[1]}")
            self.dimension = embeddings_array.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings_array)

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
        """
        粗排：快速召回 top-k 个候选文档
        """
        if self.index is None:
            raise ValueError("索引未构建，请先构建或加载索引")

        query_array = np.array([query_embedding], dtype='float32')
        distances, indices = self.index.search(query_array, top_k)

        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['coarse_score'] = float(dist)  # 粗排分数（L2距离）
                chunk['coarse_rank'] = i + 1  # 粗排排名
                chunk['embedding'] = self.embeddings[idx]  # 保存 embedding 用于 rerank
                results.append(chunk)

        return results


# ============================================================================
# Rerank 重排序器（精排）
# ============================================================================

class Reranker:
    """
    重排序器 - 对粗排结果进行精排

    方法1: 向量相似度增强（快速，推荐）
    方法2: 关键词匹配增强（适合专有名词多的场景）
    """

    def __init__(self, method: str = "vector"):
        """
        参数:
            method: rerank 方法
                - "vector": 基于向量相似度（推荐，最快）
                - "keyword": 基于关键词匹配（适合三国人名、地名）
        """
        self.method = method

    def _vector_rerank(self, query_embedding: List[float], documents: List[Dict]) -> List[Dict]:
        """
        基于向量相似度的 rerank

        原理：
        1. 使用查询向量和文档向量计算余弦相似度
        2. 结合粗排分数（L2距离）重新排序
        3. 余弦相似度比 L2 距离更适合 rerank
        """
        query_vec = np.array(query_embedding).reshape(1, -1)

        for doc in documents:
            doc_vec = doc['embedding'].reshape(1, -1)

            # 计算余弦相似度
            cosine_sim = np.dot(query_vec, doc_vec.T) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )

            # 转换 L2 距离为相似度（越小越好 → 越大越好）
            l2_dist = doc['coarse_score']
            l2_sim = 1 / (1 + l2_dist)  # 转换

            # 结合两种相似度（70%余弦 + 30%L2转换）
            doc['rerank_score'] = 0.7 * cosine_sim[0][0] + 0.3 * l2_sim
            doc['cosine_sim'] = cosine_sim[0][0]

        # 按 rerank 分数排序
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

        # 更新精排排名
        for i, doc in enumerate(reranked):
            doc['fine_rank'] = i + 1

        return reranked

    def _keyword_rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        基于关键词匹配的 rerank

        适合三国场景：人名、地名、武器名等专有名词
        """
        query_keywords = set(query.lower().split())

        for doc in documents:
            content = doc['content'].lower()

            # 计算关键词匹配分数
            keyword_matches = 0
            for keyword in query_keywords:
                if keyword in content:
                    keyword_matches += content.count(keyword)

            # 结合粗排分数和关键词匹配
            l2_dist = doc['coarse_score']
            l2_sim = 1 / (1 + l2_dist)
            doc['rerank_score'] = l2_sim * (1 + keyword_matches * 0.1)
            doc['keyword_matches'] = keyword_matches

        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

        # 更新精排排名
        for i, doc in enumerate(reranked):
            doc['fine_rank'] = i + 1

        return reranked

    def rerank(self, query: str, query_embedding: List[float], documents: List[Dict]) -> List[Dict]:
        """
        对文档进行重排序

        参数:
            query: 用户查询
            query_embedding: 查询的向量
            documents: 粗排结果列表

        返回:
            重排序后的文档列表
        """
        if self.method == "vector":
            print("🔄 使用向量相似度进行 rerank...")
            return self._vector_rerank(query_embedding, documents)
        elif self.method == "keyword":
            print("🔄 使用关键词匹配进行 rerank...")
            return self._keyword_rerank(query, documents)
        else:
            raise ValueError(f"未知的 rerank 方法: {self.method}")


# ============================================================================
# Ollama 问答生成
# ============================================================================

class OllamaChat:
    """Ollama 问答生成器"""

    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = CHAT_MODEL

    def generate_answer(self, question: str, context: str) -> str:
        """基于上下文生成答案"""
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
# 完整的 RAG 系统（带 Rerank）
# ============================================================================

class RAGSystemWithRerank:
    """带 Rerank 的三国知识库 RAG 系统"""

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
            print(f"❌ 未找到任何文档")
            return

        print(f"\n✅ 共加载 {len(documents)} 个文档")

        # 分割文本
        print("\n第二步：分割文本")
        splitter = TextSplitter()
        chunks = splitter.split_documents(documents)

        print(f"✅ 共分割成 {len(chunks)} 个文本块")

        # 生成 embedding
        print("\n第三步：生成 Embedding（使用 Ollama）")
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedder.get_embeddings_batch(texts, batch_size=10)

        # 构建索引
        print("\n第四步：构建 Faiss 索引")
        self.index.build_index(chunks, embeddings)

        # 保存
        print("\n第五步：保存索引")
        self.index.save()
        print("\n✨ 三国知识库构建完成！")

    def load_knowledge_base(self):
        """加载已构建的知识库"""
        self.index.load()
        print("✅ 三国知识库已加载")

    def query(self, question: str) -> Dict:
        """
        查询知识库（带 Rerank）

        参数:
            question: 用户问题
        """
        print(f"\n{'=' * 80}")
        print(f"🔍 查询问题: {question}")
        print(f"{'=' * 80}")

        # 1️⃣ 生成查询向量
        query_embedding = self.embedder.get_embedding(question)
        if not query_embedding:
            return {"error": "无法生成查询向量"}

        # 2️⃣ 粗排：向量检索召回
        print(f"\n📊 第一步：粗排（向量检索，召回 top-{COARSE_TOP_K}）")
        coarse_results = self.index.search(query_embedding, top_k=COARSE_TOP_K)
        print(f"✅ 粗排完成，召回 {len(coarse_results)} 个候选文档")

        # 显示粗排 top-5
        print("\n粗排 Top-5:")
        for i, result in enumerate(coarse_results[:5]):
            print(f"  [{i+1}] {result['metadata'][:50]:50s} (L2距离: {result['coarse_score']:.4f})")

        # 3️⃣ 精排：Rerank
        print(f"\n🎯 第二步：精排（Rerank）")
        reranked_results = self.reranker.rerank(question, query_embedding, coarse_results)
        print(f"✅ 精排完成")

        # 显示精排后 top-5
        print("\n精排 Top-5:")
        for i, result in enumerate(reranked_results[:5]):
            print(f"  [{i+1}] {result['metadata'][:50]:50s} (rerank分数: {result.get('rerank_score', 0):.4f})")

        # 4️⃣ 显示对比（粗排 vs 精排）
        print(f"\n📊 粗排 vs 精排对比（Top-5）:")
        print("\n" + "─" * 80)
        for i in range(min(5, len(reranked_results))):
            coarse_rank = reranked_results[i].get('coarse_rank', '-')
            fine_rank = i + 1
            metadata = reranked_results[i]['metadata'][:45]
            coarse_score = reranked_results[i]['coarse_score']
            rerank_score = reranked_results[i].get('rerank_score', 0)

            print(f"\n  [{fine_rank}] {metadata}")
            print(f"      粗排排名: #{coarse_rank:2d}  (L2: {coarse_score:.4f})")
            print(f"      精排排名: #{fine_rank:2d}  (rerank: {rerank_score:.4f})")

            if coarse_rank != fine_rank:
                print(f"      ⚠️  排名变化: #{coarse_rank} → #{fine_rank}")

        # 5️⃣ 取最终 top-k
        final_results = reranked_results[:FINAL_TOP_K]

        # 6️⃣ 组装上下文
        context = "\n\n".join([
            f"【来源: {r['metadata'][:50]}】\n{r['content'][:400]}"
            for r in final_results
        ])

        # 7️⃣ 生成答案
        print("\n💭 正在生成答案...")
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
    """主程序演示"""

    print("=" * 80)
    print("🤖 三国知识库 + Rerank 重排序系统")
    print("=" * 80)
    print("\n📚 基于《三国演义》1.7MB 文本")
    print("🎯 演示 RAG 高效召回方法：重排序（Reranking）")

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
    print("1. 向量相似度（推荐，语义匹配强）⭐")
    print("2. 关键词匹配（适合人名、地名等专有名词）")
    print("\n输入 1-2（默认 1）: ", end="")

    try:
        choice = input().strip()
        if choice == "2":
            rerank_method = "keyword"
            print("\n✅ 使用关键词匹配方法")
        else:
            rerank_method = "vector"
            print("\n✅ 使用向量相似度方法（默认）")
    except:
        rerank_method = "vector"
        print("\n✅ 使用向量相似度方法（默认）")

    # 创建 RAG 系统
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
    print("\n💡 示例问题（观察粗排vs精排的效果）:")
    print("  - 诸葛亮的武器是什么？")
    print("  - 赤壁之战谁赢了？")
    print("  - 关羽怎么死的？")
    print("  - 曹操有多少个儿子？")
    print("  - 刘备的坐骑是什么？")

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
            result = rag.query(question)

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
            print("\n📊 粗排 vs 精排最终对比（Top-3）:")
            coarse_top3 = result['coarse_results']
            final_top3 = result['sources']

            print("\n粗排 Top-3 (L2距离，越小越好):")
            for i, doc in enumerate(coarse_top3):
                rank = doc.get('coarse_rank', i+1)
                print(f"  {i+1}. #{rank:2d} {doc['metadata'][:40]:40s} ({doc['coarse_score']:.4f})")

            print("\n精排 Top-3 (rerank分数，越大越好):")
            for i, doc in enumerate(final_top3):
                rank = doc.get('fine_rank', i+1)
                score = doc.get('rerank_score', 0)
                print(f"  {i+1}. #{rank:2d} {doc['metadata'][:40]:40s} ({score:.4f})")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
