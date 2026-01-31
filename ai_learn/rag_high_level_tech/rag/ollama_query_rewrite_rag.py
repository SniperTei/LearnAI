"""
Ollama + Faiss + Query Rewrite 本地知识库检索系统
===================================================

使用本地 Ollama 模型 + Faiss + 查询重写构建的本地知识库问答系统。
演示 RAG 高效召回方法：查询重写（Query Rewriting）

查询重写方法：
1. LLM 查询重写 - 让 LLM 理解意图并改写查询
2. HyDE - 生成假设答案，用答案去检索
3. Step-back - 将具体问题抽象成更高层次的问题

特点：
- 对比不同查询重写方法的效果
- 可视化展示重写前后的查询
- 专门针对模糊、复杂问题优化

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

DOCS_DIR = "knowledge_threekingdoms"  # 改用三国知识库
INDEX_PREFIX = "threekingdoms_query_rewrite"  # 索引前缀也改一下避免冲突
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
COARSE_TOP_K = 50
FINAL_TOP_K = 3

# ============================================================================
# 查询重写器（优化版）
# ============================================================================

class QueryRewriter:
    """查询重写器 - 优化查询质量（带保护机制）"""

    def __init__(self, method: str = "llm_rewrite", enable_fallback: bool = True):
        """
        Args:
            method: 重写方法
                - "llm_rewrite": LLM 查询重写（推荐，保守策略）
                - "hyde": HyDE (Hypothetical Document Embeddings)
                - "step_back": Step-back 抽象化
            enable_fallback: 是否启用回退机制（重写失败时使用原查询）
        """
        self.method = method
        self.base_url = OLLAMA_BASE_URL
        self.model = CHAT_MODEL
        self.enable_fallback = enable_fallback

    def _llm_rewrite(self, query: str) -> str:
        """
        LLM 查询重写（超保守策略）

        原理：
        1. 完整保留原查询
        2. 只添加1-3个补充关键词
        3. 不过度重写
        """
        prompt = f"""请改进以下搜索查询，输出一个优化的查询字符串。

原查询：{query}

要求：
1. **必须包含原查询的完整内容**（可以在前面或后面添加关键词）
2. 只添加 1-3 个相关的补充关键词（同义词、相关术语）
3. 用空格分隔关键词
4. 不要改变问题的核心意思
5. 保持简洁

示例：
- "诸葛亮" → "诸葛亮 孔明 卧龙"
- "赤壁之战" → "赤壁之战 周瑜 曹操 孙权 刘备"
- "怎么做红烧肉" → "怎么做红烧肉 烹饪方法 步骤"

只输出改进后的查询字符串，不要其他内容："""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 80}
                },
                timeout=60
            )

            if response.status_code == 200:
                rewritten = response.json().get("response", "").strip()
                # 清理可能的前缀
                for prefix in ['改进后的查询字符串', '查询', '重写', '：', ':', '优化后的']:
                    rewritten = rewritten.replace(prefix, '').strip()
                # 取第一行
                rewritten = rewritten.split('\n')[0].strip()

                # 验证：重写后的查询必须包含原查询的关键词
                original_words = set(query.lower().split())
                rewritten_words = set(rewritten.lower().split())

                # 如果原查询的词都不在重写中，说明重写失败
                if original_words and not original_words & rewritten_words:
                    if self.enable_fallback:
                        print(f"⚠️  重写偏离原意，保留原查询")
                    return query

                return rewritten if rewritten else query
        except Exception as e:
            print(f"⚠️  LLM 重写失败: {e}")

        return query

    def _hyde(self, query: str) -> str:
        """
        HyDE (Hypothetical Document Embeddings)

        原理：
        1. 让 LLM 生成一个假设的答案
        2. 用假设的答案去检索（而不是原始查询）
        3. 假设答案通常包含更丰富的语义信息

        适用场景：
        - 语义查询（"怎么..." "为什么..."）
        - 概念性查询
        """
        prompt = f"""请为以下问题生成一个简短的假设性答案（50-100字）。

问题：{query}

要求：
1. 基于常识生成一个合理的答案
2. 答案应该包含相关的关键概念和术语
3. 不需要准确，但要有代表性
4. 使用清晰的段落格式

假设性答案："""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.5, "num_predict": 200}
                },
                timeout=60
            )

            if response.status_code == 200:
                hypothetical_answer = response.json().get("response", "").strip()
                # 清理多余的前缀
                hypothetical_answer = re.sub(r'^(假设性答案|答案|：|:)\s*', '', hypothetical_answer)
                return hypothetical_answer if hypothetical_answer else query
        except Exception as e:
            print(f"⚠️  HyDE 生成失败: {e}")

        return query

    def _step_back(self, query: str) -> str:
        """
        Step-back Prompting

        原理：
        1. 将具体问题抽象成更高层次的概念性问题
        2. 先回答高层次问题，再回到具体问题
        3. 适合复杂、专业的问题

        示例：
        - "诸葛亮用什么武器？" → "三国时期的军事装备和武器"
        - "Python 如何处理异常？" → "编程语言中的异常处理机制"
        """
        prompt = f"""请将以下具体问题抽象成一个更高层次的概念性问题。

原问题：{query}

要求：
1. 提取问题背后的核心概念
2. 将具体问题抽象成通用原理
3. 保持问题简洁，10-20 字
4. 只输出抽象后的问题，不要其他内容

抽象后的问题："""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 100}
                },
                timeout=60
            )

            if response.status_code == 200:
                abstract_query = response.json().get("response", "").strip()
                # 清理多余的前缀
                abstract_query = re.sub(r'^(抽象后的问题|问题|：|:)\s*', '', abstract_query)
                # 取第一行
                abstract_query = abstract_query.split('\n')[0].strip()
                return abstract_query if abstract_query else query
        except Exception as e:
            print(f"⚠️  Step-back 失败: {e}")

        return query

    def _evaluate_rewrite_quality(self, original_query: str, rewritten_query: str) -> float:
        """
        评估重写质量（打分 0-10）

        评估标准：
        1. 是否保留了原查询的关键信息
        2. 是否添加了有用的补充信息
        3. 是否改变了原意
        4. 查询是否简洁清晰
        """
        prompt = f"""请评估以下查询重写的质量（0-10分）。

原查询：{original_query}
重写后：{rewritten_query}

评分标准：
1. 保留了原查询的所有关键词（3分）
2. 添加了有用的补充信息（3分）
3. 没有改变原意（2分）
4. 简洁清晰（2分）

要求：
- 只输出一个 0-10 的数字（可以是小数）
- 如果重写后查询丢失了原查询的关键信息，给低分（<5分）
- 如果重写不合理或偏离原意，给低分（<5分）

评分："""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 10}
                },
                timeout=30
            )

            if response.status_code == 200:
                score_text = response.json().get("response", "").strip()
                # 提取数字
                score_match = re.search(r'(\d+\.?\d*)', score_text)
                if score_match:
                    score = float(score_match.group(1))
                    return min(score, 10.0)  # 确保不超过10
        except Exception as e:
            if self.enable_fallback:
                print(f"⚠️  评分失败: {e}")

        return 6.0  # 默认中等分数

    def _is_simple_query(self, query: str) -> bool:
        """
        判断是否为简单查询（不需要重写）

        简单查询特征：
        1. 长度 <= 15字
        2. 包含明确的关键词（名词、专有名词）
        3. 没有模糊表述（"那个"、"怎么"、"什么"等）
        """
        # 长度检查
        if len(query) <= 15:
            # 检查是否包含模糊词
            vague_words = ["那个", "怎么", "如何", "什么", "哪个", "哪些", "还是", "或者"]
            if not any(vw in query for vw in vague_words):
                return True

        return False

    def rewrite(self, query: str, verbose: bool = True, compare_mode: bool = False) -> Dict:
        """
        重写查询（带质量检查和回退机制）

        Args:
            query: 原始查询
            verbose: 是否显示重写信息
            compare_mode: 是否开启对比模式（同时用原查询和重写查询检索）

        Returns:
            包含重写结果和评分的字典
        """
        result = {
            "original_query": query,
            "rewritten_query": query,
            "score": 0.0,
            "use_rewrite": False,
            "reason": ""
        }

        if verbose:
            print(f"\n📝 原始查询: {query}")
            print(f"🔧 重写方法: {self.method}")

        # 智能跳过：简单查询不需要重写
        if self._is_simple_query(query):
            if verbose:
                print("💡 检测到简单查询，跳过重写（直接使用原查询）")
            result["reason"] = "简单查询，无需重写"
            return result

        # 执行重写
        if self.method == "llm_rewrite":
            rewritten = self._llm_rewrite(query)
            if verbose and rewritten != query:
                print(f"✨ 初步重写: {rewritten}")
            result["rewritten_query"] = rewritten

        elif self.method == "hyde":
            hypothetical = self._hyde(query)
            if verbose and hypothetical != query:
                print(f"💭 假设答案: {hypothetical[:100]}...")
            result["rewritten_query"] = hypothetical

        elif self.method == "step_back":
            abstract = self._step_back(query)
            if verbose and abstract != query:
                print(f"🔍 抽象问题: {abstract}")
            result["rewritten_query"] = abstract

        else:
            print(f"⚠️  未知的重写方法: {self.method}")
            result["reason"] = "未知的重写方法"
            return result

        # 如果查询没有被重写
        if result["rewritten_query"] == query:
            if verbose:
                print("ℹ️  查询无需重写")
            result["reason"] = "查询无需重写"
            return result

        # 质量评分
        if self.enable_fallback:
            score = self._evaluate_rewrite_quality(query, result["rewritten_query"])
            result["score"] = score

            if verbose:
                print(f"📊 重写质量评分: {score}/10")

            # 回退机制：如果评分低于阈值，使用原查询
            # 阈值从 5.0 降低到 3.0，更容易接受重写结果
            if score < 3.0:
                if verbose:
                    print(f"⚠️  重写质量太低（< 3分），使用原查询")
                result["rewritten_query"] = query
                result["use_rewrite"] = False
                result["reason"] = f"重写质量太低（{score:.1f}分），回退到原查询"
            else:
                if verbose:
                    if score >= 7.0:
                        print(f"✅ 重写质量优秀，使用优化后的查询")
                    else:
                        print(f"✅ 重写质量可接受（{score:.1f}分），尝试使用")
                result["use_rewrite"] = True
                result["reason"] = f"重写质量评分 {score:.1f}分"
        else:
            # 不启用回退机制，直接使用重写结果
            result["use_rewrite"] = True
            result["reason"] = "已禁用回退机制"

        return result

# ============================================================================
# 复用原有的类（简化）
# ============================================================================

class DocumentLoader:
    """文档加载器"""
    @staticmethod
    def load_txt(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def load_directory(directory: str) -> List[Tuple[str, str]]:
        documents = []
        path = Path(directory)
        for file_path in path.rglob('*.txt'):
            try:
                content = DocumentLoader.load_txt(str(file_path))
                if content.strip():
                    documents.append((file_path.name, content))
                    print(f"✅ 已加载: {file_path.name}")
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

    def split_documents(self, documents: List[Tuple[str, str]]) -> List[Dict]:
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

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """批量获取 Embedding"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                emb = self.get_embedding(text)
                if emb:
                    embeddings.append(emb)
                    print(f"✅ [{len(embeddings)}/{len(texts)}] 生成 embedding")
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
                chunk['score'] = float(dist)
                results.append(chunk)

        return results

class OllamaChat:
    """Ollama 问答生成器"""
    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = CHAT_MODEL

    def generate_answer(self, question: str, context: str) -> str:
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
                    "options": {"temperature": 0.3, "num_predict": 1000}
                },
                timeout=120
            )

            if response.status_code == 200:
                return response.json().get("response", "⚠️  无法生成答案")
            return f"⚠️  生成答案时出错: {response.status_code}"
        except Exception as e:
            return f"⚠️  生成答案时出错: {e}"

# ============================================================================
# 带 Query Rewrite 的 RAG 系统
# ============================================================================

class RAGSystemWithQueryRewrite:
    """带查询重写的 RAG 系统"""

    def __init__(self, rewrite_method: str = "llm_rewrite", enable_fallback: bool = True,
                 compare_mode: bool = False):
        """
        Args:
            rewrite_method: 重写方法
                - "llm_rewrite": LLM 查询重写（推荐，保守策略）
                - "hyde": HyDE
                - "step_back": Step-back
            enable_fallback: 是否启用回退机制（默认True）
            compare_mode: 对比模式（同时用原查询和重写查询检索）
        """
        self.embedder = OllamaEmbedding()
        self.chat = OllamaChat()
        self.index = FaissIndex()
        self.rewriter = QueryRewriter(method=rewrite_method, enable_fallback=enable_fallback)
        self.rewrite_method = rewrite_method
        self.compare_mode = compare_mode

    def build_knowledge_base(self):
        """构建知识库"""
        print("\n" + "=" * 80)
        print("📚 知识库构建")
        print("=" * 80)

        # 加载文档
        print("\n第一步：加载文档")
        loader = DocumentLoader()
        documents = loader.load_directory(DOCS_DIR)

        if not documents:
            print(f"❌ 未找到文档，请在 {DOCS_DIR}/ 目录中放入文档")
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
        print("\n✨ 知识库构建完成！")

    def load_knowledge_base(self):
        """加载知识库"""
        self.index.load()
        print("✅ 知识库已加载")

    def query(self, question: str) -> Dict:
        """查询（带对比模式）"""
        print(f"\n{'=' * 80}")
        print(f"🔍 查询: {question}")
        print(f"{'=' * 80}")

        # ========== 查询重写 ==========
        print(f"\n🔧 第一步：查询重写")
        rewrite_result = self.rewriter.rewrite(question, verbose=True)

        # 判断是否使用对比模式
        if self.compare_mode and rewrite_result["use_rewrite"]:
            print(f"\n🔄 对比模式：同时用原查询和重写查询检索")

            # 用原查询检索
            original_embedding = self.embedder.get_embedding(question)
            if not original_embedding:
                return {"error": "无法生成查询向量"}

            original_results = self.index.search(original_embedding, top_k=COARSE_TOP_K)

            # 用重写查询检索
            rewritten_embedding = self.embedder.get_embedding(rewrite_result["rewritten_query"])
            if not rewritten_embedding:
                return {"error": "无法生成查询向量"}

            rewritten_results = self.index.search(rewritten_embedding, top_k=COARSE_TOP_K)

            # 对比结果
            print(f"\n📊 第二步：对比检索结果")
            print(f"\n原查询 Top-3:")
            for i, r in enumerate(original_results[:3]):
                print(f"  [{i+1}] {r['metadata'][:40]:40s} (分数: {r['score']:.4f})")

            print(f"\n重写查询 Top-3:")
            for i, r in enumerate(rewritten_results[:3]):
                print(f"  [{i+1}] {r['metadata'][:40]:40s} (分数: {r['score']:.4f})")

            # 使用重写查询的结果
            final_results = rewritten_results[:FINAL_TOP_K]

        else:
            # 使用重写查询（或原查询，如果回退了）
            query_to_use = rewrite_result["rewritten_query"]
            print(f"\n📊 第二步：向量检索（召回 top-{COARSE_TOP_K}）")

            query_embedding = self.embedder.get_embedding(query_to_use)
            if not query_embedding:
                return {"error": "无法生成查询向量"}

            results = self.index.search(query_embedding, top_k=COARSE_TOP_K)
            print(f"✅ 检索完成")

            # 最终结果
            final_results = results[:FINAL_TOP_K]

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
            "query": question,
            "rewritten_query": rewrite_result["rewritten_query"],
            "rewrite_score": rewrite_result.get("score", 0.0),
            "use_rewrite": rewrite_result["use_rewrite"],
            "rewrite_reason": rewrite_result["reason"]
        }

# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 80)
    print("🤖 Ollama + Faiss + Query Rewrite 本地知识库检索系统")
    print("=" * 80)
    print("\n📚 演示查询重写（Query Rewriting）的效果")

    # 检查 Ollama
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            print("\n❌ Ollama 未运行")
            return
    except:
        print("\n❌ 无法连接 Ollama")
        return

    # 选择重写方法
    print("\n选择查询重写方法:")
    print("1. LLM 查询重写（推荐，保守策略）⭐")
    print("2. HyDE - 生成假设答案（适合语义查询）")
    print("3. Step-back - 抽象化问题（适合复杂问题）")
    print("4. 不使用重写（对比基线）")
    print("\n输入 1-4（默认 1）: ", end="")

    try:
        choice = input().strip()
        if choice == "2":
            rewrite_method = "hyde"
            print("\n✅ 使用 HyDE 方法")
        elif choice == "3":
            rewrite_method = "step_back"
            print("\n✅ 使用 Step-back 方法")
        elif choice == "4":
            rewrite_method = None
            print("\n✅ 不使用查询重写")
        else:
            rewrite_method = "llm_rewrite"
            print("\n✅ 使用 LLM 查询重写（默认）")
    except:
        rewrite_method = "llm_rewrite"
        print("\n✅ 使用 LLM 查询重写（默认）")

    # 是否启用回退机制
    enable_fallback = True
    if rewrite_method:
        print("\n是否启用回退机制？（重写质量低时自动使用原查询）")
        print("1. 启用回退机制（推荐）⭐")
        print("2. 禁用回退机制")
        print("\n输入 1-2（默认 1）: ", end="")

        try:
            choice = input().strip()
            enable_fallback = (choice != "2")
            if enable_fallback:
                print("\n✅ 已启用回退机制")
            else:
                print("\n⚠️  已禁用回退机制（可能会出现重写失败的情况）")
        except:
            enable_fallback = True
            print("\n✅ 已启用回退机制（默认）")

    # 是否开启对比模式
    compare_mode = False
    if rewrite_method and enable_fallback:
        print("\n是否开启对比模式？（同时用原查询和重写查询检索）")
        print("1. 不开启对比模式（默认，更快）")
        print("2. 开启对比模式（可以看到对比效果）⭐")
        print("\n输入 1-2（默认 1）: ", end="")

        try:
            choice = input().strip()
            compare_mode = (choice == "2")
            if compare_mode:
                print("\n✅ 已开启对比模式")
            else:
                print("\n✅ 未开启对比模式")
        except:
            compare_mode = False
            print("\n✅ 未开启对比模式（默认）")

    # 创建系统
    try:
        if rewrite_method:
            rag = RAGSystemWithQueryRewrite(
                rewrite_method=rewrite_method,
                enable_fallback=enable_fallback,
                compare_mode=compare_mode
            )
        else:
            # 不使用重写，使用基础系统
            rag = RAGSystemWithQueryRewrite(
                rewrite_method="llm_rewrite",
                enable_fallback=False
            )
            # 禁用重写器
            rag.rewrite_method = None
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        print("\n💡 请确保已下载 embedding 模型: ollama pull nomic-embed-text")
        return

    # 创建文档目录
    Path(DOCS_DIR).mkdir(exist_ok=True)

    # 检查是否已有文档
    if not list(Path(DOCS_DIR).rglob('*.txt')):
        print(f"\n📝 在 {DOCS_DIR}/ 目录中添加你的文档（.txt 文件）")
        print("然后重新运行程序")
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

    # 提示示例问题（重点展示查询重写的效果）
    print("\n💡 示例问题（观察查询重写的效果）:")
    print("  - 那个骑着赤兔马的人是谁？  (LLM重写 → 吕布 赤兔马 三国)")
    print("  - 怎么做红烧肉？             (HyDE → 生成假设答案)")
    print("  - Python里处理错误           (LLM重写 → Python异常处理)")
    print("  - 诸葛亮的扇子               (LLM重写 → 诸葛亮 鹅毛扇)")

    while True:
        print("\n" + "─" * 80)
        try:
            question = input("❓ 你的问题: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break

            # 如果禁用了重写，直接使用原始查询
            if rag.rewrite_method is None:
                result = rag.query(question)
                # 覆盖显示，假装没有重写
                result['rewritten_query'] = question
            else:
                result = rag.query(question)

            if "error" in result:
                print(f"\n❌ {result['error']}")
                continue

            print("\n" + "─" * 80)
            print("📖 答案:")
            print("─" * 80)
            print(result['answer'])
            print("─" * 80)

            # 显示重写信息
            if result.get('rewritten_query') and result['rewritten_query'] != result['query']:
                print(f"\n💡 查询重写信息:")
                print(f"   原始查询: {result['query']}")
                print(f"   重写查询: {result['rewritten_query']}")
                if result.get('rewrite_score') > 0:
                    print(f"   质量评分: {result['rewrite_score']}/10")
                print(f"   是否使用: {'是' if result.get('use_rewrite') else '否（回退到原查询）'}")
                print(f"   原因: {result.get('rewrite_reason', '')}")

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
