"""
三国知识库 + 混合检索系统
==========================

使用本地 Ollama 模型 + 混合检索构建的三国知识库问答系统。
演示 RAG 高效召回方法：混合检索（Hybrid Search）

混合检索 = 向量检索（FAISS）+ 关键词检索（BM25）+ 结果融合（RRF）

功能：
1. FAISS 向量检索 - 语义相似度匹配
2. BM25 关键词检索 - 精确关键词匹配
3. RRF 结果融合 - 结合两种检索结果
4. 对比三种检索方式的效果

特点：
- 向量检索擅长语义理解（"三国演义" ↔ "三国"）
- BM25擅长关键词匹配（"诸葛亮" ↔ "孔明"）
- 两者互补，检索效果显著提升

作者: Claude Code Assistant
日期: 2026-01-31
"""

import os
import json
import pickle
import math
from typing import List, Dict, Tuple
from pathlib import Path
import re
import requests
from collections import Counter

import numpy as np
import faiss

# ============================================================================
# 配置
# ============================================================================

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "deepseek-r1:7b"

DOCS_DIR = "knowledge_threekingdoms"
INDEX_PREFIX = "threekingdoms_hybrid"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
VECTOR_TOP_K = 50
BM25_TOP_K = 50
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
    """智能文本分割器 - 按段落/句子分块"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, metadata: str = "") -> List[Dict]:
        """
        智能分割文本（按段落优先）

        策略：
        1. 优先按段落（\n\n）分块
        2. 段落太长时，按句子（。）切分
        3. 保持语义完整性
        """
        chunks = []
        chunk_id = 0

        # 第一步：按段落分割（保留单换行作为段落内的换行）
        paragraphs = text.split('\n\n')

        current_chunk = ""
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para = para.replace('\n', ' ')  # 段落内的换行替换为空格
            para_length = len(para)

            # 如果单个段落就超过了chunk_size，需要切分
            if para_length > self.chunk_size:
                # 先保存当前chunk
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": metadata,
                        "chunk_id": chunk_id
                    })
                    chunk_id += 1
                    current_chunk = ""
                    current_length = 0

                # 切分长段落（按句子）
                sentences = self._split_long_paragraph(para)
                for sent in sentences:
                    if current_length + len(sent) + 2 <= self.chunk_size:
                        current_chunk += sent + "。"
                        current_length += len(sent) + 1
                    else:
                        if current_chunk:
                            chunks.append({
                                "content": current_chunk.strip(),
                                "metadata": metadata,
                                "chunk_id": chunk_id
                            })
                            chunk_id += 1
                        # 添加重叠
                        if self.chunk_overlap > 0 and current_chunk:
                            overlap_text = current_chunk[-self.chunk_overlap:]
                            current_chunk = overlap_text + sent + "。"
                            current_length = len(current_chunk)
                        else:
                            current_chunk = sent + "。"
                            current_length = len(sent) + 1

            # 如果段落可以放入当前chunk
            elif current_length + para_length + 2 <= self.chunk_size:
                current_chunk += "\n\n" + para
                current_length += para_length + 2

            # 需要新的chunk
            else:
                # 保存当前chunk
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": metadata,
                        "chunk_id": chunk_id
                    })
                    chunk_id += 1

                # 添加重叠
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + "\n\n" + para
                    current_length = len(overlap_text) + para_length + 2
                else:
                    current_chunk = para
                    current_length = para_length

        # 最后一个chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": metadata,
                "chunk_id": chunk_id
            })

        return chunks

    def _split_long_paragraph(self, text: str) -> List[str]:
        """
        切分长段落（按句子）

        保持句子完整性
        """
        sentences = []
        current_sent = ""

        i = 0
        while i < len(text):
            char = text[i]

            # 遇到句号、问号、感叹号，切分
            if char in ['。', '！', '？', '；']:
                current_sent += char
                if current_sent.strip():
                    sentences.append(current_sent.strip())
                current_sent = ""

            # 遇到换行符，也切分
            elif char == '\n':
                if current_sent.strip():
                    sentences.append(current_sent.strip())
                current_sent = ""

            else:
                current_sent += char

            i += 1

        # 最后一句
        if current_sent.strip():
            sentences.append(current_sent.strip())

        return sentences

    def split_documents(self, documents: List[Tuple[str, str]]) -> List[Dict]:
        """分割多个文档"""
        all_chunks = []
        for filename, content in documents:
            chunks = self.split_text(content, metadata=filename)
            all_chunks.extend(chunks)
        return all_chunks


# ============================================================================
# 中文分词（使用jieba + 三国专有词汇）
# ============================================================================

import jieba

class ChineseTokenizer:
    """使用jieba的中文分词器（加载三国词典）"""

    @staticmethod
    def load_threekingdoms_dict():
        """加载三国专有词汇"""
        # 三国人名、地名、武器名等
        threekingdoms_words = [
            # 人名
            '诸葛亮', '孔明', '卧龙', '刘备', '玄德', '关羽', '云长', '张飞', '翼德',
            '曹操', '孟德', '孙权', '仲谋', '周瑜', '公瑾', '吕布', '奉先', '赵云',
            '子龙', '黄忠', '汉升', '马超', '孟起', '魏延', '文长', '姜维', '伯约',
            '司马懿', '仲达', '陆逊', '伯言', '孙策', '伯符', '黄盖', '公覆',
            '董卓', '仲颖', '袁绍', '本初', '袁术', '公路', '刘表', '景升',
            # 地名
            '赤壁', '荆州', '益州', '江东', '中原', '洛阳', '长安', '成都',
            '建业', '许昌', '邺城', '合肥', '濡须', '夷陵', '五丈原',
            # 武器
            '青龙偃月刀', '丈八蛇矛', '方天画戟', '双股剑', '雌雄双剑',
            '羽扇', '鹤氅', '赤兔马', '的卢', '绝影',
            # 战役
            '官渡之战', '赤壁之战', '夷陵之战', '五丈原', '桃园结义',
            '三顾茅庐', '草船借箭', '空城计', '火烧连营',
            # 职位
            '丞相', '太尉', '大将军', '都督', '太守', '刺史',
            # 其他
            '三国演义', '三国志', '魏蜀吴'
        ]

        for word in threekingdoms_words:
            jieba.add_word(word, freq=10000)  # 高频词

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        使用jieba进行中文分词

        优化：
        1. 使用jieba精确模式
        2. 加载三国专有词汇
        3. 过滤停用词（标点符号）
        4. 保留有意义的词
        """
        # 首次使用时加载词典
        if not hasattr(ChineseTokenizer, '_dict_loaded'):
            ChineseTokenizer.load_threekingdoms_dict()
            ChineseTokenizer._dict_loaded = True

        # 使用jieba分词
        words = jieba.cut(text, cut_all=False)

        # 过滤停用词
        tokens = []
        stopwords = {' ', '\n', '\t', '\r', '，', '。', '！', '？', '；', '：', '、',
                    '「', '」', '『', '』', '（', '）', '【', '】', '《', '》',
                    ',', '.', '!', '?', ';', ':', '"', '"', "'", "'",
                    '的', '了', '是', '在', '和', '有', '我', '你', '他', '她',
                    '它', '们', '这', '那', '就', '也', '都', '而', '及', '与'}

        for word in words:
            word = word.strip()
            # 过滤单字停用词和标点
            if word and word not in stopwords and len(word) > 0:
                # 保留：中文字符、数字、英文
                if any('\u4e00' <= c <= '\u9fff' or c.isalnum() for c in word):
                    tokens.append(word)

        return tokens


# ============================================================================
# Ollama Embedding
# ============================================================================

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
        """获取文本的 Embedding"""
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


# ============================================================================
# BM25 索引
# ============================================================================

class BM25Index:
    """
    BM25 索引 - 关键词检索

    BM25 是一种改进的 TF-IDF 算法，考虑了：
    1. 词频饱和度（词频越高，权重增长越慢）
    2. 文档长度归一化
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        参数:
            k1: 词频饱和度参数（1.2更保守，适合短文档）
            b: 长度归一化参数（0.75是标准值）
        """
        self.k1 = k1
        self.b = b
        self.chunks = []
        self.corpus = []  # 分词后的文档
        self.doc_freqs = {}  # 文档频率
        self.idf = {}  # 逆文档频率
        self.doc_lens = []  # 文档长度
        self.avgdl = 0  # 平均文档长度

    def build_index(self, chunks: List[Dict]):
        """构建 BM25 索引"""
        self.chunks = chunks

        # 分词
        print("📝 正在分词...")
        for chunk in chunks:
            tokens = ChineseTokenizer.tokenize(chunk['content'])
            self.corpus.append(tokens)
            self.doc_lens.append(len(tokens))

        # 计算平均文档长度
        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0

        # 计算文档频率
        print("📊 正在计算文档频率...")
        for tokens in self.corpus:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        # 计算 IDF
        print("📈 正在计算 IDF...")
        N = len(self.corpus)
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

        print(f"✅ BM25 索引构建完成: {len(chunks)} 个文档块")
        print(f"   词汇量: {len(self.doc_freqs):,}")
        print(f"   平均文档长度: {self.avgdl:.1f}")

    def save(self):
        """保存索引"""
        with open(f"{INDEX_PREFIX}_bm25.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'corpus': self.corpus,
                'doc_freqs': self.doc_freqs,
                'idf': self.idf,
                'doc_lens': self.doc_lens,
                'avgdl': self.avgdl,
                'k1': self.k1,
                'b': self.b
            }, f)
        print(f"✅ BM25 索引已保存: {INDEX_PREFIX}_bm25.pkl")

    def load(self):
        """加载索引"""
        with open(f"{INDEX_PREFIX}_bm25.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.corpus = data['corpus']
            self.doc_freqs = data['doc_freqs']
            self.idf = data['idf']
            self.doc_lens = data['doc_lens']
            self.avgdl = data['avgdl']
            self.k1 = data['k1']
            self.b = data['b']
        print(f"✅ BM25 索引已加载: {len(self.chunks)} 个文档块")

    def search(self, query: str, top_k: int = BM25_TOP_K, debug: bool = False) -> List[Dict]:
        """BM25 检索（带调试信息）"""
        # 分词
        query_tokens = ChineseTokenizer.tokenize(query)

        if debug:
            print(f"  🔍 BM25分词结果: {query_tokens}")

        # 计算每个文档的 BM25 分数
        scores = []
        for i, doc_tokens in enumerate(self.corpus):
            score = 0
            doc_len = self.doc_lens[i]

            for token in query_tokens:
                if token not in doc_tokens:
                    continue

                # 词频
                tf = doc_tokens.count(token)

                # IDF
                idf = self.idf.get(token, 0)

                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score += idf * (numerator / denominator)

            scores.append((i, score))

        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 返回 top-k
        results = []
        for idx, score in scores[:top_k]:
            if score > 0:  # 只返回匹配的文档
                chunk = self.chunks[idx].copy()
                chunk['bm25_score'] = score
                chunk['bm25_rank'] = len(results) + 1
                results.append(chunk)

        if debug:
            print(f"  ✅ BM25检索完成: {len(results)} 个匹配文档")

        return results


# ============================================================================
# FAISS 向量索引
# ============================================================================

class FaissIndex:
    """FAISS 向量索引"""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        self.chunks = []
        self.embeddings = None

    def build_index(self, chunks: List[Dict], embeddings: List[List[float]]):
        """构建索引"""
        self.chunks = chunks
        self.embeddings = np.array(embeddings, dtype='float32')

        if self.embeddings.shape[1] != self.dimension:
            self.dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings)

        print(f"✅ FAISS 索引构建完成: {len(chunks)} 个文档块")

    def save(self):
        """保存索引"""
        faiss.write_index(self.index, f"{INDEX_PREFIX}_faiss.bin")
        with open(f"{INDEX_PREFIX}_chunks.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings
            }, f)
        print(f"✅ FAISS 索引已保存: {INDEX_PREFIX}_faiss.bin")

    def load(self):
        """加载索引"""
        self.index = faiss.read_index(f"{INDEX_PREFIX}_faiss.bin")
        with open(f"{INDEX_PREFIX}_chunks.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.embeddings = data['embeddings']

        self.dimension = self.index.d
        print(f"✅ FAISS 索引已加载: {len(self.chunks)} 个文档块")

    def search(self, query_embedding: List[float], top_k: int = VECTOR_TOP_K) -> List[Dict]:
        """向量检索"""
        query_array = np.array([query_embedding], dtype='float32')
        distances, indices = self.index.search(query_array, top_k)

        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['vector_score'] = float(dist)
                chunk['vector_rank'] = i + 1
                results.append(chunk)

        return results


# ============================================================================
# 混合检索结果融合（RRF）
# ============================================================================

class HybridSearchFusion:
    """
    混合检索结果融合

    使用 RRF (Reciprocal Rank Fusion) 算法融合多种检索结果

    RRF 公式：
    score(d) = Σ 1/(k + rank_i(d))

    其中 k 是常数（通常为60），rank_i 是文档在第i种检索方法中的排名
    """

    def __init__(self, k: int = 60):
        """
        参数:
            k: RRF 常数（默认60）
        """
        self.k = k

    def fuse_results(self, vector_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        """
        融合向量检索和BM25检索结果

        参数:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果

        返回:
            融合后的结果
        """
        # 使用字典存储融合分数 {chunk_id: score}
        fused_scores = {}
        chunk_info = {}  # 存储chunk信息

        # 处理向量检索结果
        for result in vector_results:
            chunk_id = result.get('chunk_id', -1)
            rank = result.get('vector_rank', len(vector_results))

            # RRF 分数
            score = 1.0 / (self.k + rank)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + score

            # 保存信息（第一次遇到时）
            if chunk_id not in chunk_info:
                chunk_info[chunk_id] = {
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'chunk_id': chunk_id,
                    'vector_rank': rank,
                    'vector_score': result.get('vector_score', 0),
                    'bm25_rank': None,
                    'bm25_score': 0
                }

        # 处理 BM25 检索结果
        for result in bm25_results:
            chunk_id = result.get('chunk_id', -1)
            rank = result.get('bm25_rank', len(bm25_results))

            # RRF 分数
            score = 1.0 / (self.k + rank)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + score

            # 更新信息
            if chunk_id in chunk_info:
                chunk_info[chunk_id]['bm25_rank'] = rank
                chunk_info[chunk_id]['bm25_score'] = result.get('bm25_score', 0)
            else:
                chunk_info[chunk_id] = {
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'chunk_id': chunk_id,
                    'vector_rank': None,
                    'vector_score': 0,
                    'bm25_rank': rank,
                    'bm25_score': result.get('bm25_score', 0)
                }

        # 按融合分数排序
        sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        final_results = []
        for i, (chunk_id, fused_score) in enumerate(sorted_chunks):
            chunk = chunk_info[chunk_id].copy()
            chunk['fused_score'] = fused_score
            chunk['fused_rank'] = i + 1
            final_results.append(chunk)

        return final_results


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
# 混合检索 RAG 系统
# ============================================================================

class HybridSearchRAG:
    """混合检索 RAG 系统"""

    def __init__(self):
        self.embedder = OllamaEmbedding()
        self.chat = OllamaChat()
        self.faiss_index = FaissIndex()
        self.bm25_index = BM25Index()
        self.fusion = HybridSearchFusion()

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
        print("\n第三步：生成 Embedding（用于向量检索）")
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedder.get_embeddings_batch(texts, batch_size=10)

        # 构建 FAISS 索引
        print("\n第四步：构建 FAISS 向量索引")
        self.faiss_index.build_index(chunks, embeddings)

        # 构建 BM25 索引
        print("\n第五步：构建 BM25 关键词索引")
        self.bm25_index.build_index(chunks)

        # 保存索引
        print("\n第六步：保存索引")
        self.faiss_index.save()
        self.bm25_index.save()
        print("\n✨ 三国知识库构建完成！")

    def load_knowledge_base(self):
        """加载知识库"""
        self.faiss_index.load()
        self.bm25_index.load()
        print("✅ 三国知识库已加载")

    def query(self, question: str, show_comparison: bool = True) -> Dict:
        """
        混合检索查询

        参数:
            question: 用户问题
            show_comparison: 是否显示三种检索方式对比
        """
        print(f"\n{'=' * 80}")
        print(f"🔍 查询问题: {question}")
        print(f"{'=' * 80}")

        # 1️⃣ 向量检索
        print(f"\n📊 第一步：向量检索（语义相似度）")
        query_embedding = self.embedder.get_embedding(question)
        if not query_embedding:
            return {"error": "无法生成查询向量"}

        vector_results = self.faiss_index.search(query_embedding, top_k=VECTOR_TOP_K)
        print(f"✅ 向量检索完成，召回 {len(vector_results)} 个文档")

        # 2️⃣ BM25 检索
        print(f"\n🔍 第二步：BM25 检索（关键词匹配）")
        bm25_results = self.bm25_index.search(question, top_k=BM25_TOP_K, debug=True)
        print(f"✅ BM25 检索完成，召回 {len(bm25_results)} 个文档")

        # 3️⃣ 结果融合
        print(f"\n🔄 第三步：结果融合（RRF 算法）")
        fused_results = self.fusion.fuse_results(vector_results, bm25_results)
        print(f"✅ 融合完成")

        # 4️⃣ 显示对比
        if show_comparison:
            self._show_comparison(vector_results, bm25_results, fused_results)

        # 5️⃣ 最终结果
        final_results = fused_results[:FINAL_TOP_K]

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
            "vector_results": vector_results[:FINAL_TOP_K],
            "bm25_results": bm25_results[:FINAL_TOP_K],
            "query": question
        }

    def _show_comparison(self, vector_results: List[Dict], bm25_results: List[Dict], fused_results: List[Dict]):
        """显示三种检索方式的对比"""
        print(f"\n{'=' * 80}")
        print("📊 三种检索方式对比（Top-5）")
        print('=' * 80)

        print("\n【向量检索 Top-5】（语义相似度）")
        for i, r in enumerate(vector_results[:5]):
            rank = r.get('vector_rank', i+1)
            score = r.get('vector_score', 0)
            print(f"  [{i+1}] #{rank:2d} {r['metadata'][:45]:45s} (L2: {score:.4f})")

        print("\n【BM25 检索 Top-5】（关键词匹配）")
        for i, r in enumerate(bm25_results[:5]):
            rank = r.get('bm25_rank', i+1)
            score = r.get('bm25_score', 0)
            print(f"  [{i+1}] #{rank:2d} {r['metadata'][:45]:45s} (BM25: {score:.2f})")

        print("\n【混合检索 Top-5】（融合结果）⭐")
        for i, r in enumerate(fused_results[:5]):
            v_rank = r.get('vector_rank', '-')
            b_rank = r.get('bm25_rank', '-')
            score = r.get('fused_score', 0)

            v_str = f"#{v_rank}" if v_rank != '-' else " - "
            b_str = f"#{b_rank}" if b_rank != '-' else " - "

            print(f"  [{i+1}] {r['metadata'][:45]:45s}")
            print(f"      向量排名: {v_str:3s} | BM25排名: {b_str:3s} | 融合分数: {score:.4f}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""

    print("=" * 80)
    print("🤖 三国知识库 + 混合检索系统")
    print("=" * 80)
    print("\n📚 基于《三国演义》1.7MB 文本")
    print("🎯 演示 RAG 高效召回方法：混合检索（Hybrid Search）")
    print("\n🔍 混合检索 = 向量检索（FAISS）+ BM25 + RRF 融合")

    # 检查 Ollama
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            print("\n❌ Ollama 未运行")
            return
    except:
        print("\n❌ 无法连接 Ollama")
        return

    # 创建系统
    try:
        rag = HybridSearchRAG()
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        print("\n💡 请确保已下载 embedding 模型: ollama pull nomic-embed-text")
        return

    # 构建或加载知识库
    if Path(f"{INDEX_PREFIX}_faiss.bin").exists() and Path(f"{INDEX_PREFIX}_bm25.pkl").exists():
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

    print("\n💡 示例问题（观察混合检索的效果）:")
    print("  - 诸葛亮的扇子（关键词+语义）")
    print("  - 赤壁之战的胜利者")
    print("  - 关羽的武器（青龙偃月刀）")
    print("  - 刘备的三顾茅庐")

    while True:
        print("\n" + "─" * 80)
        try:
            question = input("❓ 你的问题: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break

            result = rag.query(question, show_comparison=True)

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
