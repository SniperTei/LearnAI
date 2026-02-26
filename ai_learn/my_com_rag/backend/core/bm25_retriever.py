"""
BM25关键词检索模块
结合向量检索实现混合检索
"""
from typing import List, Dict, Any
from pathlib import Path
import json
import pickle
import math
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25关键词检索器"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化BM25检索器

        Args:
            k1: 调节词频饱和度的参数 (默认1.5)
            b: 调节文档长度归一化的参数 (默认0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_ids = []
        self.idf = {}
        self.doc_lens = []
        self.avg_doc_len = 0
        self.index_file = Path("data/vector_db/bm25_index.pkl")

    def _tokenize(self, text: str) -> List[str]:
        """
        简单的分词（中文按字符切分，英文按单词切分）

        Args:
            text: 输入文本

        Returns:
            分词列表
        """
        tokens = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 中文字符
                tokens.append(char)
            elif char.isalnum():  # 字母或数字
                tokens.append(char.lower())
            elif char.isspace():  # 空格分隔
                pass
        return tokens

    def index_documents(self, documents: List[Document]):
        """
        建立BM25索引

        Args:
            documents: 文档列表
        """
        logger.info(f"开始建立BM25索引，文档数: {len(documents)}")

        self.corpus = []
        self.doc_ids = []
        self.doc_lens = []

        # 处理每个文档
        for idx, doc in enumerate(documents):
            tokens = self._tokenize(doc.page_content)
            self.corpus.append(tokens)
            self.doc_ids.append(idx)
            self.doc_lens.append(len(tokens))

        # 计算平均文档长度
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0

        # 计算IDF
        self._calculate_idf()

        # 保存索引
        self._save_index()

        logger.info(f"BM25索引建立完成，平均文档长度: {self.avg_doc_len:.2f}")

    def _calculate_idf(self):
        """计算逆文档频率（IDF）"""
        n_docs = len(self.corpus)
        df = {}  # 文档频率

        # 统计每个词出现在多少个文档中
        for tokens in self.corpus:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] = df.get(token, 0) + 1

        # 计算IDF
        for token, freq in df.items():
            self.idf[token] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        BM25搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            搜索结果列表 [(doc_id, score), ...]
        """
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = []

        # 计算每个文档的BM25分数
        for doc_idx, doc_tokens in enumerate(self.corpus):
            score = 0
            doc_len = self.doc_lens[doc_idx]

            for token in query_tokens:
                if token not in doc_tokens:
                    continue

                # 计算词频
                tf = doc_tokens.count(token)

                # BM25公式
                idf = self.idf.get(token, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))

                score += idf * (numerator / denominator)

            if score > 0:
                scores.append((doc_idx, score))

        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 返回top k
        return [{"doc_id": doc_id, "score": score} for doc_id, score in scores[:k]]

    def _save_index(self):
        """保存索引到文件"""
        try:
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_file, 'wb') as f:
                pickle.dump({
                    'corpus': self.corpus,
                    'doc_ids': self.doc_ids,
                    'idf': self.idf,
                    'doc_lens': self.doc_lens,
                    'avg_doc_len': self.avg_doc_len,
                    'k1': self.k1,
                    'b': self.b
                }, f)
            logger.info(f"BM25索引已保存: {self.index_file}")
        except Exception as e:
            logger.error(f"保存BM25索引失败: {e}")

    def _load_index(self) -> bool:
        """从文件加载索引"""
        try:
            if not self.index_file.exists():
                return False

            with open(self.index_file, 'rb') as f:
                data = pickle.load(f)

            self.corpus = data['corpus']
            self.doc_ids = data['doc_ids']
            self.idf = data['idf']
            self.doc_lens = data['doc_lens']
            self.avg_doc_len = data['avg_doc_len']
            self.k1 = data.get('k1', 1.5)
            self.b = data.get('b', 0.75)

            logger.info(f"BM25索引已加载，文档数: {len(self.corpus)}")
            return True
        except Exception as e:
            logger.error(f"加载BM25索引失败: {e}")
            return False

    def is_indexed(self) -> bool:
        """检查是否已建立索引"""
        return self.index_file.exists() and len(self.corpus) > 0
