# 语义分块（Semantic Chunking）

> 用 embedding 判断"语义边界"，让每个 chunk 内部高度一致。
> 听起来很美，但坑也不少。

---

## 原理

```
句子序列: S1 → S2 → S3 → S4 → S5 → ...

算每对相邻句子的 embedding 余弦相似度:
  sim(S1, S2) = 0.85
  sim(S2, S3) = 0.82
  sim(S3, S4) = 0.45   ← 显著下降 → 这里语义"换主题"了
  sim(S4, S5) = 0.80

→ 在 S3 和 S4 之间切一刀
```

核心思路：**相邻句语义距离大 → 是个自然的主题切换点**。

---

## 最小实现

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def semantic_chunk(
    text: str,
    embed_fn,
    breakpoint_threshold: float = 0.5,
    min_chunk_size: int = 100,
) -> list[str]:
    """
    text: 待切分的文本
    embed_fn: 函数，输入文本输出 embedding
    breakpoint_threshold: 相似度下降多少个百分点作为切点（0-1）
    """
    # 1. 切成句子
    import re
    sentences = re.split(r'(?<=[。！？.!?])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [text]

    # 2. 算每对相邻句子的 embedding
    embs = np.array([embed_fn(s) for s in sentences])
    similarities = [
        cosine_similarity([embs[i]], [embs[i+1]])[0][0]
        for i in range(len(embs) - 1)
    ]

    # 3. 找切点：相似度低于 (均值 - 阈值 * 标准差)
    sim_arr = np.array(similarities)
    threshold = sim_arr.mean() - breakpoint_threshold * sim_arr.std()

    # 4. 按切点合并句子
    chunks = []
    current = sentences[0]
    for i, sim in enumerate(similarities):
        if sim < threshold and len(current) >= min_chunk_size:
            chunks.append(current)
            current = sentences[i + 1]
        else:
            current += sentences[i + 1]
    if current:
        chunks.append(current)

    return chunks
```

---

## LangChain 等价

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",   # 用百分位数
    breakpoint_threshold_amount=95,           # 相似度最低的 5% 作为切点
    min_chunk_size=100,
)
chunks = splitter.split_text(long_text)
```

---

## 阈值怎么选

`breakpoint_threshold` 决定切多激进：

| 阈值 | 切分粒度 | 适用 |
|------|---------|------|
| 低（< 0.3） | 切得少，chunk 大 | 主题集中 |
| 中（0.5） | 平衡 | 通用 |
| 高（> 0.8） | 切得多，chunk 小 | 主题多变 |

**实测**：在你的 golden set 上扫一遍，找 RAGAS 最高的。

---

## 优点（理论）

✅ chunk 内部语义高度一致
✅ 自动找到主题边界
✅ 对叙事类、长文档友好

---

## 现实问题（必读）

### 问题 1：慢且贵
- 每个句子都要调一次 embedding API
- 10KB 文档 ≈ 100 句 ≈ 100 次 embedding 调用
- 大知识库（100MB）→ 成本爆炸

**对策**：
- 只在初次入库时跑（一次性成本）
- 用本地 embedding 模型（BGE-m3 自部署）

### 问题 2：chunk 大小不可控
- 长主题段 → chunk 可能 5000 字
- 短主题段 → chunk 可能 50 字

**对策**：
- 加 `min_chunk_size` 和 `max_chunk_size`
- 超长再二次切分（用 recursive）

### 问题 3：中文断句不稳
中文用句号分句，但：
- 用得多的是逗号
- 一句话可能很长（包含多个主题）

**对策**：
- 用 LLM 分句（贵但准）
- 用中文 NLP 工具（如 jieba / HanLP）

### 问题 4：embedding 噪声
embedding 模型对短句子的表示不稳。
相邻两个无关短句，可能因为"句子结构相似"而相似度高。

**对策**：
- 把相邻 N 句合并后再算 embedding（窗口法）

---

## 什么时候真的需要语义分块

**优先级判断**：

```
你的文档是结构化的吗？
├── 是 → 文档感知分块（[document_aware.md](./document_aware.md)）就够了
└── 否
    └── 你的文档主题变化频繁吗？
        ├── 是（如长文章、访谈记录） → 语义分块有用
        └── 否（如产品介绍、条款列表） → 递归分块就够了
```

**结论**：90% 的 RAG 项目，**用不到**语义分块。
真正需要的场景：长篇叙事文档、跨主题讨论、生成式 RAG（如对话历史压缩）。

---

## 进阶变体

### Proposition-based（基于命题的分块）
用 LLM 把文档拆成原子命题：

```python
prompt = """
把下面文本拆成独立的、原子化的命题，每个命题一个完整事实。

文本: {text}

输出 JSONL:
{"proposition": "..."}
"""
```

```
原文: "公司年假 15 天起步，工龄 5 年以上 20 天，10 年以上 25 天"
→
"公司年假起步 15 天"
"工龄 5 年以上享 20 天年假"
"工龄 10 年以上享 25 天年假"
```

每个命题作为独立 chunk——召回精度极高。
**代价**：LLM 调用昂贵，仅适合关键场景。

### Late Chunking
先 embed 整文档，再切分。
- 优势：每个 chunk 保留了整个文档的上下文信息
- 是 2024 年的新方法，部分模型（Jina v3）支持

---

## 反模式

❌ **首次跑就上语义分块** → 性价比低，先用递归
❌ **不限制 chunk 大小** → chunk 可能 10KB，爆 context
❌ **用英文断句器处理中文** → 句子分不对，全废
❌ **在每次查询时实时跑语义分块** → 太慢，应入库时一次性做

---

## 实战：在 my_com_rag 试一下

### Step 1：选 1–2 份长文档（5KB+）
不要全库做（成本太高），先小范围试。

### Step 2：对比
```python
# 三种 chunking 跑同一份文档
recursive_chunks = recursive_chunk(doc, 500, 50)
semantic_chunks = semantic_chunk(doc, embed_fn)
proposition_chunks = proposition_chunk(doc, llm)  # 用 LLM

# 各自建索引
# 在 golden set 上跑 RAGAS
```

### Step 3：决策
- 提升显著（>5%）→ 全库迁移
- 提升不显著 → 留在递归

**预期**：大多数场景，提升 < 5%。这是为什么推荐递归为主。

---

## 自测题

1. 为什么语义分块对中文文档效果可能不如英文？
2. 100MB 的知识库全做语义分块，估算成本？
3. semantic chunk 切出来一个 chunk 是 8000 字，怎么办？
4. 什么时候你**会**选语义分块而非递归？
5. Proposition-based chunking 的代价是什么？

---

> 下一步：[document_aware.md](./document_aware.md)
