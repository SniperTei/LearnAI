# Embedding 到底编码了什么

> 你调过 embedding API，但可能从没想过——
> 那 1536 个数字，到底代表什么？

---

## Embedding 是什么

把任意文本映射成固定维度的向量：

```
"年假"          → [0.21, -0.43, 0.88, ..., 0.12]  (1536 维)
"annual leave"  → [0.19, -0.41, 0.85, ..., 0.10]  (相近)
"菜单"          → [-0.55, 0.32, 0.10, ..., -0.7]  (远离)
```

**核心性质**：语义相近的文本，向量也相近。

度量相似度通常用：
- **余弦相似度**（最常用）：方向是否一致
- **欧氏距离**：绝对距离
- **点积**：兼顾方向和长度

---

## Embedding 编码了什么（最关键的认知）

不是"文本字面"，而是**文本在大量语料中学到的"语义上下文"**。

### 关键推论 1：相似性 ≠ 同义
```
"我喜欢这家餐厅" → embedding A
"这家餐厅不错"   → embedding B（相似）
"我讨厌这家餐厅" → embedding C（也比较相似！）
```

**embedding 区分不了"喜欢"和"讨厌"——它们上下文高度重合**（餐厅、评价、情感词）。

**对策**：情感分析、矛盾检测不能纯靠 embedding，需要专门训练或用 LLM。

### 关键推论 2：Embedding 是"主题聚类"
```
"年假政策"     → 接近 "员工福利"、"HR"
"年假 15 天"   → 接近 "具体数字"
```

embedding 把"主题相关"的内容聚在一起。
**这对"找相关文档"很好用，对"找精确答案"不够**。

### 关键推论 3：Embedding 不编码事实
```
"巴黎是法国首都"
"巴黎是日本首都"  ← embedding 高度相似（结构相同）
```

embedding 不知道哪个是对的——它只看"文本长什么样"，不看"文本真假"。

**这是为什么 RAG 需要 reranking + LLM 判断**，光靠 embedding 不够。

### 关键推论 4：Embedding 长度敏感
```
"年假"               → embedding A
"公司年假政策详细"   → embedding B（可能不太相似）
```

短文本信息少，embedding 可能不稳定。

**对策**：
- 短 query 用 instruction prefix（如 BGE 的 "为这个句子生成表示..."）
- chunk 不要太短（< 50 字慎用）

---

## 主流 Embedding 模型对比

### 英文 / 通用
| 模型 | 维度 | 特点 | 成本 |
|------|------|------|------|
| OpenAI text-embedding-3-small | 1536 | 通用强，国外 API | 便宜 |
| OpenAI text-embedding-3-large | 3072 | 顶级，长文本好 | 中 |
| Cohere embed-v3 | 1024 | 支持搜索/分类多任务 | 中 |
| Voyage AI | 视模型 | Anthropic 推荐，质量高 | 中-高 |

### 中文 / 多语言
| 模型 | 维度 | 特点 | 成本 |
|------|------|------|------|
| BGE-large-zh-v1.5 | 1024 | 中文 SOTA，开源免费 | 自部署 |
| BGE-m3 | 1024 | 多语言、多功能（稠密+稀疏+ColBERT） | 自部署 |
| E5 / GTE / Jina v3 | 不同 | 各有强项 | 自部署 |
| 阿里通义 embedding | 不同 | 国内 API 方便 | 便宜 |

### 选型建议（FDE 视角）

```
是否数据敏感 / 要私有化？
├── 是 → BGE-m3 / BGE-large-zh（自部署）
└── 否
    └── 中文为主？
        ├── 是 → 阿里通义 / BGE-m3 API / Voyage-multilingual
        └── 否 → OpenAI text-embedding-3-large / Voyage-3
```

**别纠结**：BGE-m3 是目前开源里最强的中文/多语言选择，闭源选 OpenAI 3-large 或 Voyage。

---

## 维度（Dimension）怎么选

模型固定后维度就定了，但有些模型支持**降维**：
- OpenAI text-embedding-3 支持 `dimensions` 参数
- Matryoshka 训练的模型可以截断

**维度 trade-off**：
| 维度 | 精度 | 存储 | 检索速度 |
|------|------|------|---------|
| 3072 | 最高 | 大 | 慢 |
| 1536 | 高 | 中 | 中 |
| 768 | 中 | 小 | 快 |
| 384 | 低 | 极小 | 极快 |

**经验**：
- 千万级以下文档：1536 是甜点
- 亿级文档：考虑 768 + 量化
- 实时性要求极高：384 + 量化

---

## Embedding 的常见坑

### 坑 1：query 和 chunk 用不同前缀
不对称检索（asymmetric retrieval）：

```
# 错误做法
query_emb = embed("年假多少天")
doc_emb = embed("员工手册第三章：入职享 15 天年假...")

# 部分模型（如 BGE）要求 query 加前缀
query_emb = embed("为这个查询生成表示以用于检索相关文档：年假多少天")
doc_emb = embed("员工手册第三章：...")  # 文档不加
```

**务必看模型文档**！E5、BGE、GTE 都有不同前缀规则。

### 坑 2：跨语言不可靠
```
查询: "annual leave policy"
文档: "年假政策"
```
embedding 不一定把它们拉近——除非用多语言模型（BGE-m3、multilingual-E5）。

**对策**：
- 中英混合场景必用多语言模型
- 或：先翻译再 embedding（成本高）

### 坑 3：长文本截断
每个 embedding 模型都有 max_tokens（通常 512 tokens）。
超出会被截断 → 长文档后半部分信息丢失。

**对策**：
- chunk 别超过模型 max_tokens
- 长文档先用 LLM 生成摘要，再 embed 摘要

### 坑 4：旧 embedding 不能复用
换 embedding 模型 → **所有向量必须重算**。
不同模型的 embedding 空间不兼容。

**对策**：
- 早期评估阶段别频繁换
- 上线后，每次换模型 = 一次重建索引的项目

### 坑 5：归一化 / 距离
不同库对距离的定义不同：
- FAISS 默认 L2
- 余弦相似度需要先归一化

**对策**：明确你的距离度量，全栈一致。

---

## Embedding 评估指标

### 任务级评估
1. **检索任务**：给定 query，能否召回正确文档？
   - MRR（Mean Reciprocal Rank）
   - NDCG@k
   - Recall@k
2. **聚类任务**：相似文本能否聚到一起？
3. **分类任务**：embedding + 简单分类器效果如何？

### MTEB 排行榜
HuggingFace 的 MTEB 是 embedding 模型的标准排行榜。
**FDE 必看**——选型前先查 MTEB 中文/多语言榜。

**注意**：MTEB 上排名靠前的，**不一定**在你具体业务场景好。
**最终还是要自己测**。

---

## 一个最小对比实验

```python
from FlagEmbedding import FlagModel
from openai import OpenAI

queries = ["年假多少天", "怎么申请报销", ...]
docs = ["员工手册：年假 15 天...", "财务规定：报销需发票...", ...]
relevant = [(0, 0), (1, 1), ...]  # 哪个 query 对应哪个 doc

def eval_embed(name, embed_fn):
    q_embs = [embed_fn(q) for q in queries]
    d_embs = [embed_fn(d) for d in docs]

    hit = 0
    for q_idx, d_idx in relevant:
        sims = [cosine(q_embs[q_idx], d) for d in d_embs]
        if sims.index(max(sims)) == d_idx:
            hit += 1
    print(f"{name}: Recall@1 = {hit/len(relevant):.2%}")

# 对比
bge = FlagModel("BAAI/bge-large-zh-v1.5")
eval_embed("BGE-zh", lambda x: bge.encode(x))

client = OpenAI()
eval_embed("OpenAI-3-large", lambda x: client.embeddings.create(
    input=x, model="text-embedding-3-large"
).data[0].embedding)
```

**FDE 必做的功课**：用业务数据测，别迷信榜单。

---

## Embedding + 数据库的工程现实

```
查询流程:
1. 用户 query → embedding
2. 向量库做 ANN（近似最近邻）搜索 → top-K
3. （可选）rerank
4. 拼 prompt → LLM 生成

关注点:
- ANN 速度：HNSW / IVF / Flat
- 内存 vs 磁盘
- 增删改的代价
```

详见 [vector_db_comparison.md](./vector_db_comparison.md)

---

## 反模式

❌ **embedding 一切**——把它当万能相似度工具
❌ **不看模型文档**——前缀、长度、归一化全靠猜
❌ **跨语言用纯英文模型**——中文场景必然翻车
❌ **频繁换模型**——每次都重建索引
❌ **不测就用**——盲信 MTEB 排名
❌ **embedding 当唯一检索信号**——加 BM25 / rerank 才稳

---

## 自测题

1. 为什么 "我喜欢" 和 "我讨厌" 的 embedding 可能很相似？
2. embedding 能判断事实真假吗？为什么？
3. BGE-zh 和 OpenAI text-embedding-3，你的中文 RAG 选哪个？为什么？
4. 换 embedding 模型，老索引能用吗？
5. 你的 query 是 5 个字，chunk 是 1000 字，会有什么问题？

---

> 下一步：[vector_db_comparison.md](./vector_db_comparison.md)
