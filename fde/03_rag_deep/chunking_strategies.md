# 分块策略对比（Chunking Strategies）

> 分块是 RAG 的地基。
> **地基错了，上面再花哨的检索都是装饰。**

---

## 为什么要分块

LLM 上下文有限（哪怕 200k 也有成本），知识库是 GB 级。
不可能把整个知识库塞进 prompt → 必须先分成小块再选择性召回。

但分块不只是"切小"——它直接决定了三件事：
1. **召回精度**：chunk 太大，召回的 chunk 里大部分是噪音
2. **答案质量**：chunk 太小，模型拿不到完整上下文，编造
3. **成本**：chunk 大 → token 多 → 成本高、延迟高

**Trade-off 三角**：
```
       精度（小 chunk）
           ↕
       完整性（大 chunk）
           ↕
        成本
```
没有"最佳大小"，**只有最适合你场景的大小**。

---

## 4 种主流分块策略

### 1. 固定大小分块（Fixed-size）
按字符数（或 token）切，可重叠。

```python
def fixed_chunk(text, size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i+size])
    return chunks
```

**优点**：简单、快、可预测
**缺点**：可能切断语义（句子中间断开）

**适用**：
- 内容均匀的文档（手册、合同条款）
- 起步阶段、PoC
- 不需要语义对齐的场景

### 2. 递归分块（Recursive）
按分隔符层级切：先段落，段落太大再按句子，句子太大再按字符。

```python
separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
# 先按双换行（段落）切
# chunk > size → 按单换行切
# 还太大 → 按句号切
# ……
```

**优点**：尽量保留语义边界
**缺点**：chunk 大小不均

**适用**：
- 多数 RAG 场景（默认推荐）
- Markdown / HTML / 结构化文档
- LangChain 的 `RecursiveCharacterTextSplitter` 就是这个

### 3. 语义分块（Semantic Chunking）
按"语义相似度变化"切——相邻句子语义变化大时断开。

```python
# 算每对相邻句子的 embedding 相似度
# 相似度低于阈值 → 在这里切
```

**优点**：每个 chunk 内部语义高度一致
**缺点**：
- 慢（要算 embedding）
- chunk 大小不可控
- 中文表现不如英文（句号不如英文清晰）

**适用**：
- 长文、叙事类（文章、小说、演讲）
- chunk 内一致性要求高（学术检索）

详见 [chunking/semantic_chunking.md](./chunking/semantic_chunking.md)

### 4. 文档感知分块（Document-aware）
按文档结构切：标题 / 章节 / 列表 / 表格。

```python
# Markdown：按 # ## ### 切
# HTML：按 <section> / <div> 切
# PDF：按章节标题切
# 代码：按函数 / 类切
```

**优点**：保留文档逻辑结构，元数据丰富
**缺点**：每种文档格式要写解析器

**适用**：
- 结构化文档（你 my_com_rag 的知识库很可能就是）
- 需要按章节追溯的场景（法律、技术文档）

详见 [chunking/document_aware.md](./chunking/document_aware.md)

---

## 关键参数（必懂）

### chunk_size
**经验值**：
- 256 tokens：短事实查询（"年假多少天"）
- 512 tokens：通用默认值
- 1024 tokens：长推理（"对比这两份合同"）
- 2048+ tokens：摘要任务

**国内中文场景**：1 个汉字 ≈ 1–2 tokens，512 tokens ≈ 300–500 字。

### overlap（重叠）
相邻 chunk 重叠的字符/token 数。

**作用**：避免切断上下文。

```
chunk1: [..........]
chunk2:        [..........]   ← 中间重叠部分
```

**经验值**：chunk_size 的 10–20%。
- 512 chunk → 50–100 overlap
- 太大：冗余、成本高
- 太小：边界丢信息

### 分隔符优先级
不同分隔符对"语义破坏"程度不同：

```
\n\n     ← 段落边界（最该切）
\n       ← 行边界
。       ← 句子边界
，       ← 分句边界（差）
空格     ← 词边界（很差）
字符     ← 字符边界（最差）
```

**递归分块的本质**：从最该切的开始尝试，不行再退一档。

---

## 不同 chunk 大小的实测影响

假设同一份知识库，跑 RAGAS：

| chunk size | Context Precision | Faithfulness | 成本/请求 |
|-----------|-------------------|--------------|-----------|
| 128 | 高（精准） | 低（信息不全） | 低 |
| 256 | 中-高 | 中 | 低 |
| 512 | 中 | 中-高 | 中 |
| 1024 | 低（噪音多） | 高（信息全） | 高 |
| 2048 | 低 | 高 | 高 |

**这不是绝对的**——具体数字取决于：
- 你的 embedding 模型对长短文本的偏好
- query 类型（短事实 vs 长分析）
- 文档本身

**唯一可信的方法**：**实测**。

---

## 一个对比实验（必做）

给 my_com_rag 做这个实验，就知道哪个 size 适合你：

```python
import pandas as pd

results = []
for size in [128, 256, 512, 1024]:
    for overlap in [0, 50, 100]:
        # 重新切分知识库
        rebuild_index(chunk_size=size, overlap=overlap)

        # 跑 RAGAS
        score = ragas_eval(golden_set)

        results.append({
            "chunk_size": size,
            "overlap": overlap,
            "faithfulness": score.faithfulness,
            "context_precision": score.context_precision,
            "avg_tokens": measure_avg_tokens(),
        })

df = pd.DataFrame(results)
print(df.sort_values("faithfulness", ascending=False))
```

**结果会告诉你**：你的场景下，什么 chunk size 是最优。

---

## 进阶策略

### 父子分块（Parent-Child / Small-to-Big）
- **检索用小 chunk**（精准定位）
- **生成用大 chunk**（提供完整上下文）

```
检索：用 256-token 的子 chunk 找到位置
返回：取该子 chunk 所在的 1024-token 父 chunk 给 LLM
```

**优势**：兼顾精度和完整性，目前被认为是 RAG 的高级最佳实践之一。

### 基于命题的分块（Proposition-based）
用 LLM 把文档拆成"独立命题"：

```
原文："公司入职享 15 天年假，工龄 5 年以上享 20 天，10 年以上享 25 天"
→
命题 1: "入职享 15 天年假"
命题 2: "工龄 5 年以上享 20 天"
命题 3: "工龄 10 年以上享 25 天"
```

每个命题作为独立 chunk，召回精度极高。
**代价**：用 LLM 切分，慢且贵。

### 滑动窗口 + 摘要
每个 chunk 都附一个 LLM 生成的摘要。
检索时同时召回原文和摘要。

---

## 决策树：你的项目应该用哪个

```
你的文档是结构化的吗？
├── 是（Markdown / HTML / PDF 有清晰结构）
│   └── 文档感知分块（推荐）
└── 否
    └── 你需要 chunk 大小可控吗？
        ├── 是（控制成本/性能）
        │   └── 递归分块（默认）
        └── 否（追求语义最优）
            └── 语义分块

补充：
- 检索精度是瓶颈？→ 加 Small-to-Big（父子）
- 文档非常长且复杂？→ 加 Proposition-based
```

---

## 反模式

❌ **不分块**（整文档丢进去）→ context 巨长，成本爆炸，注意力衰减
❌ **盲目用默认值**（LangChain 默认 1000）→ 你的场景不一定合适
❌ **chunk 太小（64）** → 召回的 chunk 信息不全，模型编造
❌ **chunk 太大（8192）** → 噪音多，注意力散，成本高
❌ **不分文档类型**（PDF 和 Markdown 用同一套）→ 浪费结构信息
❌ **不评估就改 chunk size** → 凭感觉优化，可能改反了

---

## 实战：给 my_com_rag 做分块优化

### Phase 1：基线
- 用当前 chunking 跑 RAGAS，记录基线

### Phase 2：扫参数
- chunk size: 256 / 512 / 1024
- overlap: 0 / 50 / 100
- 共 9 组实验
- 找出最优组合

### Phase 3：换策略
- 用最优 size，但换递归 → 文档感知
- 看是否进一步提升

### Phase 4：进阶
- 加 Small-to-Big
- 测提升

**产出**：一份"分块策略对比报告"——这是你 FDE 简历上的硬资产。

---

## 自测题

1. 为什么 chunk 不是越小越好？
2. overlap 设 100% 会怎样？设 0 会怎样？
3. 你的 chunk 是 512，但 RAGAS Context Recall 很低，可能什么原因？
4. 同一份文档用递归分块和语义分块，结果差多大？
5. Small-to-Big 为什么比单一 chunking 好？

---

> 下一步：
> - [chunking/fixed_vs_recursive.md](./chunking/fixed_vs_recursive.md) — 实测代码
> - [embedding_theory.md](./embedding_theory.md) — 向量到底编码了什么
