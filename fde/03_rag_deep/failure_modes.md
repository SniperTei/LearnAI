# RAG 失败的 5 大模式

> 评估做得再好，不会诊断失败 = 不会改进。
> RAG 失败长什么样、根因是什么——FDE 必备的诊断能力。

---

## 总框架：失败发生在哪一环

```
用户 query
    │
    │ ① 查询理解失败
    ↓
改写后的 query
    │
    │ ② 检索失败
    ↓
召回的 chunks
    │
    │ ③ 上下文整合失败
    ↓
prompt
    │
    │ ④ 生成失败
    ↓
答案
    │
    │ ⑤ 任务完成失败
    ↓
用户
```

每一环都可能挂，**症状相似但根因不同**。
FDE 的核心能力：**从症状反推是哪一环挂了**。

---

## ① 查询理解失败

### 症状
- 用户问 A，系统回答 B
- 同一问题，换个问法就答不对

### 根因
**用户的真实意图和字面 query 不一致**。

```
用户: "我入职三个月了"
字面: 信息陈述
意图: 想知道自己享有什么福利 / 试用期如何
```

### 修复方向
- **Query Rewriting**：用 LLM 把口语化 query 改成"检索友好"版本
- **HyDE**：让 LLM 先生成假设答案，用假设答案去检索（详见 [retrieval/hyde.md](./retrieval/hyde.md)）
- **Multi-Query**：生成多个变体 query，并行检索后融合
- **Clarification**：当 query 模糊时，反问用户

---

## ② 检索失败

### 子类型 2a：召回错误（该召回的没召回）

**症状**：Context Recall 低（RAGAS）

**根因**：
- chunk 太小 → 完整答案被切成两半，一半没召回
- embedding 模型对该主题不敏感
- query 和文档措辞差异大
- BM25 和 embedding 都错过了（罕见但可能）

**修复**：
- 加大 chunk 或 Small-to-Big
- 换 embedding 模型（实测）
- 加 HyDE（用假设答案匹配）
- 加 Multi-Query 扩召回

### 子类型 2b：精度差（不该召回的召回了）

**症状**：Context Precision 低

**根因**：
- chunk 太大，召回的 chunk 里大部分是噪音
- embedding 区分不开"主题相近但具体不同"的文档
- 没有 reranking，把不相关的排在了前面

**修复**：
- 缩小 chunk
- **加 reranking**（最有效，详见 [retrieval/reranking.md](./retrieval/reranking.md)）
- 加 metadata 过滤（"只查 2024 年的"）

### 子类型 2c：跨语言 / 跨表达
中文 query 召回英文文档失败 → 用多语言 embedding 或翻译。

---

## ③ 上下文整合失败

### 症状
- 召回的 context 明明有答案
- LLM 还是答错或编造

### 根因
**信息太多，LLM 注意力散了（Lost in the Middle）**。

论文证明：长 context 中，**中间**的信息容易被忽略。

```
context = [chunk_开头] + [chunk_中间] + [chunk_结尾]
                          ↑
                    信息容易在这里丢失
```

### 修复方向
- 减少召回 chunk 数量（top-3 到 top-5 通常比 top-20 好）
- 重要信息放 context 头部或尾部
- **reranking 后**按相关性排序，让最重要的在头部
- 用 Long Context 模型（Claude 200k 等）但成本高

### 子类型 3b：信息冲突
context 里有多份矛盾文档（如 2023 vs 2024 政策）。
**LLM 不知道信哪份**。

**修复**：
- 加 metadata（文档日期、版本）
- prompt 里显式说"优先用日期新的"
- 召回时按时间过滤

---

## ④ 生成失败

### 子类型 4a：编造（Hallucination）

**症状**：Faithfulness 低（RAGAS）

**根因**：
- prompt 没限制"只基于 context 回答"
- 模型能力不够
- context 太模糊，模型"脑补"

**修复**：
```
prompt 加：
1. "只基于下面 context 回答"
2. "context 没说就说不知道"
3. "每个事实必须引用 context 中的段落"
```
- 换更强模型
- 让模型输出"引用 → 答案"的映射

### 子类型 4b：啰嗦 / 跑题

**症状**：Answer Relevancy 低

**根因**：prompt 没限定回答风格。

**修复**：
- prompt 里明确"直接回答，不要解释背景"
- 给 few-shot 示例展示期望风格
- 加 Answer Relevancy 自动评估

### 子类型 4c：拒绝回答（应该回答但不答）

**症状**：AI 说"我不知道" 但 context 里有答案

**根因**：
- 安全策略过严
- prompt 写了"不确定就拒答"
- 模型保守

**修复**：调 prompt，平衡"诚实"和"有用"。

---

## ⑤ 任务完成失败

### 症状
- 单看回答没问题
- 用户却说"没用"

### 根因
**回答正确 ≠ 解决问题**。

```
用户问: "我下个月要休 5 天年假，怎么走流程？"
AI 答:   "请假流程是：提交 OA → 主管审批 → HR 备案"
                                       ← 任务没完成
应该:     给出 OA 链接、需要哪些附件、主管是谁
```

**修复**：
- 加 Function Calling，让 AI 能直接执行动作（开请假单、查主管）
- prompt 加"提供可操作的下一步"
- 在线反馈采集这类失败

---

## 诊断流程：从症状到根因

```
症状: 用户反馈答错
   │
   ├─ 答非所问？ ───────────────────→ ① 查询理解 / ⑤ 任务理解
   │
   ├─ 编造内容？ ───────────────────→ 检查 context:
   │                                  ├ context 有正确答案 → ④ 生成
   │                                  └ context 没有正确答案 → ② 检索
   │
   ├─ 召回的 context 全错？ ────────→ ② 检索（精度差）
   │
   ├─ 召回的 context 噪音多？ ──────→ ② 检索（精度差）/ 加 reranking
   │
   ├─ context 部分有，部分没有？ ───→ ② 检索（召回率） / chunk 太小
   │
   ├─ 多文档冲突？ ────────────────→ ③ 上下文整合
   │
   └─ 回答对但没用？ ───────────────→ ⑤ 任务完成
```

---

## FDE 的诊断工具箱

### 工具 1：Trace 可见
每次请求都记录：
- 原始 query
- 改写后的 query（如果有）
- 召回的 chunks（含分数、距离）
- 完整 prompt
- LLM 输出

（用 Langfuse / Phoenix，见 [../06_observability/](../06_observability/))

### 工具 2：失败 case 分类
建一个 Notion / DB 表：

| case | 症状 | 根因 | 修复 | 状态 |
|------|------|------|------|------|
| Q-001 | 编造年假 | context 没召回 | 换 embedding | ✅ |
| Q-002 | 答非所问 | query 太模糊 | 加 query rewrite | 🚧 |

### 工具 3：分桶评估
按维度分桶看 RAGAS：
- 按主题（HR / 财务 / 法律）
- 按难度（easy / hard）
- 按 query 长度
- 按用户类型

找出哪类问题失败最多 → 优先改。

### 工具 4：人工抽检
每周抽 20 个请求，人工判断：
- 检索准吗？
- 答案对吗？
- 用户满意吗？

**LLM-as-Judge 看不出的细微问题，人能看出**。

---

## 一个真实诊断案例

**症状**：my_com_rag 中，用户问"试用期工资"，系统答"试用期年假"。

**诊断**：
1. 看 trace → query = "试用期工资"
2. 看召回 → top-3 chunks 全是"试用期年假"
3. 看向量分数 → "试用期工资"和"试用期年假"embedding 距离很近
4. 根因 → embedding 把"试用期 X"主题聚一起，工资/年假区分不开

**修复**：
- 加 BM25（关键词匹配）
- 加 reranking（语义细判）
- 加 metadata 过滤（按 chunk 主题标签）

**验证**：
- 跑 RAGAS → Context Precision 从 0.6 提到 0.85

**闭环**：
- 这个 case 加入 hard negatives golden set
- 写入"failure case 文档"

---

## 反模式：把"幻觉"当万能借口

❌ "AI 就是会幻觉，没办法" → 没诊断根因
❌ "换个模型就好了" → 模型不是唯一变量
❌ "调一下 prompt" → 没指方向
❌ "加更多 context" → 信息过多反而更糟

**正确做法**：每次失败都走一遍诊断流程，定位根因，再改。

---

## 实战：给 my_com_rag 建失败档案

### Week 1：收集
- 把所有 👎 的 case 收集起来
- 人工诊断：属于 5 类失败中的哪类
- 加到 failure_cases.jsonl

### Week 2：分析
- 按失败类型分桶
- 找最大失败模式（如 50% 是检索失败）

### Week 3：定向修复
- 选最大失败模式，对症下药
- 修复后跑 RAGAS，量化提升

### Week 4：闭环
- 修复的 case 转为回归测试
- 写一篇博客："我修了 50 个 RAG 失败 case，学到了什么"

**这是 FDE 最值钱的实战经验**。

---

## 自测题

1. 用户说"答错了"——你怎么判断是检索问题还是生成问题？
2. Context Recall 低 + Context Precision 高，根因是什么？
3. 系统答得很啰嗦，加 more context 能解决吗？
4. "Lost in the Middle" 是什么？怎么应对？
5. 你怎么把一个失败 case 变成改进循环？

---

## 03_rag_deep 顶层小结

回到 4 篇顶层笔记的串联：

```
[分块策略]  ─── 决定 chunk 长什么样
     ↓
[Embedding] ─── 把 chunk / query 变向量
     ↓
[向量库]    ─── 存向量，做检索
     ↓
[失败模式]  ─── 系统坏了，按图索骥
```

下面进入 chunking/ 和 retrieval/ 的细节（实战代码 + 高级技巧）：

> 下一步：
> - [chunking/fixed_vs_recursive.md](./chunking/fixed_vs_recursive.md)
> - [retrieval/hybrid_search.md](./retrieval/hybrid_search.md)
> - [retrieval/reranking.md](./retrieval/reranking.md)（强烈推荐先读）
