# 如何从零构建 Golden Set

> Golden Set（黄金集）是所有评估的地基。
> 没有它，你前面学的所有评估方法都白搭。

---

## Golden Set 是什么

一组**人工标注的、可信的、有代表性的**问答样本：

```jsonl
{"question": "公司年假多少天？", "ground_truth": "入职 15 天，工龄 5 年以上 20 天", "context": "...", "tags": ["hr", "factoid"]}
{"question": "请帮我总结这份合同的关键风险", "ground_truth": "...", "context": "...", "tags": ["legal", "summarization"]}
```

每条样本告诉评估系统：
- 应该问什么（question）
- 正确答案是什么（ground_truth / reference answer）
- （可选）相关上下文（用于 retrieval 评估）
- （可选）难度、分类标签

---

## Golden Set 的两大用途

### 用途 1：评估检索（Retrieval）
- 给定 question
- 检查系统召回的 context 是否包含 ground_truth 所需的信息
- 指标：Context Precision / Recall

### 用途 2：评估生成（Generation）
- 给定 question
- 检查系统生成的 answer 是否接近 ground_truth
- 指标：Faithfulness / Answer Relevancy / 准确率

**好的 Golden Set 同时支持两者**。

---

## 一份合格的 Golden Set 的 5 个标准

### 1. 有规模
- 最少 50 条（PoC）
- 推荐 100–300 条（生产）
- 关键场景 1000+ 条（核心产品）

**经验**：少于 30 条 → 任何指标都没统计意义。

### 2. 有代表性
**不要**只挑你会答的。覆盖：
- 简单 / 中等 / 困难（按 3:5:2 分布）
- 不同主题（按业务重要性加权）
- 不同问法（同义改写、口语化、错别字）
- 边界场景（超出范围的问题、模糊问题）

### 3. 有"金标答案"
- 准确的 ground_truth（不是"差不多对"）
- 最好是**多个标注员一致同意**的答案
- 客观题有唯一答案；主观题有"参考答案 + 评分 rubric"

### 4. 有标签 / 元数据
```jsonl
{
  "question": "...",
  "ground_truth": "...",
  "difficulty": "hard",
  "category": "policy",
  "language": "zh",
  "source": "real_user_query",  // 真实用户问的还是造的
  "tags": ["multi-hop", "needs-context"]
}
```

便于：分桶分析、按类别看质量。

### 5. 有版本管理
Golden Set 会演化——加新 case、修正错的、删除过时的。
- 用 Git 管理
- 每次变更记录原因
- 评估时记录用的哪个版本

---

## 从哪找问题（Question Source）

### 来源 1：真实用户 query（最强）
- 从日志里挖
- 去重、清洗、分类
- 这是**最值钱**的样本

**比例**：成熟系统中应占 60% 以上。

### 来源 2：业务专家出题
- 让客户/PM/SME 写
- 覆盖他们关心的场景
- 通常质量高，但有"出题人偏差"（出题人潜意识挑简单的）

### 来源 3：从知识库反推
- 看一份文档，自己想"用户可能问什么"
- 适合冷启动

### 来源 4：合成（LLM 生成）
- 用 LLM 看知识库，生成可能的 query
- 量大，但有"AI 偏差"（生成的问题不像真人问的）

### 来源 5：竞品 / 公开数据集
- 同领域的公开数据集
- 充实早期数据

---

## 标准流程：从 0 到 100 条

### Phase 1：冷启动（30 条，1 天）

1. **选 30 个核心场景**
   - 列出业务的 top 10 痛点问题
   - 每个痛点延伸 3 个变体

2. **每个场景写 1 条样本**
   ```jsonl
   {
     "question": "入职第一年的年假是多少？",
     "ground_truth": "15 天",
     "context": "员工手册 v3.2 第 4 章",
     "category": "hr",
     "difficulty": "easy"
   }
   ```

3. **自己跑一遍系统**
   - 哪些答对？哪些答错？
   - 错的标"hard"

### Phase 2：扩展（100 条，3–5 天）

4. **从真实日志补 50 条**
   - 系统上线（哪怕小范围）后，看用户问什么
   - 把高频问题加入

5. **加边界 case（10–20 条）**
   - 系统应该拒答的（"本公司不处理 X 业务"）
   - 模糊的（"年假相关的事"）
   - 跨主题的（"年假和病假的区别"）

6. **加对抗 case（5–10 条）**
   - 提示注入尝试
   - 越狱尝试
   - 错误前提（"公司年假 30 天对吗？" → 应该纠正）

### Phase 3：质量保证

7. **双人标注 + 一致性检查**
   - 每条样本 2 人独立标
   - 不一致的讨论，改 rubric

8. **专家审核**
   - 让业务专家审 20 条
   - 修正 ground truth

9. **打标签 / 元数据**
   - 类别、难度、tag

### Phase 4：上线 + 维护

10. **第一次基线评估**
    - 用 RAGAS / LLM-as-Judge 跑
    - 记录基线分数

11. **持续维护**
    - 每周 / 每月加 5–10 条新 case
    - 从在线 👎 反馈里挑（关键！）
    - 删除过时的（如政策变化）

---

## 反模式

❌ **只造简单问题** → 系统在简单题上 100%，一上线全崩
❌ **AI 全自动生成** → LLM 生成的问题不像真人，评估失真
❌ **没有 ground truth** → 只能评"流畅度"，不能评"对错"
❌ **没有标签** → 无法分桶分析，找不到失败模式
❌ **造完一次不动** → 系统改进后，旧集子测不出新问题
❌ **样本数太少就拍板** → 30 条得到"准确率 90%"，可能根本没意义

---

## 一个最小可用的 Golden Set 模板

`golden_set.jsonl`：
```jsonl
{"id": "q001", "question": "公司年假多少天？", "ground_truth": "入职 15 天，工龄 5 年以上 20 天", "context_ref": "handbook_v3_p4", "category": "hr", "difficulty": "easy", "source": "real", "tags": ["factoid"]}
{"id": "q002", "question": "请对比下我们两款主力产品的优缺点", "ground_truth": "产品A: 优势... 劣势...；产品B: 优势... 劣势...", "context_ref": "product_specs_v2", "category": "product", "difficulty": "hard", "source": "real", "tags": ["comparison", "multi-source"]}
{"id": "q003", "question": "我老板说要给我加薪，你能帮我改合同吗", "ground_truth": "（拒答：AI 不提供法律文件修改）", "category": "safety", "difficulty": "edge", "source": "real", "tags": ["refusal", "out-of-scope"]}
```

字段说明：
- `id`：唯一 ID，便于回归测试追踪
- `question`：用户问法
- `ground_truth`：金标答案（或"应该拒答"）
- `context_ref`：相关文档（用于 retrieval 评估）
- `category`：业务分类
- `difficulty`：easy / medium / hard / edge
- `source`：real / expert / synthetic
- `tags`：自由标签

---

## 维护策略

### 每周（轻量）
- 加 3–5 条从在线反馈发现的失败 case
- 删 1–2 条已过时的

### 每月（中量）
- 添加 10–20 条新场景
- 重审"hard"分类，看是否还 hard
- 更新因业务变化的 ground_truth

### 每季度（大量）
- 全量 review
- 重新算基线
- 调整分布（避免某些类别过度膨胀）

---

## 实战：给 my_com_rag 造 100 条

### Week 1
- Day 1: 从现有知识库找 20 个核心问题（手写）
- Day 2: 自己回答 + 让系统回答 + 标记差异
- Day 3-5: 找 5 个朋友/同事，让他们"如果你是用户会问什么" → 30 条

### Week 2
- 把 my_com_rag 部署到测试环境，给 3-5 个用户试用
- 收集真实 query → 整理成 30 条

### Week 3
- 加 20 条边界 / 对抗 case
- 双人 review

### Week 4
- 跑第一次基线评估
- 写报告

**结果**：100 条带标签的高质量 Golden Set，**这是你 FDE 简历上的硬通货**。

---

## 自测题

1. 30 条样本得到的"准确率 85%"，能信吗？
2. 你的 Golden Set 全是 LLM 生成的，会有什么问题？
3. 怎么保证 Golden Set 的 ground_truth 是对的？
4. 系统升级后，老 Golden Set 还有用吗？
5. 没钱请标注员，怎么造 100 条像样的？

---

> 下一步：[synthetic_data.md](./synthetic_data.md) — 怎么用 LLM 造数据（又不踩坑）
