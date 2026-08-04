# FDE 6 个月学习路径（个性化版）

> 基于你已有的 `LearnAI/ai_learn/` 基础定制。
> 不是从零开始，是把"会用"深化为"懂 why"。

---

## 起点：你已经有什么

盘点 `ai_learn/` 已有的能力：

| 能力 | 已掌握（会用） | 待深化（懂 why） |
|------|----------------|-------------------|
| RAG 完整链路 | ✅ `my_com_rag` | chunking 策略对比、reranking、评估 |
| Function Calling / Agent | ✅ 入门 | 多步编排、错误恢复、何时不用 Agent |
| LangChain | ✅ 入门 | LangGraph、何时自研轻量编排 |
| Fine-tuning | ✅ 跑通过 | 何时**不**微调，LoRA/QLoRA 决策 |
| Text2SQL 多版本 | ✅ 6 个迭代 | 评估方法、失败模式分析 |
| Docker 部署 | ✅ 会用 | CI/CD、监控、降级 |
| Embedding | ✅ 会调 | 不同 embedding 模型差异、维度选择 |
| 评估 / 可观测性 | ❌ 几乎空白 | **最大短板** |
| 客户沟通 / 业务思维 | ❌ 几乎空白 | **护城河** |

**结论**：技术广度够，缺**深度 + 业务感 + 评估**。

---

## 总览：6 个月路线图

```
Month 1   ──── 深化已有项目（my_com_rag 升级）
Month 2   ──── 补齐评估 + 可观测性（最大短板）
Month 3   ──── 生产工程化（Guardrails / 成本 / 路由）
Month 4   ──── 第一个垂直行业 case study
Month 5   ──── Agent 深入 + LangGraph
Month 6   ──── 作品集 + 求职准备
```

---

## Month 1：深化 my_com_rag（2–4 周）

**目标**：把已有项目改造成"作品级"，作为后续所有模块的练手靶子。

### 学习产出
- [ ] 用 RAGAS 给项目跑一次完整评估，得到量化指标
- [ ] 接入 Langfuse，每次调用都有 trace
- [ ] 给 RAG 加 reranking（如 bge-reranker），对比前后效果
- [ ] 重构 agent_manager.py，支持工具并发 + 失败重试
- [ ] 写一篇复盘："我给 my_com_rag 加了评估和监控，发现了什么"

### 配套学习（fde/ 内）
- `03_rag_deep/chunking/` — 你的 chunking 策略对吗？
- `03_rag_deep/retrieval/` — reranking 为什么有效
- `05_evaluation/offline/` — RAGAS 怎么用
- `06_observability/tracing.md` — Langfuse 实战

### 验收标准
能向陌生人讲清楚：**"这个项目上线后，回答准确率从 X 提到 Y，成本下降 Z%，靠的是 ABC 三个改动"**。

---

## Month 2：评估方法 + 可观测性（3–4 周）

**目标**：补齐最大短板。FDE 没有评估能力，等于没有落地能力。

### 学习产出
- [ ] 给 my_com_rag 构建一个 50–100 条的 golden set
- [ ] 跑 RAGAS 4 大指标（faithfulness / answer relevancy / context precision / recall）
- [ ] 实现 LLM-as-Judge pipeline，对比人类标注一致性
- [ ] 设计在线反馈采集（用户点 👍👎 + 修正文本）
- [ ] 实现 A/B 测试框架（哪怕手动切流也行）

### 配套学习
- `05_evaluation/why_eval.md` — 没评估就没 AI 产品
- `05_evaluation/offline/`、`online/`、`golden_set_recipes/`
- `06_observability/metrics.md`、`feedback_loops.md`

### 验收标准
有一份**"my_com_rag 质量报告"**：4 个 RAGAS 指标 + 在线用户反馈率 + 改进路线图。

---

## Month 3：生产工程化（3–4 周）

**目标**：从"能跑"到"敢上"。

### 学习产出
- [ ] 实现输入/输出 Guardrails（提示注入防护 + 敏感信息过滤）
- [ ] 加缓存层（语义缓存 or 精确缓存），测出成本节省
- [ ] 实现模型路由（简单问题走 Haiku，难题走 Opus/Sonnet）
- [ ] 部署到云（你已经有 Docker，加域名 + HTTPS + CI）
- [ ] 压测：延迟 P50/P95/P99 + 成本 / 1k 请求

### 配套学习
- `07_production/guardrails.md`、`caching_and_routing.md`、`cost_optimization.md`、`deployment_patterns.md`
- `01_llm_foundations/model_selection.md`

### 验收标准
一份压测 + 成本报告：**"系统在 100 QPS 下 P95 1.5s，单次请求平均成本 ¥0.03"**。

---

## Month 4：第一个垂直 case study（4 周）

**目标**：把所有能力整合到**一个完整的客户场景**里。

### 选一个行业
推荐（按上手难度）：
- **法律**：合同审查 / 法律检索（数据公开、痛点清晰）
- **教育**：个性化学习助手（用户好找、反馈快）
- **财务**：报销审核 / 财报分析（业务价值明确）
- **HR**：简历筛选 / 内部问答（数据相对好造）

### 完整走一遍
1. **需求假设**：写 1 页纸（业务问题、用户画像、ROI 假设）
2. **数据准备**：找 / 造数据，构建 golden set
3. **方案设计**：技术选型 + 为什么这么选
4. **实现**：复用你前面所有的能力
5. **评估**：用 Month 2 的 pipeline 跑指标
6. **部署 + Demo**：能给任何人演示
7. **复盘**：1 篇博客

### 配套学习
- `08_customer_skills/`（边做边学）
- `projects/README.md` — 项目模板

### 验收标准
有一个**完整可演示**的项目 + 一篇 1500 字以上的 case study 博客。

---

## Month 5：Agent 深入 + 扩展技能（3–4 周）

**目标**：补 Agent 这块的深度，并扩展边界。

### 学习产出
- [ ] 学 LangGraph，用 my_com_rag 的 Agent 部分重写
- [ ] 实现一个 ReAct / Plan-Execute 模式的多步 Agent
- [ ] 用 LoRA 微调一个小模型（如 Qwen 7B），记录**什么时候微调有效**
- [ ] 入门 MCP 协议（Anthropic 的标准），跑通一个 MCP server
- [ ] 写博客："Agent 不是万能的——这 3 个场景我建议你别用"

### 配套学习
- `04_agents/`（全部）
- `01_llm_foundations/`（token、context window 真相）

### 验收标准
能清楚说出：**"什么时候用 Agent，什么时候用 RAG，什么时候用 fine-tune，什么时候啥都不用"**。

---

## Month 6：作品集 + 求职准备（4 周）

**目标**：把 5 个月的产出变成可以拿 offer 的材料。

### 学习产出
- [ ] GitHub 仓库 README 全部重写：每个项目都讲清楚业务价值
- [ ] 整理 2–3 篇深度博客（你的真实踩坑记录）
- [ ] 1 份英文简历：突出 case study 和量化结果
- [ ] 准备 FDE 风格的 case interview（不是 LeetCode）
- [ ] 找 3 家目标公司，研究他们的客户 case

### 验收标准
- 能在 5 分钟内向 recruiter 讲清楚"我做过什么、解决了什么业务问题"
- 至少投递 5 个 FDE 岗位

---

## 学习节奏（每周）

```
周一 晚上 (2h)    读理论 / 文档
周三 晚上 (2h)    写代码 / 实验
周五 晚上 (2h)    复盘 / 写笔记
周末 (4–6h)       深度项目时间
```

每周 10 小时左右，6 个月 ≈ 240 小时，**足够从入门到求职**。

---

## 评估自己的进步（每月自检）

每月最后一天，回答：

1. 我这个月做出的最骄傲的东西是什么？
2. 我能用非技术语言讲清楚它的业务价值吗？
3. 我有量化指标证明它有效吗？
4. 我学到的最重要的非技术经验是什么？
5. 下个月如果不做计划，我最想先做什么？

> 这 5 个问题比任何技术面试都更能检验 FDE 成长。

---

## 不要做的事

- ❌ 一周学 3 个新技术（吃不下）
- ❌ 不写复盘只写代码（FDE 的价值一半在表达）
- ❌ 跳过评估直接做 Agent（地基没打好）
- ❌ 只学不用（每个知识点都要落在 my_com_rag 或新项目上）
- ❌ 比较自己和别人（FDE 是个人化路径）

---

## 给 6 个月后的你

如果坚持下来，你应该能：
- ✅ 拿出 1 个有量化指标的 RAG+Agent 系统（my_com_rag 升级版）
- ✅ 1 个完整垂直行业 case study
- ✅ 2–3 篇被同行收藏的博客
- ✅ 一份能进面试的英文简历
- ✅ 清楚知道 FDE 是不是你真的想做的方向

更重要的是：**你会用 FDE 的眼光看世界**——看到一个产品就能拆解它，看到一个需求就能找到真问题。

---

> 现在开始：打开 [../03_rag_deep/](../03_rag_deep/) 或 [../05_evaluation/](../05_evaluation/) 起步。
> 推荐：**先去 `05_evaluation/`**——这是你最大的短板，也是 FDE 最核心的能力。
