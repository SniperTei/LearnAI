# FDE 学习手册（Forward Deployed Engineer）

> 这是我从"会调 API"走向"能在客户现场交付 AI 系统"的系统化学习笔记。
> 不是为了追新技术，而是为了**真正理解每一个选择的 why**。

---

## 🚦 如果你刚来，先读这一段

我之前的 `ai_learn/` 里写了一堆代码（RAG、Agent、Docker、Streamlit 大屏），
但**坦白说大部分是 AI 写的，我没真正懂**。

所以我先暂停"深化"，从基础重来一遍。

**两个入口，按你的真实水平选：**

### 🔰 新手入口（推荐）
[**00_start_here/**](./00_start_here/) — 新手村，5 个动手练习，从纯 HTTP 调 API 开始。
**如果你和我一样需要从基础补，从这里开始，不要跳过。**

### 🎓 进阶入口
[**00_foundation/**](./00_foundation/) — FDE 认知 + 6 个月学习路径。
**只有当你能独立解释 LLM API / token / 流式 / 多轮对话，才从这里开始。**

> 判断方法：你能用 3 句话讲清"为什么 LLM 没有记忆，但 ChatGPT 能多轮对话"吗？
> 答得出 → 进 [00_foundation](./00_foundation/)
> 答不出 → 进 [00_start_here](./00_start_here/)

---

## 这份手册的定位

把"能跑起来"的代码，变成"我能讲清楚每个为什么"的能力。

- **不是**再写一遍 demo
- **是**把已有知识深化成可以放在简历上、可以在面试中讲清楚、可以在客户现场拍板的能力

---

## 学习地图

| # | 模块 | 核心问题 | 状态 |
|---|------|----------|------|
| 🔰 | [**Start Here**](./00_start_here/) | **新手村：从裸 HTTP 调 API 开始** | [ ] ⭐ 先做这个 |
| 00 | [Foundation](./00_foundation/) | FDE 到底是什么？我应该怎么学？ | [ ] |
| 01 | [LLM Foundations](./01_llm_foundations/) | 模型底层到底在做什么？ | [ ] |
| 02 | [Prompt Engineering](./02_prompt_engineering/) | 为什么这么写 prompt 就有效？ | [ ] |
| 03 | [RAG Deep](./03_rag_deep/) | RAG 每个环节的 trade-off 是什么？ | [ ] |
| 04 | [Agents](./04_agents/) | 什么时候用 Agent，什么时候不要？ | [ ] |
| 05 | [Evaluation](./05_evaluation/) | 没有评估，就没有 AI 产品 | [ ] |
| 06 | [Observability](./06_observability/) | 上线之后，怎么知道它还好用？ | [ ] |
| 07 | [Production](./07_production/) | 把 demo 变成 7×24 服务 | [ ] |
| 08 | [Customer Skills](./08_customer_skills/) | FDE 真正的护城河 | [ ] |
| P  | [Projects](./projects/) | 完整 case study 实战 | [ ] |

---

## 学习节奏建议

- **每周 1 个子主题**，不要贪多
- **每个模块必须有产出**（笔记 / 代码 / 复盘）
- **每个模块结束回答 3 个问题**：
  1. 我能用人话向非技术人讲明白吗？
  2. 我能列出 3 个 trade-off 吗？
  3. 我能在 my_com_rag 或新项目里用上吗？

---

## 推荐优先级（基于你现有基础）

0. **🔴 必须先做** → `00_start_here/`（5 个动手练习，建立 LLM 真实理解）
1. **再打地基** → `00_foundation/`
2. **补最大短板** → `05_evaluation/`（你目前最缺的能力）
3. **深化最常用** → `03_rag_deep/`（你 my_com_rag 用得到，立刻能验证）
4. **生产化必备** → `06_observability/` + `07_production/`
5. **其余按需补** → Prompt / Agent / Customer Skills 并行
6. **最终落地** → `projects/`

---

## 与已有目录的关系

| 已有目录 | 在 fde 中的对应深化 |
|----------|---------------------|
| `ai_learn/my_com_rag/` | `03_rag_deep/`、`05_evaluation/`、`06_observability/` 的练手靶子 |
| `ai_learn/function_calling/` | `04_agents/` |
| `ai_learn/langchain/`、`langchain_a/` | `04_agents/orchestration_patterns.md` |
| `ai_learn/fine_tuning/` | `01_llm_foundations/model_selection.md`（何时微调） |
| `ai_learn/embedding/` | `03_rag_deep/embedding_theory.md` |
| `p_lesson/` | 不深化，作为基础练习存档 |

---

## 进度日志

> 每完成一个模块，在这里记一行：日期 / 模块 / 一句话收获

- 2026-07-24 — 开始 FDE 学习规划

---

> 下一步：打开 [00_foundation/](./00_foundation/) 开始。
