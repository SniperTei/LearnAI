# 05 Evaluation

> **FDE 最大短板、最高 ROI 的模块。**
> 没有评估，就没有 AI 产品——只有 demo。

## 学完应该能回答

- 客户问"你的 AI 好不好用"，你怎么用数据回答？
- RAGAS 四大指标到底在测什么？什么时候它们会骗你？
- LLM-as-Judge 什么时候靠谱，什么时候不靠谱？
- 上线后用户点 👍👎，怎么用这个数据改进系统？
- 没有 ground truth，怎么造一个？

## 待写笔记

- [ ] `why_eval.md` — 为什么评估是 FDE 的核心能力
- [ ] `eval_taxonomy.md` — 离线 / 在线 / 人在环 三类评估

### `offline/`
- [ ] `ragas.md` — RAGAS 实战 + 4 指标解读
- [ ] `llm_as_judge.md` — 用什么模型当 judge、prompt 怎么写、偏差如何控制
- [ ] `pairwise_vs_scoring.md` — 两两对比 vs 打分，哪个更稳
- [ ] `human_alignment.md` — 让 LLM 评委对齐人类标注

### `online/`
- [ ] `user_feedback.md` — 隐式 / 显式反馈采集
- [ ] `ab_testing.md` — AI 产品的 A/B 测试陷阱
- [ ] `adoption_metrics.md` — 采纳率 / 接管率 / 留存

### `golden_set_recipes/`
- [ ] `how_to_build.md` — 从 0 造 50–100 条 golden set 的方法
- [ ] `synthetic_data.md` — 用 LLM 生成评测集
- [ ] `hard_negatives.md` — 故意造难的 case

## 实战任务（在 my_com_rag 上做）

- [ ] 造一份 100 条的 golden set（含 easy/medium/hard）
- [ ] 跑 RAGAS 4 指标，得到基线分数
- [ ] 加 reranking 后再跑，量化提升
- [ ] 实现 LLM-as-Judge pipeline，和 RAGAS 对比一致性
- [ ] 设计在线反馈采集（前端 👍👎 + 修正文本）

## 参考资源

- RAGAS 官方文档
- Langfuse / Phoenix 文档
- Hamel Husain 的评估博客（必读）
- Eugene Yan："Evaluating LLM Applications"
