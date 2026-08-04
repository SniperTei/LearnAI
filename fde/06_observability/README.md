# 06 Observability

> Demo 上线那天不是终点，是**问题的开始**。
> 没有可观测性，你连"它好不好用"都不知道。

## 学完应该能回答

- 上线后用户反馈"AI 答得不好"，你怎么定位是哪一步的问题？
- 一次请求花了 5 秒，时间花在哪？
- 这个月花了 5 万 token 费用，谁用的？哪个功能用的？
- 用户点 👍 多还是 👎 多？👎 的 case 长什么样？
- 系统突然变差了，能 5 分钟内发现吗？

## 待写笔记

- [ ] `tracing.md` — Trace 每一次调用：Langfuse / Phoenix / LangSmith 选型
- [ ] `metrics.md` — 延迟 P50/P95/P99、token、成本、错误率
- [ ] `feedback_loops.md` — 用户反馈采集 → 数据闭环 → 模型/prompt 迭代
- [ ] `alerting.md` — 什么时候报警：质量下降 / 成本飙升 / 错误激增
- [ ] `dashboards.md` — 给工程团队看 vs 给客户看的不同 dashboard

## 实战任务

- [ ] 给 my_com_rag 接 Langfuse（开源版自部署）
- [ ] 实现 trace 链路：每个 prompt / RAG 召回 / 工具调用都可见
- [ ] 建一个"成本 dashboard"：按用户 / 功能 / 时间维度
- [ ] 实现反馈采集 UI，反馈自动进 Langfuse
- [ ] 写一个"线上问题溯源"复盘：从用户反馈 → 定位根因 → 修复

## 参考资源

- Langfuse 官方文档（开源，强烈推荐）
- Arize Phoenix
- LangSmith 文档
- OpenTelemetry for LLM
