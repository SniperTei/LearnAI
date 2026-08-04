# 07 Production

> 从"能跑"到"敢上"的最后一公里。
> Demo 丑没关系，生产**必须**稳。

## 学完应该能回答

- 用户输入恶意 prompt 想窃取 system prompt，怎么防？
- OpenAI / Anthropic API 挂了，你的系统还能工作吗？
- 100 QPS 下你的系统延迟是多少？成本是多少？
- 怎么用便宜模型处理 80% 的简单请求，贵模型只处理难题？
- 客户要在内网部署，怎么办？

## 待写笔记

- [ ] `guardrails.md` — 输入过滤 / 输出过滤 / 敏感信息脱敏
- [ ] `prompt_injection_defense.md` — 防御策略（没有银弹）
- [ ] `caching_and_routing.md` — 语义缓存 / 模型路由
- [ ] `cost_optimization.md` — 降成本的 7 个杠杆
- [ ] `rate_limiting.md` — 限流、降级、熔断
- [ ] `streaming.md` — 流式输出的工程挑战
- [ ] `deployment_patterns.md` — 多区域 / 内网 / 私有化部署
- [ ] `ci_cd_for_llm.md` — LLM 应用的 CI/CD：测试什么、怎么测

## 实战任务

- [ ] 给 my_com_rag 加输入/输出 Guardrails（可用 guardrails-ai 库）
- [ ] 实现语义缓存（相同或相近问题命中缓存）
- [ ] 实现模型路由：Haiku 处理简单，Sonnet 处理复杂
- [ ] 压测：用 locust 跑 100 QPS，给出 P50/P95/P99 报告
- [ ] 部署到云（任意），域名 + HTTPS + 基础告警
- [ ] 实现一次"模型厂商故障演练"：手动停 API，看系统表现

## 参考资源

- Guardrails AI 文档
- LiteLLM（模型路由 / 统一接口）
- Redis 语义缓存方案
- "Productionizing LLMs" by Chip Huyen
