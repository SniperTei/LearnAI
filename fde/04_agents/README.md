# 04 Agents

> 2024–2026 最被滥用的概念之一。
> FDE 必须能清楚说出"什么时候用 Agent，什么时候**不**用"。

## 学完应该能回答

- Agent 和 Function Calling 是什么关系？
- ReAct、Plan-Execute、Reflexion 这些模式本质区别是什么？
- 客户的需求**真的**需要 Agent 吗？还是 RAG 就够了？
- Agent 失败了怎么办？错误恢复怎么设计？
- LangGraph / 自研编排，选哪个？为什么？

## 待写笔记

- [ ] `tool_use_theory.md` — function calling 底层：模型是怎么"决定"调哪个函数的
- [ ] `orchestration_patterns.md` — ReAct / Plan-Execute / Reflexion / Multi-Agent 对比
- [ ] `error_recovery.md` — 工具调用失败、循环、超时的处理
- [ ] `state_management.md` — Agent 的"记忆"该怎么做
- [ ] `when_not_agent.md` — **最重要的笔记**：什么场景不要用 Agent
- [ ] `langgraph_vs_custom.md` — 框架选型
- [ ] `mcp_protocol.md` — Anthropic MCP 协议入门

## 实战任务

- [ ] 用 LangGraph 重写 my_com_rag 的 agent_manager.py
- [ ] 实现一个 Plan-Execute 模式的 Agent（自己写编排，不用框架）
- [ ] 给 Agent 加错误恢复：工具失败 → 重试 / 降级 / 人工接管
- [ ] 找一个**不该用 Agent** 的场景，写出不用 Agent 的方案

## 参考资源

- Lilian Weng："LLM Powered Autonomous Agents"
- LangGraph 官方教程
- Anthropic："Building Effective Agents"（必读）
- Harrison Miller 的 multi-agent patterns
