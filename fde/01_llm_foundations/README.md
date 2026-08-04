# 01 LLM Foundations

> 调 API 谁都会，懂底层才能在客户现场拍板。

## 学完应该能回答

- 同样一段话，为什么 token 数和我估的不一样？
- 温度 0.7 vs 0.2，到底什么变了？什么时候用高/低？
- 客户说"我要支持 100 万字"，你怎么回答？
- GPT-4o / Claude / Gemini / Qwen，**这个场景**该选谁？为什么？
- 上下文窗口扩到 200k，真的等于"能记住 200k"吗？

## 待写笔记

- [ ] `tokens_and_context.md` — tokenizer 是什么，为什么中文和英文 token 差很多
- [ ] `sampling_params.md` — temperature / top_p / top_k / repeat_penalty 的物理含义
- [ ] `context_window_reality.md` — 长上下文的真相：注意力衰减、KV cache、成本爆炸
- [ ] `model_selection.md` — 模型选型决策树（能力 / 成本 / 延迟 / 合规）
- [ ] `pricing_reality.md` — 一句话到底多少钱？token 成本心算

## 实战任务

- [ ] 用 tiktoken 给自己的项目算一次"真实 token 成本"
- [ ] 同一任务用 3 个模型跑，对比成本 / 延迟 / 质量
- [ ] 测试 200k 上下文下，模型在文档**中间**位置的准确率（"Lost in the Middle"）

## 参考资源

- Anthropic、OpenAI 各自 tokenizer 文档
- "Lost in the Middle" 论文（长上下文衰减）
- 各家模型 pricing 页面（每月更新）
- Chip Huyen《AI Engineering》第 2–3 章
