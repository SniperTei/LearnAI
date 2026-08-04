# 02 Prompt Engineering

> Prompt 工程没过时，依然是 FDE 最高 ROI 的技能。
> 但这里学的不是套模板，是**知道为什么这么写就有效**。

## 学完应该能回答

- few-shot 给 3 个例子和给 10 个例子，效果差别在哪？
- Chain-of-Thought 为什么对一些任务有用，对另一些有害？
- 客户给的 prompt 经常失效，你怎么系统化排查？
- 怎么稳定让模型输出 JSON？
- 提示注入（Prompt Injection）是什么？我能防到什么程度？

## 待写笔记

### `fundamentals/`
- [ ] `few_shot.md` — 例子的数量 / 顺序 / 多样性的影响
- [ ] `chain_of_thought.md` — CoT 的边界：什么时候有用，什么时候画蛇添足
- [ ] `role_and_persona.md` — "你是 XX 专家"到底改了什么
- [ ] `instruction_order.md` — 提示词里信息顺序的影响

### 顶层
- [ ] `structured_output.md` — JSON mode / tool use / 结构化输出的取舍
- [ ] `prompt_security.md` — 提示注入、越狱、防护策略
- [ ] `prompt_versioning.md` — Prompt 也是代码，需要版本管理

## `exercises/`

- [ ] `01_extract_structured.md` — 同一任务，3 种 prompt 写法对比
- [ ] `02.jailbreak_lab.md` — 自己尝试越狱自己的系统，再打补丁
- [ ] `03.long_context_prompt.md` — 长上下文下 prompt 放头/中/尾的差异

## 参考资源

- Anthropic 官方 Prompt Engineering Guide
- OpenAI Cookbook：structured output 章节
- Lilian Weng 博客：Prompt Engineering
- "Prompt Injection Attacks" 综述
