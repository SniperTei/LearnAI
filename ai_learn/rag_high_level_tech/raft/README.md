# RAFT 技术学习

RAFT (Retrieval-Augmented Fine Tuning) - RAG 高级技术的学习和实现。

## 什么是 RAFT？

RAFT 是一种 RAG 高级技术，通过特殊的微调方式让模型学会：
- 从混合文档中识别相关信息
- 忽略干扰文档
- 生成带引用的答案

## 核心理念

```
传统 RAG:  检索 → 直接使用文档 → 生成答案
RAFT:      相关+干扰文档 → 训练模型识别 → 生成答案
```

## 文件说明

### raft_simple_demo.py
RAFT 的简化实现，包含：

1. **数据结构**
   - `Document`: 文档类
   - `RaftSample`: 训练样本类

2. **核心组件**
   - `RaftTrainer`: 训练器
   - `RaftInference`: 推理器
   - `RaftDataGenerator`: 训练数据生成器
   - `RaftEvaluator`: 评估器

3. **完整流程**
   - 创建训练样本
   - 准备数据集
   - 训练模型（演示流程）
   - 推理演示
   - 评估指标

### raft_training_data.json
示例训练数据，包含：
- 问答对
- 文档（相关+干扰）
- 答案和引用

## 快速开始

```bash
# 运行完整演示
python raft_simple_demo.py
```

## 输出示例

```
===============================================================================
RAFT (Retrieval-Augmented Fine Tuning) 简化版演示
===============================================================================

📚 准备文档库...
✅ 文档库包含 12 个文档

📝 创建训练样本...
✅ 创建样本: 什么是RAG？
✅ 创建样本: RAFT是什么？
✅ 创建样本: 什么是向量数据库？

🚀 开始训练
...
```

## 代码结构

### 数据生成
```python
# 创建训练样本
data_generator = RaftDataGenerator()
sample = data_generator.create_training_sample(question, documents)
```

### 训练
```python
trainer = RaftTrainer()
dataset = trainer.prepare_dataset(samples)
trainer.train(dataset, epochs=3)
```

### 推理
```python
inference = RaftInference()
result = inference.query(question, document_store)
```

### 评估
```python
evaluator = RaftEvaluator()
metrics = evaluator.evaluate_sample(prediction, gold_standard)
```

## 核心概念

### 1. 干扰文档（Distractor Documents）
训练时混入不相关文档，让模型学会识别：
- 随机干扰
- 困难负样本（语义相似但无关）
- 同领域不同主题

### 2. Oracle 模型
使用强模型（如 GPT-4）基于相关文档生成标准答案：
- 保证答案质量
- 提供引用

### 3. 评估指标
- **引用准确率**: 是否引用正确文档
- **干扰文档排除率**: 是否成功忽略干扰

## 与传统 RAG 的区别

| 维度 | Native RAG | RAFT |
|------|------------|------|
| 训练 | ❌ 不需要 | ✅ 需要微调 |
| 抗干扰 | ❌ 弱 | ✅ 强 |
| 引用 | ❌ 不保证 | ✅ 训练目标 |
| 成本 | ⭐ 低 | ⭐⭐⭐ 高 |

## 适用场景

✅ 适合：
- 检索质量较差的场景
- 需要精确引用（医学、法律）
- 有专门团队维护

❌ 不适合：
- 快速原型验证
- 预算有限的项目
- 检索质量已经很高

## 实际应用注意

这是一个**教育性简化实现**，用于理解概念。生产环境需要：

1. 真实的向量检索
   - 使用 sentence-transformers 或 OpenAI Embedding
   - 集成向量数据库（Faiss/Pinecone）

2. 实际的模型训练
   - 使用 transformers + PEFT
   - LoRA/QLoRA 微调

3. 更大的数据集
   - 使用 GPT-4 生成高质量训练数据
   - 数据增强（不同干扰组合）

4. 完整的评估
   - ROUGE/BLEU 评估答案质量
   - 语义相似度评估

## 学习建议

1. 先运行 `raft_simple_demo.py` 理解流程
2. 阅读代码中的注释了解实现细节
3. 尝试修改参数观察效果
4. 参考 `../rag/` 中的完整 RAG 实现

## 相关资源

- RAFT 论文: "RAFT: Adapting Language Models for Retrieval-Augmented Generation"
- LoRA 技术: 参数高效微调
- Faiss: 向量相似度搜索

## 作者

Claude Code Assistant
日期: 2026-01-27
