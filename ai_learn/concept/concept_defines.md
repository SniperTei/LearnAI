## 概念、定义

#### PyTorch

由 Meta（Facebook）开发的深度学习框架。采用**动态计算图**，调试方便，语法接近 Python 原生，是学术界最流行的深度学习框架。支持 GPU 加速、自动求导（Autograd），广泛用于研究和生产。

### 模型训练工具

#### TensorFlow

由 Google 开发的深度学习框架。采用**静态计算图**（TF2.x 也支持动态图），生态完整，从训练到部署一条龙。适合大规模工业生产环境，内置 TensorBoard 可视化工具、TF Serving 部署服务等。

#### Keras

一个高级神经网络 API，最初独立开发，现为 TensorFlow 的官方高级接口。**主打简洁易用**，几行代码就能搭建一个神经网络，非常适合初学者和快速原型开发。

#### Hugging Face

AI 领域的 "GitHub"。提供：
- **Models Hub**：海量预训练模型仓库（NLP、图像、音频等）
- **Transformers 库**：统一接口调用各种模型
- **Datasets**：常用数据集一键加载
- **Spaces**：在线演示和部署模型

是当前获取和使用预训练模型的首选平台。

#### Transformers

一种**神经网络架构**，也是 Hugging Face 的核心库名。

作为架构：2017 年 Google 论文《Attention is All You Need》提出，核心机制是**自注意力（Self-Attention）**，能并行处理序列数据。GPT、BERT、LLaMA 等所有主流大语言模型都基于此架构。

作为库：Hugging Face 提供的 Python 库，统一接口加载和使用各种 Transformer 模型。

### 模型部署工具

#### TensorFlow Lite

TensorFlow 的**轻量级版本**，专为移动端（Android/iOS）和嵌入式设备设计。将训练好的模型转换压缩后，在手机、树莓派等边缘设备上高效推理。

### ONNX Runtime

ONNX（Open Neural Network Exchange）是微软主导的**开放模型格式**，让模型可以在不同框架间互转。ONNX Runtime 是其推理引擎，支持跨平台（CPU/GPU/Edge）高性能运行模型，兼容 PyTorch、TensorFlow 等训练的模型。

#### TorchServe

PyTorch 官方的**模型部署服务工具**。由 AWS 和 Meta 合作开发，可以把训练好的 PyTorch 模型打包成 REST API 或 gRPC 服务，支持多模型管理、批处理、日志监控等生产级特性。

### 应用开发框架

#### LangChain

目前最流行的 **LLM 应用开发框架**。核心思想是用"链（Chain）"把 LLM 与外部工具串联起来。提供：
- **Chain**：多步骤任务编排
- **Agent**：让 LLM 自主决定调用哪些工具
- **RAG**：检索增强生成，让 LLM 结合外部知识
- 丰富的集成：向量数据库、API、工具等

适合构建聊天机器人、知识库问答、AI Agent 等应用。

#### LlamaIndex

专注**数据索引和检索**的 LLM 框架。核心功能是将你的私有数据（文档、数据库、API 等）构建成索引，让 LLM 能高效查询。与 LangChain 互补——LlamaIndex 擅长数据接入和检索，LangChain 擅长编排和 Agent 逻辑。

#### AutoGPT

一个**自主 AI Agent** 项目。给定一个目标，AutoGPT 会自主分解任务、搜索信息、编写代码、执行操作，循环迭代直到完成目标。是最早爆火的 Agent 概念验证项目，展示了 LLM 自主规划能力的可能性。

### 机器人开发框架

#### ROS

**Robot Operating System（机器人操作系统）**，不是真正的 OS，而是一套机器人开发的**中间件框架**。提供：
- 节点间通信机制（话题 Topic、服务 Service）
- 丰富的机器人工具包（导航、SLAM、感知、控制）
- 强大的开源生态

ROS2 是新一代版本，支持实时性和分布式场景。在 AI + 机器人领域，ROS 是将 AI 模型部署到实体机器人上的标准平台。
