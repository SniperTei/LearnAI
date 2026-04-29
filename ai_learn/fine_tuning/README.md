# Fine-Tuning 学习路径

## 📚 学习目标
掌握大模型微调的核心概念、方法和实践技能

## 🎯 学习路线

### 第一阶段：基础概念
- [ ] 什么是Fine-tuning
- [ ] 预训练 vs 微调
- [ ] 什么时候需要微调
- [ ] 微调的类型

### 第二阶段：微调方法
- [ ] Full Fine-tuning
- [ ] LoRA (Low-Rank Adaptation)
- [ ] QLoRA (Quantized LoRA)
- [ ] Prompt Tuning
- [ ] Prefix Tuning

### 第三阶段：数据准备
- [ ] 数据收集
- [ ] 数据清洗
- [ ] 数据格式化
- [ ] 数据质量评估

### 第四阶段：实战项目
- [ ] 文本分类微调
- [ ] 指令微调 (Instruction Tuning)
- [ ] 对话模型微调
- [ ] 领域适配微调

## 📁 目录结构
```
fine_tuning/
├── basics/                  # 基础概念和理论
│   ├── 01-什么是Fine-tuning.md
│   ├── 02-微调类型对比.md
│   └── 03-学习路径指南.md
├── data/                    # 数据准备
│   └── prepare_domain_data.py  # 领域数据生成脚本
├── practical/               # 实战代码
│   ├── 01-QLoRA入门实战.md
│   ├── quick_start.py       # 快速开始示例
│   └── domain_finetuning.py # 特定领域微调
├── methods/                 # 各种微调方法详解
├── notebooks/               # Jupyter notebooks
├── requirements.txt         # 环境依赖
├── USAGE_GUIDE.md           # 详细使用指南
└── README.md                # 本文件
```

## 🚀 快速开始

### 方式1: 基础快速体验

```bash
# 安装依赖
pip install -r requirements.txt

# 运行基础示例（10条简单数据）
python practical/quick_start.py
```

### 方式2: 特定领域微调 ⭐ 推荐

```bash
# 1. 安装依赖
pip install torch transformers peft datasets accelerate

# 2. 准备领域数据（医疗/法律/金融）
python data/prepare_domain_data.py --domain medical

# 3. 开始微调（自动适配硬件）
python practical/domain_finetuning.py --domain medical
```

详细教程请查看 [USAGE_GUIDE.md](USAGE_GUIDE.md)

## 🔧 推荐工具
- **Transformers**: Hugging Face 核心库
- **PEFT**: Parameter-Efficient Fine-Tuning
- **TRL**: Transformer Reinforcement Learning
- **Axolotl**: 微调框架
- **LLaMA-Factory**: 开源微调平台

## 🎯 特定领域微调

本项目提供三个特定领域的完整微调示例：

| 领域 | 数据量 | 应用场景 | 文档 |
|------|--------|----------|------|
| **🏥 医疗** | 20条×50 = 1000样本 | 医疗咨询、健康问答 | [开始使用](USAGE_GUIDE.md) |
| **⚖️ 法律** | 15条×50 = 750样本 | 法律咨询、合规问答 | [开始使用](USAGE_GUIDE.md) |
| **💰 金融** | 15条×50 = 750样本 | 投资理财、金融知识 | [开始使用](USAGE_GUIDE.md) |

### 特点

- ✅ **开箱即用**: 无需准备数据，直接运行
- ✅ **硬件自适应**: 自动检测CPU/Mac/NVIDIA GPU
- ✅ **中文支持**: 提供高质量中文数据
- ✅ **完整流程**: 从数据准备到训练到推理

## 📖 学习资源
- [Hugging Face PEFT 文档](https://huggingface.co/docs/peft)
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/)
- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [详细使用指南](USAGE_GUIDE.md)
