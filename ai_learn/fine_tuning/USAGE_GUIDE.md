# 特定领域Fine-tuning 使用指南

本指南将带你完成一个完整的特定领域模型微调项目。

## 📋 目录

- [快速开始](#快速开始)
- [分步详细教程](#分步详细教程)
- [领域数据说明](#领域数据说明)
- [常见问题](#常见问题)
- [进阶技巧](#进阶技巧)

---

## 🚀 快速开始

### 5分钟快速体验

```bash
# 1. 安装依赖
pip install torch transformers peft datasets accelerate

# 2. 准备医疗领域数据
python data/prepare_domain_data.py --domain medical

# 3. 开始微调（会自动检测硬件）
python practical/domain_finetuning.py --domain medical
```

就这么简单！脚本会：
- 自动检测你的硬件（CPU/Mac GPU/NVIDIA GPU）
- 选择合适的模型和配置
- 完成训练并测试结果

---

## 📚 分步详细教程

### 步骤1: 环境准备

#### 安装基础依赖
```bash
# 核心依赖
pip install torch>=2.0.0
pip install transformers>=4.35.0
pip install peft>=0.7.0
pip install datasets>=2.14.0
pip install accelerate>=0.25.0

# 可选: NVIDIA GPU用户（用于4位量化）
pip install bitsandbytes>=0.41.0
```

#### 验证安装
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 步骤2: 选择领域

我们提供三个特定领域的数据集：

| 领域 | 数据量 | 适用场景 | 难度 |
|------|--------|----------|------|
| **医疗** | 20条×50重复 = 1000样本 | 医疗咨询、健康问答 | ⭐⭐⭐ |
| **法律** | 15条×50重复 = 750样本 | 法律咨询、合规问答 | ⭐⭐⭐⭐ |
| **金融** | 15条×50重复 = 750样本 | 投资理财、金融知识 | ⭐⭐⭐⭐ |

#### 选择建议

- **初学者**: 建议从医疗领域开始，问答相对直观
- **有经验**: 可以尝试法律或金融领域
- **专业人士**: 可以使用自己的领域数据

### 步骤3: 准备数据

```bash
# 医疗领域
python data/prepare_domain_data.py --domain medical

# 法律领域
python data/prepare_domain_data.py --domain legal

# 金融领域
python data/prepare_domain_data.py --domain finance
```

数据会保存到 `./data/{domain}_dataset.json`

#### 自定义数据重复次数

默认每条数据重复50次，你可以调整：

```bash
# 重复100次 = 2000样本
python data/prepare_domain_data.py --domain medical --repeat 100

# 重复20次 = 400样本（训练更快）
python data/prepare_domain_data.py --domain legal --repeat 20
```

### 步骤4: 开始训练

#### 基础训练（推荐新手）

```bash
# 医疗领域
python practical/domain_finetuning.py --domain medical

# 法律领域
python practical/domain_finetuning.py --domain legal

# 金融领域
python practical/domain_finetuning.py --domain finance
```

脚本会自动：
- ✅ 检测硬件并选择最优配置
- ✅ 下载合适的模型
- ✅ 配置LoRA参数
- ✅ 开始训练
- ✅ 保存并测试模型

#### 高级配置

```bash
# 指定模型
python practical/domain_finetuning.py \
    --domain medical \
    --model meta-llama/Llama-2-7b-hf

# 调整训练参数
python practical/domain_finetuning.py \
    --domain medical \
    --epochs 5 \
    --batch_size 8 \
    --learning_rate 1e-4

# 调整LoRA参数
python practical/domain_finetuning.py \
    --domain medical \
    --lora_r 32 \
    --lora_alpha 64
```

### 步骤5: 查看结果

训练完成后，脚本会自动测试模型。你也可以手动测试：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载LoRA权重
model = PeftModel.from_pretrained(base_model, "./outputs/medical_xxx")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 测试
prompt = "### Instruction:\n什么是糖尿病？\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📊 领域数据说明

### 医疗领域数据

包含20个精选医疗问答，涵盖：

- **疾病知识**: 高血压、糖尿病、肺炎、中风等
- **检查方法**: CT、MRI、心电图、血常规等
- **药物知识**: 阿司匹林、抗生素、他汀类药物等
- **预防保健**: 体检、疫苗、健康生活方式等
- **中医概念**: 气血、针灸等基础理论

### 法律领域数据

包含15个精选法律问答，涵盖：

- **基础法律**: 刑法、民法典、合同法等
- **权利保护**: 正当防卫、知识产权、消费者权益等
- **特殊领域**: 劳动法、婚姻法、继承法等
- **程序法**: 诉讼、证据、行政复议等

### 金融领域数据

包含15个精选金融问答，涵盖：

- **投资工具**: 股票、债券、基金、ETF、期权等
- **估值指标**: PE、PB、ROE等
- **技术分析**: K线图、移动平均线等
- **投资策略**: 分散投资、止盈止损等
- **经济指标**: GDP、通货膨胀等

---

## 🔧 硬件适配

### NVIDIA GPU

| 显存 | 建议模型 | 批大小 | 预计时间 |
|------|----------|--------|----------|
| 8GB | TinyLlama-1.1B | 2-4 | ~15分钟 |
| 16GB | LLaMA-7B + QLoRA | 4-8 | ~30分钟 |
| 24GB | LLaMA-13B + QLoRA | 4-8 | ~45分钟 |
| 40GB+ | LLaMA-13B Full | 8-16 | ~30分钟 |

### Apple Silicon (M1/M2/M3)

- 模型: Qwen-1.8B或TinyLlama-1.1B
- 批大小: 4
- 预计时间: ~20-30分钟
- 注意: MPS对fp16支持有限，使用fp32训练

### CPU

- 模型: TinyLlama-1.1B（最小型）
- 批大小: 1-2
- 预计时间: 2-4小时
- 建议: 仅用于学习和测试

---

## ❓ 常见问题

### Q1: 训练过程中出现OOM（显存不足）

**解决方案**:

```bash
# 减小批大小
python practical/domain_finetuning.py --domain medical --batch_size 2

# 增加梯度累积
python practical/domain_finetuning.py --domain medical --gradient_accumulation 8

# 减小序列长度
python practical/domain_finetuning.py --domain medical --max_length 256

# 使用更小的模型
python practical/domain_finetuning.py --domain medical --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### Q2: 训练速度太慢

**解决方案**:

```bash
# 减少训练轮数
python practical/domain_finetuning.py --domain medical --epochs 1

# 减少数据重复
python data/prepare_domain_data.py --domain medical --repeat 20

# 增大批大小（如果显存允许）
python practical/domain_finetuning.py --domain medical --batch_size 8
```

### Q3: 效果不好

**解决方案**:

- 增加训练数据量和质量
- 增加训练轮数 (--epochs 5)
- 调整学习率 (--learning_rate 1e-4)
- 增大LoRA rank (--lora_r 32)

### Q4: 想使用自己的数据

创建自定义数据集：

```python
# 你的数据格式
your_data = [
    {
        "instruction": "你的问题",
        "input": "",  # 可选的额外输入
        "output": "期望的回答"
    },
    # ... 更多数据
]

# 保存为JSON
import json
with open('my_data.json', 'w', encoding='utf-8') as f:
    json.dump(your_data, f, ensure_ascii=False)
```

然后修改训练脚本加载你的数据。

---

## 🎯 进阶技巧

### 1. 多数据集混合

```python
# 合并多个领域的数据
from datasets import concatenate_datasets

medical_dataset = load_dataset('./data/medical_dataset.json')
legal_dataset = load_dataset('./data/legal_dataset.json')

combined = concatenate_datasets([medical_dataset, legal_dataset])
```

### 2. 学习率调度

```bash
# 使用余弦退火
python practical/domain_finetuning.py \
    --domain medical \
    --learning_rate 2e-4 \
    --warmup_ratio 0.1
```

### 3. 模型合并

训练完成后，可以合并LoRA权重：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("base_model_path")
model = PeftModel.from_pretrained(base_model, "lora_path")

# 合并
merged = model.merge_and_unload()
merged.save_pretrained("merged_model")
```

### 4. 批量推理

```python
test_questions = [
    "什么是高血压？",
    "如何预防糖尿病？",
    "CT检查的原理是什么？"
]

for question in test_questions:
    prompt = f"### Instruction:\n{question}\n\n### Response:"
    # ... 生成回答
```

---

## 📈 训练监控

### 查看训练日志

```bash
# TensorBoard（如果启用）
tensorboard --logdir ./outputs

# 查看保存的检查点
ls -la ./outputs/medical_*/
```

### 评估指标

虽然本示例主要用于学习，但实际项目中应该：

1. 准备验证集
2. 计算Perplexity
3. 人工评估生成质量
4. BLEU/ROUGE分数（用于文本生成）

---

## 🎓 下一步学习

掌握基础后可以：

1. **数据增强**: 扩充你的领域数据
2. **评估系统**: 建立完整的评估流程
3. **模型部署**: 将微调模型部署为服务
4. **RLHF**: 使用人类反馈进一步优化
5. **多模态**: 图文等多模态微调

---

## 📞 获取帮助

- GitHub Issues: 查看和提交问题
- Hugging Face Forums: 社区讨论
- 本项目README: 查看更多资源

**祝你学习顺利！**
