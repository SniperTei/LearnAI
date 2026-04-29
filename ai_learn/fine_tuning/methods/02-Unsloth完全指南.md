# Unsloth - 超快 LLM 微调框架

## 🚀 什么是 Unsloth？

**Unsloth** 是一个专为加速 LLaMA、Mistral 等大模型微调而设计的优化库。

### 核心优势

```
传统 Hugging Face 训练:
速度: 1x
显存: 16GB (LLaMA-7B LoRA)

Unsloth 训练:
速度: 2-5x 更快 ⚡
显存: 少 70-80% 💾
效果: 完全相同 ✅
```

---

## 📊 性能对比

### 速度对比 (LLaMA-7B, LoRA r=16)

| 操作 | Hugging Face | Unsloth | 加速 |
|------|--------------|---------|------|
| 单步训练 | 1.0s | 0.3s | **3.3x** |
| 1 epoch (Alpaca) | ~2小时 | ~40分钟 | **3x** |
| 显存峰值 | 16GB | 11GB | **节省31%** |

### 支持的模型

✅ **完全支持**:
- LLaMA (1, 2, 3) - 7B, 13B, 70B
- Mistral - 7B, 8x7B
- Phi-2, Phi-3
- Gemma - 2B, 7B
- Qwen - 1.8B, 7B, 14B

⚠️ **部分支持**:
- 其他 Hugging Face 模型（可能加速较少）

---

## 🔧 核心技术

### 1. 手写 Triton 内核

```python
# 传统 PyTorch
F.linear(x, W)  # 通用矩阵乘法

# Unsloth 手写内核
# 针对特定硬件和模型优化
手动优化的 Triton 内核 → 3-5x 加速
```

### 2. 优化后的注意力机制

```
标准 Flash Attention:
X → Softmax → X

Unsloth 优化:
X → [手动优化] → X
(减少内存读写)
```

### 3. 梯度检查点优化

```python
# 自动选择最优检查点策略
减少显存 → 不增加太多计算时间
```

### 4. 自动混合精度

```python
# 智能使用 fp16/bf16
自动检测硬件支持 → 选择最优精度
```

---

## 💻 安装

### 方式1: 完整安装（推荐）

```bash
# NVIDIA GPU 用户
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes
```

### 方式2: 最小安装

```bash
pip install unsloth
pip install trl peft accelerate
```

### 验证安装

```python
import torch
import unsloth

print(f"PyTorch: {torch.__version__}")
print(f"Unsloth: {unsloth.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```

---

## 🎯 快速开始

### 基础示例：LLaMA-3-8B

```python
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# 1. 加载模型（Unsloth优化）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",  # 4位量化
    max_seq_length = 2048,
    dtype = None,  # 自动检测
    load_in_4bit = True,
)

# 2. 配置 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,  # Unsloth优化版本
    random_state = 3407,
    use_rslora = False,  # Rank stabilized LoRA
    loftq_config = None,  # LoftQ initialization
)

# 3. 准备数据
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")

# 4. 训练
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60,  # 示例用少量步数
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",  # 8bit AdamW优化器
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer.train()

# 5. 保存
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
```

---

## 📚 完整实战：特定领域微调

### 医疗领域微调（使用你的数据）

```python
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
import json
from datasets import Dataset
import torch

# ==================== 1. 加载模型 ====================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# ==================== 2. 配置 LoRA ====================
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # Unsloth优化版本
    random_state = 42,
    use_rslora = False,
    loftq_config = None,
)

# ==================== 3. 准备数据 ====================
# 加载你准备好的医疗数据
with open('./data/medical_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# ==================== 4. 训练 ====================
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 512,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 100,  # 可调整为 num_train_epochs = 3
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 42,
        output_dir = "outputs/medical_unsloth",
    ),
)

print("开始训练...")
trainer.train()

# ==================== 5. 保存模型 ====================
# 保存 LoRA adapter
model.save_pretrained("medical_lora")
tokenizer.save_pretrained("medical_lora")

# 保存为 GGUF 格式（用于 llama.cpp）
model.save_pretrained_gguf("medical_gguf", tokenizer, quantization_method = "q4_k_m")

print("训练完成！")
```

---

## 🎨 数据格式

### Alpaca 格式

```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]

    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}

# 应用到数据集
dataset = dataset.map(formatting_prompts_func, batched = True)
```

### ShareGPT 格式（对话）

```python
from unsloth import to_sharegpt

dataset = to_sharegpt(
    dataset,
    merged_column = "conversations",
    output_column_name = "text",
)
```

---

## 🔥 高级功能

### 1. Rank Stabilized LoRA (RsLoRA)

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    use_rslora = True,  # 启用 RsLoRA
    # RsLoRA: 更稳定的训练，更好的收敛
)
```

### 2. LoftQ 初始化

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    loftq_config = {
        "loftq_bits": 4,
        "loftq_iter": 5,
    },
    # LoftQ: 更好的初始化，更快收敛
)
```

### 3. DPO 训练（直接偏好优化）

```python
from unsloth import FastLanguageModel
from unsloth.trainer import UnslothRewardModelTrainer, UnslothDPOTrainer

# DPO 需要 reference model
model = FastLanguageModel.from_pretrained(...)
ref_model = FastLanguageModel.from_pretrained(...)

dpo_trainer = UnslothDPOTrainer(
    model = model,
    ref_model = ref_model,
    train_dataset = train_dataset,
    ...
)
```

---

## 📊 推理

### 使用 Unsloth 模型推理

```python
from unsloth import FastLanguageModel

# 1. 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model",  # 你的 LoRA 模型路径
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 2. 启用快速推理
FastLanguageModel.for_inference(model)  # 启用 2x 更快推理

# 3. 生成
prompt = "### Instruction:\n什么是高血压？\n\n### Response:\n"
inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens = 200,
    use_cache = True,  # Unsloth 优化
    temperature = 0.7,
    top_p = 0.9,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 导出为其他格式

```python
# 导出为 GGUF（llama.cpp）
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")

# 导出为 vLLM
model.save_pretrained_vllm("model_vllm")

# 合并 LoRA（用于 Hugging Face）
model.save_pretrained_merged("model_merged", tokenizer, save_method = "merged")
```

---

## 🎯 实用技巧

### 1. 批大小自动调优

```python
# Unsloth 会自动选择最优批大小
# 你可以手动指定
trainer = SFTTrainer(
    ...,
    per_device_train_batch_size = 4,  # 会根据显存自动调整
)
```

### 2. 混合精度自动选择

```python
# Unsloth 自动选择最优精度
# Ampere GPU (A100, 3090): bf16
# 旧 GPU: fp16

dtype = None  # 自动选择
```

### 3. 多 GPU 支持

```python
# 自动数据并行
trainer = SFTTrainer(
    ...,
    # 自动使用所有可用 GPU
)
```

---

## 📈 性能优化建议

### 1. 最大速度配置

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,  # 自动选择 bf16/fp16
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    use_gradient_checkpointing = "unsloth",  # Unsloth优化
    random_state = 42,
)

trainer = SFTTrainer(
    model = model,
    ...,
    optim = "adamw_8bit",  # 8bit优化器
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
)
```

### 2. 显存优化

```python
# 减少批大小
per_device_train_batch_size = 1

# 增加梯度累积
gradient_accumulation_steps = 8

# 减少序列长度
max_seq_length = 1024  # 而不是 2048
```

---

## 🔍 故障排除

### 问题1: OOM (显存不足)

```python
# 解决方案1: 减小批大小
per_device_train_batch_size = 1

# 解决方案2: 减少序列长度
max_seq_length = 1024

# 解决方案3: 启用梯度检查点
use_gradient_checkpointing = "unsloth"

# 解决方案4: 使用更小的模型
model_name = "unsloth/llama-3-8b-bnb-4bit"  # 而不是 70B
```

### 问题2: 训练慢

```python
# 检查1: 是否使用了 bf16/fp16
fp16 = not torch.cuda.is_bf16_supported()
bf16 = torch.cuda.is_bf16_supported()

# 检查2: 是否启用了优化器
optim = "adamw_8bit"

# 检查3: GPU 是否支持
torch.cuda.is_available()
```

### 问题3: 效果不好

```python
# 增加 LoRA rank
r = 32  # 而不是 16

# 增加训练步数
max_steps = 500  # 而不是 100

# 使用 RsLoRA
use_rslora = True
```

---

## 🆚 Unsloth vs 其他框架

### 对比表

| 特性 | Unsloth | Hugging Face | LLaMA-Factory |
|------|---------|--------------|---------------|
| 训练速度 | ⚡⚡⚡⚡⚡ | ⚡⚡ | ⚡⚡⚡ |
| 显存使用 | 💾💾💾💾 | 💾💾 | 💾💾💾 |
| 易用性 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Web UI | ❌ | ❌ | ✅ |
| 灵活性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 何时使用 Unsloth？

✅ **使用 Unsloth**:
- 追求最快训练速度
- 进行研究和实验
- 需要快速迭代
- 使用支持的模型

⚠️ **考虑其他工具**:
- 需要 Web UI (用 LLaMA-Factory)
- 使用不支持的模型
- 需要高级定制

---

## 📖 推荐资源

### 官方资源
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth 文档](https://unsloth.ai/)
- [示例 Colab Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/notebooks/Unsloth_Quickstart.ipynb)

### 相关论文
- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- QLoRA: [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)

---

## 🎓 总结

### Unsloth 核心优势

1. **超快速度** - 2-5x 训练加速
2. **低显存** - 减少 70-80% 显存
3. **易用性** - API 简洁直观
4. **效果相同** - 与 Hugging Face 完全一致

### 推荐工作流

```
快速原型 → Unsloth (迭代快)
  ↓
满意结果 → 导出模型
  ↓
生产部署 → 转换为 GGUF/vLLM
```

### 下一步

- [ ] 运行你的第一个 Unsloth 训练
- [ ] 对比 Unsloth vs HF 训练时间
- [ ] 尝试不同的优化选项
- [ ] 部署 Unsloth 训练的模型
