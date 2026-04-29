# PEFT (Parameter-Efficient Fine-Tuning) 完全指南

## 🎯 什么是PEFT？

**PEFT**（参数高效微调）是一类只微调少量参数而冻结大部分参数的方法。

### 核心思想

```
传统Full Fine-tuning:
模型参数: 7B (全部更新)
显存需求: ~140GB
训练成本: 极高

PEFT (如LoRA):
模型参数: 7B (只更新 <1%)
显存需求: ~16GB
训练成本: 大幅降低
```

---

## 🤔 为什么需要PEFT？

### 1. 成本问题

| 模型 | 参数量 | Full微调显存 | LoRA显存 | 节省 |
|------|--------|-------------|---------|------|
| LLaMA-7B | 7B | ~140GB | ~16GB | 88% |
| LLaMA-13B | 13B | ~260GB | ~24GB | 91% |
| LLaMA-33B | 33B | ~660GB | ~48GB | 93% |
| LLaMA-65B | 65B | ~1300GB | ~80GB | 94% |

### 2. 实际问题

**场景**: 你有3个不同任务
- 客服助手
- 代码生成
- 医疗问答

**Full Fine-tuning**:
```
需要3个完整模型 = 3 × 7B = 21B 参数
存储: 21B × 2 bytes (fp16) = 42GB
```

**PEFT (LoRA)**:
```
需要1个基础模型 + 3个adapter = 7B + 3 × 40M
存储: 14GB + 240MB ≈ 14GB
节省: ~95%
```

---

## 📚 PEFT 主要方法

### 方法对比图

```
PEFT方法分类树
├── Adapter-based (适配器方法)
│   ├── Adapters (Houlsby et al.)
│   └── AdapterFusion
├── Prefix-based (前缀方法)
│   ├── Prefix Tuning
│   ├── Prompt Tuning
│   └── P-Tuning v2
├── LoRA-based (低秩方法) ⭐最流行
│   ├── LoRA
│   └── QLoRA (量化版)
└── 其他
    ├── BitFit
    ├── Compactor
    └── Diff Pruning
```

---

## 🔥 方法详解

### 1. Adapters（适配器）

**原理**: 在每个Transformer层后插入小型神经网络层

```
原始层输出 → Adapter → 下一层输入
           ↓
    [Down → ReLU → Up]
```

**代码示例**:
```python
from transformers import AutoModelForSeq2SeqLM
from adapters import AutoAdapterModel

model = AutoAdapterModel.from_pretrained("google/flan-t5-large")

# 添加adapter
model.add_adapter("sentiment", config="seq_bn")
model.train_adapter("sentiment")

# 只有adapter参数会被训练
```

**参数量**: 每层约 3-4% 的额外参数
**优点**: 灵活，可堆叠
**缺点**: 增加推理延迟（额外层）

---

### 2. Prefix Tuning（前缀微调）

**原理**: 在每层添加可训练的"虚拟tokens"

```
输入层: [PREFIX][真实输入]
每层attention: [PREFIX][Key][Value]
```

**示例**:
```python
from peft import PrefixTuningConfig, get_peft_model

config = PrefixTuningConfig(
    peft_type="PREFIX_TUNING",
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,  # 虚拟token数量
)

model = get_peft_model(model, config)
```

**参数量**: 只需 0.1% 参数 (每任务约几万个)
**优点**: 参数量最小，不影响推理速度
**缺点**: 大模型上效果好，小模型效果差

**对比**:
| 模型大小 | Prefix Tuning | Full Fine-tuning |
|---------|---------------|------------------|
| < 10B   | 效果一般      | 效果更好         |
| > 10B   | 效果接近      | -                |

---

### 3. Prompt Tuning（提示微调）

**原理**: 只在输入层添加可训练的prompt

```
[SOFT_PROMPT][用户输入]
  ↓ 可训练
```

**示例**:
```python
from peft import PromptTuningConfig, get_peft_model

config = PromptTuningConfig(
    peft_type="PROMPT_TUNING",
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="分类以下文本：",
)

model = get_peft_model(model, config)
```

**关键发现** (Liu et al., 2022):
- 模型 > 10B 参数时，Prompt Tuning ≈ Full Fine-tuning
- 更简单，更高效

---

### 4. LoRA (Low-Rank Adaptation) ⭐

**原理**: 在权重矩阵旁边添加低秩分解矩阵

```
原始: h = W₀x

LoRA: h = W₀x + ΔWx
          ↓
     ΔW = BA  (低秩分解)
     B: d×r
     A: r×d
     r << d (例如 r=16, d=4096)
```

**可视化**:
```
     W₀ (预训练权重, 冻结)
─────────────────┐
                 │
         x ─────→┼──→ h
                 │
         B ─────→┤
    (r×d)       │
         A ─────→┤
    (d×r)       │
     (可训练)    │
```

**代码示例**:
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                    # 秩
    lora_alpha=32,           # 缩放系数
    target_modules=["q_proj", "v_proj"],  # 应用位置
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# 输出: trainable params: 40M || all params: 7B || trainable%: 0.57%
```

**LoRA核心超参数**:

| 参数 | 作用 | 典型值 | 影响 |
|------|------|--------|------|
| r (rank) | 低秩秩 | 4-64 | 越大效果越好但参数越多 |
| lora_alpha | 缩放系数 | 2*r | 控制LoRA层的影响权重 |
| lora_dropout | Dropout | 0.05-0.1 | 防止过拟合 |
| target_modules | 应用位置 | q_proj, v_proj | 选择要微调的模块 |

**target_modules选择**:

```python
# 保守型 (最轻量)
target_modules=["q_proj", "v_proj"]

# 平衡型 (推荐)
target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]

# 积极型 (更多参数，更好效果)
target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
```

**LoRA优点**:
- ✅ 不增加推理延迟（可合并到基础模型）
- ✅ 参数量极小 (<1%)
- ✅ 效果接近Full Fine-tuning
- ✅ 可以轻松切换任务

---

### 5. QLoRA (Quantized LoRA) 🔥

**原理**: LoRA + 4位量化

```
基础模型: 4位量化 (NF4/FP4)
LoRA adapter: 16位训练
```

**关键创新** (Dettmers et al., 2023):

1. **4位NormalFloat (NF4)**
   - 信息论上最优的4位量化
   - 专门为正态分布权重设计

2. **双量化**
   - 对量化常数也进行量化
   - 进一步节省0.5bit/参数

3. **分页优化器**
   - 将优化器状态从GPU移到CPU RAM
   - 训练更大的模型

**示例**:
```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,          # 4位量化
    bnb_4bit_use_double_quant=True,  # 双量化
    bnb_4bit_quant_type="nf4",  # NF4量化类型
)

lora_config = LoraConfig(r=16, ...)
model = get_peft_model(model, lora_config)
```

**对比 LoRA vs QLoRA**:

| 模型 | LoRA显存 | QLoRA显存 | 节省 |
|------|----------|-----------|------|
| 7B   | ~16GB    | ~12GB     | 25%  |
| 13B  | ~24GB    | ~20GB     | 17%  |
| 33B  | ~48GB    | ~36GB     | 25%  |
| 65B  | ~80GB    | ~48GB     | 40%  |

**QLoRA使单卡训练65B模型成为可能！**

---

## 🎯 如何选择PEFT方法？

### 决策树

```
开始
  ↓
模型 > 10B 参数?
  ├─ 是 → Prompt Tuning / Prefix Tuning
  └─ 否 ↓
      推理速度敏感?
      ├─ 是 → LoRA (可合并)
      └─ 否 ↓
          显存 < 24GB?
          ├─ 是 → QLoRA
          └─ 否 → LoRA
```

### 推荐配置

| 场景 | 推荐方法 | r值 | 显存需求 |
|------|----------|-----|----------|
| 个人学习 | QLoRA | 16 | 12GB |
| 研究实验 | LoRA | 8-16 | 16GB |
| 生产部署 | LoRA | 32-64 | 24GB+ |
| 多任务切换 | LoRA | 16 | 16GB×模型数 |

---

## 💻 实战代码

### 完整训练示例

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,  # QLoRA
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. 配置LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. 准备数据
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4. 训练
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./lora-alpaca",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=100,
    ),
    train_dataset=tokenized_dataset,
)

trainer.train()

# 5. 保存LoRA
model.save_pretrained("./my_lora_adapter")
tokenizer.save_pretrained("./my_lora_adapter")
```

### 加载和推理

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# 加载LoRA
model = PeftModel.from_pretrained(base_model, "./my_lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("./my_lora_adapter")

# 推理
prompt = "什么是机器学习？"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 合并模型

```python
# 合并LoRA到基础模型
merged_model = model.merge_and_unload()

# 保存完整模型
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")

# 现在可以像普通模型一样加载
model = AutoModelForCausalLM.from_pretrained("./merged_model")
```

---

## 📊 性能对比

### 数据来源: Hu et al. (2022)

| 任务 | Full FT | LoRA | Prefix | Prompt |
|------|---------|------|--------|--------|
| SST-2 | 95.7 | 95.5 | 94.8 | 94.2 |
| MRPC  | 89.2 | 88.8 | 88.1 | 87.5 |
| RTE   | 76.3 | 75.9 | 75.2 | 74.8 |

**结论**: LoRA在各种任务上最接近Full Fine-tuning

### 显存对比 (LLaMA-7B)

| 方法 | 显存 | 可训练参数 |
|------|------|-----------|
| Full FT | ~140GB | 7B (100%) |
| Adapters | ~16GB | 21M (0.3%) |
| Prefix Tuning | ~14GB | 0.6M (0.008%) |
| **LoRA** | **~16GB** | **40M (0.57%)** |
| **QLoRA** | **~12GB** | **40M (0.57%)** |

---

## 🚀 进阶技巧

### 1. 多LoRA合并

```python
from peft import PeftModel

# 合并多个adapter
model = PeftModel.from_pretrained(base_model, "./adapter1")
model.load_adapter("./adapter2", adapter_name="adapter2")

# 推理时切换
model.set_adapter("adapter1")  # 使用adapter1
model.set_adapter("adapter2")  # 使用adapter2
```

### 2. LoRA权重融合

```python
# 线性插值两个adapter
from copy import deepcopy

def merge_loras(model1, model2, alpha=0.5):
    merged = deepcopy(model1)
    for name, param in merged.named_parameters():
        if 'lora' in name:
            param.data = alpha * param.data + (1-alpha) * model2.state_dict()[name]
    return merged
```

### 3. 自动选择rank

```python
import numpy as np

# 使用奇异值分解确定合适rank
def estimate_rank(matrix, threshold=0.95):
    U, S, V = np.linalg.svd(matrix)
    explained = np.cumsum(S**2) / np.sum(S**2)
    return np.argmax(explained >= threshold) + 1
```

---

## 🎓 总结

### PEFT发展历程

```
2019: Adapters (Houlsby)
  ↓
2021: LoRA (Hu et al.) - 突破性工作
  ↓
2022: Prefix Tuning, Prompt Tuning
  ↓
2023: QLoRA (Dettmers) - 让大模型微调普及
```

### 关键要点

1. **PEFT是必须**: 没有PEFT，大模型微调几乎不可行
2. **LoRA最实用**: 效果好、易用、节省资源
3. **QLoRA性价比最高**: 单卡微调65B模型
4. **选择合适方法**: 根据硬件和需求选择

### 推荐阅读

- LoRA论文: [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- QLoRA论文: [QLoRA: Efficient Finetuning](https://arxiv.org/abs/2305.14314)
- Hugging Face PEFT: [官方文档](https://huggingface.co/docs/peft)

---

## 📖 下一步

- [ ] 运行你的第一个LoRA微调
- [ ] 对比不同r值的效果
- [ ] 尝试多任务LoRA
- [ ] 学习PEFT源码
