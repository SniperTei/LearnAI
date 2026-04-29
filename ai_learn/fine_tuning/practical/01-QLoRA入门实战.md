# QLoRA 微调实战入门

## 环境准备

### 安装依赖

```bash
# 创建虚拟环境
python -m venv ft_env
source ft_env/bin/activate  # Linux/Mac
# ft_env\Scripts\activate  # Windows

# 安装核心库
pip install torch>=2.0.0
pip install transformers>=4.35.0
pip install peft>=0.7.0
pip install bitsandbytes>=0.41.0
pip install datasets>=2.14.0
pip install trl>=0.7.0  # 包含SFT Trainer
pip install accelerate
```

---

## 快速开始：5行代码微调

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

# 1. 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,  # 4位量化
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. 配置LoRA
lora_config = LoraConfig(
    r=16,           # LoRA秩
    lora_alpha=32,  # LoRA缩放参数
    target_modules=["q_proj", "v_proj"],  # 应用的模块
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. 应用LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 查看可训练参数

# 4. 准备数据
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

# 5. 训练
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir="./outputs",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
    ),
)
trainer.train()
```

---

## 详细参数解释

### LoRA 关键参数

```python
LoraConfig(
    # === 核心 ===
    r=16,              # 秩 (rank)。越大=效果越好但参数越多
                       # 典型值: 4, 8, 16, 32, 64

    lora_alpha=32,     # 缩放系数。通常设为 2*r
                       # 调节LoRA层的影响权重

    # === 应用位置 ===
    target_modules=[    # 对哪些模块应用LoRA
        "q_proj",      # 查询投影
        "v_proj",      # 值投影
        "k_proj",      # 键投影
        "o_proj",      # 输出投影
        "gate_proj",   # Gate项目 (FFN)
        "up_proj",     # 上投影 (FFN)
        "down_proj",   # 下投影 (FFN)
    ],

    # === 其他 ===
    lora_dropout=0.05, # Dropout率
    bias="none",       # bias训练方式: "none", "all", "lora_only"
    task_type="CAUSAL_LM",  # 任务类型
)
```

### target_modules 选择建议

| 目标模块 | 参数量增加 | 效果 | 推荐 |
|---------|-----------|------|------|
| 仅 q_proj, v_proj | 最少 | 基础注意力适应 | ✅ 入门 |
| + k_proj, o_proj | 中等 | 更完整的注意力 | ✅ 推荐 |
| + FFN模块 | 较多 | 全方位适应 | ⚠️ 大模型 |

---

## 数据格式

### 格式1: 指令微调

```json
{
    "instruction": "解释什么是机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支..."
}
```

```python
# 数据处理函数
def format_instruction(sample):
    return f"""
### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}
""".strip()

# 应用到数据集
dataset = dataset.map(lambda x: {"text": format_instruction(x)})
```

### 格式2: 对话数据

```json
{
    "conversations": [
        {"from": "human", "value": "你好"},
        {"from": "gpt", "value": "你好！有什么可以帮你的？"}
    ]
}
```

```python
# 对话格式化
def format_conversation(sample):
    conversation = sample['conversations']
    text = ""
    for turn in conversation:
        if turn['from'] == 'human':
            text += f"User: {turn['value']}\n"
        else:
            text += f"Assistant: {turn['value']}\n"
    return text.strip()
```

### 格式3: 纯文本 (继续预训练)

```json
{"text": "这是用于继续预训练的领域文本..."}
```

---

## 推理与加载

### 保存模型

```python
# 训练完成后
trainer.save_model("./my_lora_model")

# 只保存LoRA权重 (~100MB)
model.save_pretrained("./my_lora_model")
tokenizer.save_pretrained("./my_lora_model")
```

### 加载微调模型

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device_map="auto"
)

# 2. 加载LoRA adapter
model = PeftModel.from_pretrained(base_model, "./my_lora_model")

# 3. 推理
prompt = "什么是Fine-tuning？"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 合并模型 (可选)

```python
# 将LoRA权重合并到基础模型
merged_model = model.merge_and_unload()

# 保存完整模型 (7B)
merged_model.save_pretrained("./merged_model")
```

---

## 常用数据集

### 中文数据集

| 数据集 | 描述 | 链接 |
|-------|------|------|
| BelleGroup | 多样化中文指令 | Hugging Face |
| COIG-CQIA | 高质量中文指令 | GitHub |
| InstinctWild | 中文野生指令 | Hugging Face |
| Firefly | 中文多任务 | Hugging Face |

### 英文数据集

| 数据集 | 描述 | 规模 |
|-------|------|------|
| Alpaca | 英文指令 | 52K |
| Dolly | Databricks指令 | 15K |
| OpenAssistant | 开源对话 | 10K |
| ShareGPT | 真实对话 | 90K |

---

## 调优技巧

### 1. 学习率调度

```python
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=len(train_dataset) // batch_size * epochs
)
```

### 2. Gradient Checkpointing

```python
# 节省显存，速度稍慢
model.gradient_checkpointing_enable()
```

### 3. 混合精度训练

```python
# 自动混合精度
from torch.cuda.amp import autocast

with autocast():
    outputs = model(**inputs)
```

---

## 常见问题

### Q1: 显存不足 OOM

**解决方案**:
- 使用 QLoRA (4位)
- 减少 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 启用 `gradient_checkpointing`

### Q2: 训练不稳定

**解决方案**:
- 降低学习率 (1e-5 到 5e-5)
- 增加 `warmup_steps`
- 检查数据质量
- 减少 `max_seq_length`

### Q3: 效果不好

**解决方案**:
- 增加训练数据量和质量
- 调整 LoRA rank (r)
- 增加训练轮数
- 尝试调整 `target_modules`

---

## 完整训练脚本

见 `notebooks/full_training_script.py`

---

## 下一步

- [ ] 运行你的第一次微调
- [ ] 评估微调效果
- [ ] 尝试不同数据集
- [ ] 调优超参数
