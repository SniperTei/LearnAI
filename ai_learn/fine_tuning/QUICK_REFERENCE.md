# 快速参考命令卡

## 📋 常用命令速查

### 数据准备

```bash
# 医疗领域
python data/prepare_domain_data.py --domain medical

# 法律领域
python data/prepare_domain_data.py --domain legal

# 金融领域
python data/prepare_domain_data.py --domain finance

# 自定义重复次数
python data/prepare_domain_data.py --domain medical --repeat 100
```

### 模型训练

```bash
# 基础训练（自动检测硬件）
python practical/domain_finetuning.py --domain medical
python practical/domain_finetuning.py --domain legal
python practical/domain_finetuning.py --domain finance

# 指定模型
python practical/domain_finetuning.py --domain medical --model Qwen/Qwen-1_8B-Chat

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

### 推理测试

```python
# 加载微调模型
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "./outputs/medical_xxx")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 测试
prompt = "### Instruction:\n什么是高血压？\n\n### Response:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 🔧 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--domain` | - | 领域选择 (medical/legal/finance) |
| `--model` | auto | 模型名称 |
| `--epochs` | 3 | 训练轮数 |
| `--batch_size` | auto | 批大小 |
| `--learning_rate` | 2e-4 | 学习率 |
| `--lora_r` | 16 | LoRA秩 |
| `--lora_alpha` | 32 | LoRA缩放系数 |
| `--max_length` | 512 | 最大序列长度 |

## ⚡ 性能优化

### 显存不足？

```bash
# 1. 减小批大小
--batch_size 2

# 2. 增加梯度累积
--gradient_accumulation 8

# 3. 减小序列长度
--max_length 256

# 4. 使用更小模型
--model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### 训练太慢？

```bash
# 1. 减少训练轮数
--epochs 1

# 2. 减少数据重复
python data/prepare_domain_data.py --domain medical --repeat 20

# 3. 增大批大小
--batch_size 8
```

## 📊 硬件建议

| 硬件 | 建议模型 | 批大小 | 预计时间 |
|------|----------|--------|----------|
| CPU | TinyLlama-1.1B | 1-2 | 2-4小时 |
| Mac M1/M2 | Qwen-1.8B | 4 | 20-30分钟 |
| GPU 8GB | TinyLlama-1.1B | 2-4 | ~15分钟 |
| GPU 16GB | LLaMA-7B + QLoRA | 4-8 | ~30分钟 |
| GPU 24GB+ | LLaMA-13B + QLoRA | 4-8 | ~45分钟 |

## 🆘 故障排除

### OOM (显存不足)
```bash
# 减小batch_size或max_length
--batch_size 1 --max_length 256
```

### 训练不收敛
```bash
# 降低学习率
--learning_rate 1e-4
```

### 效果不好
```bash
# 增加训练轮数或数据量
--epochs 5 --repeat 100
```

## 📚 学习路径

1. **新手**: 运行医疗领域示例
2. **进阶**: 尝试不同参数组合
3. **高级**: 使用自己的数据

详细教程: [USAGE_GUIDE.md](USAGE_GUIDE.md)
