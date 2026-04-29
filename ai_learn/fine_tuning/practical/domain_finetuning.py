"""
特定领域Fine-tuning完整训练脚本
适配不同硬件环境: CPU, MPS (Mac), CUDA (NVIDIA GPU)

使用示例:
python domain_finetuning.py --domain medical --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
python domain_finetuning.py --domain legal --model Qwen/Qwen-1_8B-Chat
python domain_finetuning.py --domain finance --epochs 3 --batch_size 2
"""

import os
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType


# ==================== 配置 ====================
DOMAIN_CONFIGS = {
    'medical': {
        'name': '医疗问答',
        'data_file': './data/medical_dataset.json',
        'model_name': 'medical-llama',
        'prompt_example': '什么是高血压？'
    },
    'legal': {
        'name': '法律问答',
        'data_file': './data/legal_dataset.json',
        'model_name': 'legal-llama',
        'prompt_example': '什么是正当防卫？'
    },
    'finance': {
        'name': '金融知识',
        'data_file': './data/finance_dataset.json',
        'model_name': 'finance-llama',
        'prompt_example': '什么是股票？'
    }
}


# ==================== 硬件检测与配置 ====================
def detect_hardware():
    """检测可用硬件并返回配置"""
    device_info = {
        'device': 'cpu',
        'device_name': 'CPU',
        'use_bitsandbytes': False,
        'fp16': False,
        'suggested_model': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    }

    print("\n" + "="*60)
    print("硬件检测")
    print("="*60)

    # 检测CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        device_info['device'] = 'cuda'
        device_info['device_name'] = f"{gpu_name} ({gpu_memory:.1f}GB)"
        device_info['fp16'] = True

        # 根据显存大小建议配置
        if gpu_memory >= 40:
            device_info['use_bitsandbytes'] = False
            device_info['suggested_model'] = 'meta-llama/Llama-2-13b-hf'
            device_info['recommended_batch_size'] = 4
        elif gpu_memory >= 24:
            device_info['use_bitsandbytes'] = True
            device_info['suggested_model'] = 'meta-llama/Llama-2-7b-hf'
            device_info['recommended_batch_size'] = 4
        elif gpu_memory >= 16:
            device_info['use_bitsandbytes'] = True
            device_info['suggested_model'] = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
            device_info['recommended_batch_size'] = 8
        elif gpu_memory >= 10:
            device_info['use_bitsandbytes'] = True
            device_info['suggested_model'] = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
            device_info['recommended_batch_size'] = 4
        else:
            device_info['use_bitsandbytes'] = True
            device_info['suggested_model'] = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
            device_info['recommended_batch_size'] = 2

        print(f"✓ 检测到NVIDIA GPU: {gpu_name}")
        print(f"  显存: {gpu_memory:.1f} GB")
        print(f"  建议模型: {device_info['suggested_model']}")
        print(f"  建议批大小: {device_info['recommended_batch_size']}")

    # 检测MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_info['device'] = 'mps'
        device_info['device_name'] = 'Apple Silicon GPU (MPS)'
        device_info['fp16'] = False  # MPS对fp16支持有限
        device_info['suggested_model'] = 'Qwen/Qwen-1_8B-Chat'
        device_info['recommended_batch_size'] = 4

        print(f"✓ 检测到Apple Silicon GPU (MPS)")
        print(f"  建议模型: {device_info['suggested_model']}")
        print(f"  建议批大小: {device_info['recommended_batch_size']}")

    # CPU
    else:
        print(f"✓ 使用CPU训练")
        print(f"  警告: CPU训练速度较慢，建议使用小型模型")
        print(f"  建议模型: {device_info['suggested_model']}")
        print(f"  建议批大小: 2")
        device_info['recommended_batch_size'] = 2

    print("="*60 + "\n")

    return device_info


# ==================== 数据加载 ====================
def load_dataset(data_file):
    """加载准备好的数据集"""
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"数据文件不存在: {data_file}\n"
            f"请先运行: python prepare_domain_data.py --domain <domain>"
        )

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = Dataset.from_list(data)
    print(f"✓ 加载数据集: {len(dataset)} 条样本")

    return dataset


# ==================== 模型加载 ====================
def load_model_and_tokenizer(model_name, hardware_config):
    """根据硬件配置加载模型"""
    print(f"加载模型: {model_name}")

    load_kwargs = {
        'trust_remote_code': True,
    }

    # 根据硬件配置加载参数
    if hardware_config['device'] == 'cuda':
        if hardware_config['use_bitsandbytes']:
            try:
                import bitsandbytes as bnb
                load_kwargs.update({
                    'load_in_4bit': True,
                    'bnb_4bit_compute_dtype': torch.float16,
                    'bnb_4bit_use_double_quant': True,
                })
                print("  使用4位量化 (QLoRA)")
            except ImportError:
                print("  警告: 未安装bitsandbytes，使用fp16")
                load_kwargs['torch_dtype'] = torch.float16
        else:
            load_kwargs['torch_dtype'] = torch.float16
            print("  使用fp16精度")

        load_kwargs['device_map'] = 'auto'

    elif hardware_config['device'] == 'mps':
        load_kwargs['torch_dtype'] = torch.float32
        print("  使用fp32精度 (MPS)")
    else:  # CPU
        print("  使用CPU")

    # 加载模型
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as e:
        print(f"  错误: {e}")
        print(f"  尝试下载模型...")
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ 模型加载成功")

    return model, tokenizer


# ==================== LoRA配置 ====================
def setup_lora(model, lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """配置LoRA"""
    print(f"\n配置LoRA:")
    print(f"  Rank (r): {lora_r}")
    print(f"  Alpha: {lora_alpha}")
    print(f"  Dropout: {lora_dropout}")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"✓ LoRA配置完成\n")

    return model


# ==================== 训练 ====================
def train_model(model, tokenizer, dataset, output_dir, hardware_config, args):
    """训练模型"""

    print("\n" + "="*60)
    print("开始训练")
    print("="*60)

    # Tokenization
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    print("Tokenizing数据...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "instruction", "output"],
        desc="Tokenizing"
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        fp16=hardware_config['fp16'],
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        overwrite_output_dir=True,
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    print(f"\n训练参数:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Max length: {args.max_length}")
    print()

    trainer.train()

    print("\n✓ 训练完成")

    # 保存模型
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ 模型已保存到: {output_dir}\n")

    return trainer


# ==================== 测试 ====================
def test_model(model, tokenizer, test_prompt, domain_name):
    """测试微调后的模型"""
    print("\n" + "="*60)
    print(f"测试微调后的模型 - {domain_name}")
    print("="*60 + "\n")

    model.eval()

    full_prompt = f"""### Instruction:
{test_prompt}

### Response:"""

    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split('### Response:')[-1].strip()

    print(f"问题: {test_prompt}")
    print(f"\n回答: {answer}")
    print("\n" + "="*60)


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='特定领域Fine-tuning')

    # 必需参数
    parser.add_argument('--domain', type=str, required=True,
                        choices=['medical', 'legal', 'finance'],
                        help='领域选择')

    # 模型参数
    parser.add_argument('--model', type=str, default=None,
                        help='模型名称 (不指定则使用硬件检测的建议模型)')

    # LoRA参数
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank (默认16)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha (默认32)')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout (默认0.05)')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练轮数 (默认3)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批大小 (不指定则使用硬件建议值)')
    parser.add_argument('--gradient_accumulation', type=int, default=4,
                        help='梯度累积步数 (默认4)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='学习率 (默认2e-4)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度 (默认512)')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='日志记录间隔 (默认10)')
    parser.add_argument('--save_steps', type=int, default=100,
                        help='保存间隔 (默认100)')

    # 其他
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='输出目录 (默认./outputs)')
    parser.add_argument('--skip_prepare', action='store_true',
                        help='跳过数据准备 (如果数据已存在)')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("特定领域Fine-tuning训练")
    print("="*60)
    print(f"领域: {DOMAIN_CONFIGS[args.domain]['name']}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 1. 硬件检测
    hardware_config = detect_hardware()

    # 2. 模型选择
    if args.model is None:
        args.model = hardware_config['suggested_model']
        print(f"使用建议模型: {args.model}\n")

    # 3. 准备数据
    data_file = DOMAIN_CONFIGS[args.domain]['data_file']

    if not os.path.exists(data_file) and not args.skip_prepare:
        print(f"准备数据集...")
        os.system(f"python data/prepare_domain_data.py --domain {args.domain}")

    dataset = load_dataset(data_file)

    # 4. 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model, hardware_config)

    # 5. 配置LoRA
    model = setup_lora(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    # 6. 设置输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"{args.domain}_{timestamp}")

    # 7. 训练
    if args.batch_size is None:
        args.batch_size = hardware_config.get('recommended_batch_size', 4)

    trainer = train_model(model, tokenizer, dataset, output_dir, hardware_config, args)

    # 8. 测试
    test_prompt = DOMAIN_CONFIGS[args.domain]['prompt_example']
    test_model(model, tokenizer, test_prompt, DOMAIN_CONFIGS[args.domain]['name'])

    print(f"\n✓ 完成！模型保存在: {output_dir}")
    print("\n使用模型进行推理:")
    print(f"from peft import PeftModel")
    print(f"base_model = AutoModelForCausalLM.from_pretrained('{args.model}')")
    print(f"model = PeftModel.from_pretrained(base_model, '{output_dir}')")


if __name__ == "__main__":
    main()
