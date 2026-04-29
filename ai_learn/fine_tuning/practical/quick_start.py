"""
Fine-tuning 快速开始示例
使用 QLoRA 微调一个小型模型

运行前请先安装依赖:
pip install torch transformers peft datasets bitsandbytes trl accelerate
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os

# ==================== 配置 ====================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 小模型适合快速学习
OUTPUT_DIR = "./tinyllama-lora"

# LoRA 配置
LORA_R = 16           # 秩
LORA_ALPHA = 32       # 缩放系数
LORA_DROPOUT = 0.05   # Dropout

# 训练配置
NUM_EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512


# ==================== 准备示例数据 ====================
def create_sample_data():
    """创建一个简单的指令微调数据集"""

    data = [
        {
            "instruction": "什么是机器学习？",
            "output": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并改进，而无需明确编程。"
        },
        {
            "instruction": "解释什么是深度学习",
            "output": "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的层次表示。"
        },
        {
            "instruction": "Python中如何创建列表？",
            "output": "在Python中，可以使用方括号创建列表。例如：my_list = [1, 2, 3] 或 my_list = ['a', 'b', 'c']"
        },
        {
            "instruction": "什么是神经网络？",
            "output": "神经网络是一种受人脑结构启发的计算模型，由相互连接的节点（神经元）组成，用于模式识别和数据处理。"
        },
        {
            "instruction": "如何遍历Python字典？",
            "output": "可以使用for循环遍历字典。例如：for key, value in my_dict.items(): print(key, value)"
        },
        {
            "instruction": "什么是自然语言处理？",
            "output": "自然语言处理（NLP）是人工智能的一个领域，专注于计算机与人类语言之间的交互，包括理解和生成文本。"
        },
        {
            "instruction": "解释什么是过拟合",
            "output": "过拟合是指模型在训练数据上表现很好，但在新数据上表现较差的现象，通常是因为模型过于复杂或训练数据太少。"
        },
        {
            "instruction": "Python中如何读取文件？",
            "output": "可以使用with语句安全地读取文件。例如：with open('file.txt', 'r') as f: content = f.read()"
        },
        {
            "instruction": "什么是强化学习？",
            "output": "强化学习是机器学习的一种方法，智能体通过与环境交互并获得奖励或惩罚来学习最优策略。"
        },
        {
            "instruction": "如何安装Python包？",
            "output": "使用pip命令可以安装Python包。例如：pip install package_name 或 pip install -r requirements.txt"
        },
    ]

    # 格式化数据
    formatted_data = []
    for item in data:
        text = f"""### Instruction:
{item['instruction']}

### Response:
{item['output']}"""
        formatted_data.append({"text": text.strip()})

    # 创建Dataset对象并重复以增加数据量
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.repeat(50)  # 重复50次得到500条样本

    return dataset.shuffle(seed=42)


# ==================== 加载模型 ====================
def load_model_and_tokenizer():
    """加载模型和分词器"""

    print(f"加载模型: {MODEL_NAME}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad token

    # 加载模型 (4位量化)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("✓ 模型加载成功")
    return model, tokenizer


# ==================== 配置LoRA ====================
def setup_lora(model):
    """配置并应用LoRA"""

    print(f"配置LoRA: r={LORA_R}, alpha={LORA_ALPHA}")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],  # 应用到attention
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("✓ LoRA配置完成")
    return model


# ==================== 训练 ====================
def train_model(model, tokenizer, dataset):
    """训练模型"""

    print("开始训练...")

    # 简单的训练循环
    from transformers import Trainer, DataCollatorForLanguageModeling

    # Tokenization函数
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

    # 处理数据
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        overwrite_output_dir=True,
        report_to="none",  # 不使用wandb等
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM不是masked LM
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    trainer.train()

    print("✓ 训练完成")

    # 保存模型
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ 模型已保存到: {OUTPUT_DIR}")

    return trainer


# ==================== 测试 ====================
def test_model(model, tokenizer):
    """测试微调后的模型"""

    print("\n" + "="*50)
    print("测试微调后的模型")
    print("="*50)

    test_prompts = [
        "什么是深度学习？",
        "如何遍历Python列表？",
    ]

    model.eval()

    for prompt in test_prompts:
        full_prompt = f"""### Instruction:
{prompt}

### Response:"""

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n问题: {prompt}")
        print(f"回答: {response.split('### Response:')[-1].strip()}")
        print("-" * 50)


# ==================== 主函数 ====================
def main():
    """主流程"""

    print("="*60)
    print("Fine-tuning 快速开始示例")
    print("="*60)
    print()

    # 1. 准备数据
    print("准备训练数据...")
    dataset = create_sample_data()
    print(f"✓ 数据集大小: {len(dataset)}")
    print()

    # 2. 加载模型
    model, tokenizer = load_model_and_tokenizer()
    print()

    # 3. 配置LoRA
    model = setup_lora(model)
    print()

    # 4. 训练
    trainer = train_model(model, tokenizer, dataset)
    print()

    # 5. 测试
    test_model(model, tokenizer)

    print("\n" + "="*60)
    print("完成！")
    print("="*60)


if __name__ == "__main__":
    main()
