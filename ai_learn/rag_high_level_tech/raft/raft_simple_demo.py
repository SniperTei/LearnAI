"""
简化版RAFT (Retrieval-Augmented Fine Tuning) 实现
================================================

这是一个教育性质的简化实现，用于理解RAFT的核心概念。
实际生产环境需要更复杂的实现和更多优化。

作者: Claude Code Assistant
日期: 2026-01-27
"""

import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class Document:
    """文档数据结构"""
    id: str
    content: str
    is_relevant: bool  # 是否与问题相关


@dataclass
class RaftSample:
    """RAFT训练样本"""
    question: str
    documents: List[Document]
    answer: str
    citations: List[str]


# ============================================================================
# 模拟数据集
# ============================================================================

def create_sample_dataset() -> List[Document]:
    """
    创建示例文档库
    实际应用中应从文件或数据库加载
    """
    documents = [
        # 相关文档（关于RAG）
        Document("D1", "RAG（检索增强生成）是一种结合了检索和生成的AI技术。", True),
        Document("D2", "RAG通过从外部知识库检索相关文档来增强大语言模型的生成能力。", True),
        Document("D3", "RAFT是RAG的一种高级技术，通过微调模型提升抗干扰能力。", True),
        Document("D4", "Embedding将文本转换为向量，使语义相似的文本在向量空间中距离更近。", True),
        Document("D5", "向量数据库如Chroma、Pinecone用于存储和检索文档向量。", True),

        # 干扰文档（关于其他主题）
        Document("D6", "Python是一种高级编程语言，以其简洁的语法而闻名。", False),
        Document("D7", "机器学习是人工智能的一个分支，使计算机能够从数据中学习。", False),
        Document("D8", "深度学习使用神经网络模拟人脑的学习过程。", False),
        Document("D9", "JavaScript主要用于Web开发，可以在浏览器中运行。", False),
        Document("D10", "SQL是用于管理和查询关系数据库的语言。", False),
        Document("D11", "Docker是一个容器化平台，可以打包应用程序及其依赖项。", False),
        Document("D12", "Git是一个分布式版本控制系统，用于跟踪代码变化。", False),
    ]
    return documents


def create_training_samples() -> List[RaftSample]:
    """
    创建训练样本
    实际应用中应使用Oracle模型（如GPT-4）生成答案
    """
    samples = [
        RaftSample(
            question="什么是RAG？",
            documents=create_sample_dataset()[:8],  # 包含相关和干扰文档
            answer="根据文档[D1]和[D2]，RAG（检索增强生成）是一种结合了检索和生成的AI技术。它通过从外部知识库检索相关文档来增强大语言模型的生成能力。",
            citations=["D1", "D2"]
        ),
        RaftSample(
            question="RAFT是什么？",
            documents=create_sample_dataset()[:8],
            answer="根据文档[D3]，RAFT是RAG的一种高级技术，通过微调模型提升抗干扰能力。",
            citations=["D3"]
        ),
        RaftSample(
            question="什么是向量数据库？",
            documents=create_sample_dataset(),
            answer="根据文档[D5]，向量数据库如Chroma、Pinecone用于存储和检索文档向量。",
            citations=["D5"]
        ),
    ]
    return samples


# ============================================================================
# RAFT核心功能
# ============================================================================

class RaftTrainer:
    """RAFT训练器（简化版）"""

    def __init__(self):
        self.prompt_template = """你是一个专业的问答助手。请仔细阅读提供的文档，并回答用户的问题。

要求：
1. 仅基于提供的文档回答问题
2. 如果文档中没有相关信息，明确说明"提供的文档中没有包含该问题的答案"
3. 回答时引用使用的文档，格式为[文档ID]
4. 忽略与问题无关的文档

文档：
{documents}

问题：{question}

答案："""

    def format_documents(self, documents: List[Document]) -> str:
        """格式化文档"""
        return "\n\n".join([
            f"[{doc.id}] {doc.content}"
            for doc in documents
        ])

    def format_training_sample(self, sample: RaftSample) -> Tuple[str, str]:
        """
        格式化训练样本

        返回: (输入文本, 目标输出)
        """
        docs_text = self.format_documents(sample.documents)

        input_text = self.prompt_template.format(
            documents=docs_text,
            question=sample.question
        )

        target_output = sample.answer

        return input_text, target_output

    def prepare_dataset(self, samples: List[RaftSample]) -> List[Tuple[str, str]]:
        """
        准备训练数据集

        实际应用中应该：
        1. 添加更多样本
        2. 使用数据增强（不同干扰文档组合）
        3. 分割训练集和验证集
        """
        dataset = []
        for sample in samples:
            input_text, output = self.format_training_sample(sample)
            dataset.append((input_text, output))
        return dataset

    def train(self, dataset: List[Tuple[str, int]], epochs: int = 3):
        """
        训练模型（简化版 - 仅演示流程）

        实际应用中应该使用transformers + PEFT进行微调：
        ```python
        from transformers import AutoModelForCausalLM, Trainer
        from peft import LoraConfig, get_peft_model

        model = AutoModelForCausalLM.from_pretrained("base_model")
        lora_config = LoraConfig(r=8, lora_alpha=32, ...)
        model = get_peft_model(model, lora_config)

        trainer = Trainer(model=model, train_dataset=dataset)
        trainer.train()
        ```
        """
        print(f"🚀 开始RAFT训练（简化版演示）")
        print(f"📊 训练样本数: {len(dataset)}")
        print(f"🔄 训练轮数: {epochs}")

        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

            for i, (input_text, output) in enumerate(dataset):
                print(f"\n样本 {i + 1}:")
                question_line = input_text.split('问题：')[1].split('\n')[0]
                print(f"问题: {question_line}")
                print(f"目标答案: {output[:100]}...")

                # 模拟训练过程
                # 实际应用中这里是模型的前向传播和反向传播
                print("✅ 训练步骤完成")

        print("\n✨ 训练完成！")


class RaftInference:
    """RAFT推理器（简化版）"""

    def __init__(self, model_path: str = None):
        """
        初始化推理器

        实际应用中应该加载微调后的模型：
        ```python
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained("base_model")
        tokenizer = AutoTokenizer.from_pretrained("base_model")

        if model_path:
            model = PeftModel.from_pretrained(base_model, model_path)
        ```
        """
        self.prompt_template = """你是一个专业的问答助手。请仔细阅读提供的文档，并回答用户的问题。

要求：
1. 仅基于提供的文档回答问题
2. 如果文档中没有相关信息，明确说明"提供的文档中没有包含该问题的答案"
3. 回答时引用使用的文档，格式为[文档ID]
4. 忽略与问题无关的文档

文档：
{documents}

问题：{question}

答案："""

    def format_documents(self, documents: List[Document]) -> str:
        """格式化文档"""
        return "\n\n".join([
            f"[{doc.id}] {doc.content}"
            for doc in documents
        ])

    def retrieve_documents(self, question: str, all_docs: List[Document], top_k: int = 8) -> List[Document]:
        """
        检索相关文档（简化版 - 实际应使用向量检索）

        实际应用中应该：
        1. 对问题进行Embedding
        2. 在向量数据库中搜索
        3. 返回top-k相关文档 + 一些干扰文档
        """
        # 简化版：随机返回一些文档
        # 实际应用中应使用语义相似度检索
        return random.sample(all_docs, min(top_k, len(all_docs)))

    def generate_answer(self, question: str, documents: List[Document]) -> str:
        """
        生成答案（简化版 - 仅演示流程）

        实际应用中应该：
        ```python
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=256)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ```
        """
        docs_text = self.format_documents(documents)
        prompt = self.prompt_template.format(
            documents=docs_text,
            question=question
        )

        # 简化版：返回模拟答案
        # 实际应用中应该使用微调后的模型生成
        relevant_docs = [doc for doc in documents if doc.is_relevant]

        if relevant_docs:
            citations = ", ".join([f"[{doc.id}]" for doc in relevant_docs])
            answer = f"根据文档{citations}，这是问题的答案。"
        else:
            answer = "提供的文档中没有包含该问题的答案。"

        return answer

    def query(self, question: str, document_store: List[Document]) -> Dict:
        """
        完整的查询流程
        """
        print(f"\n🔍 查询: {question}")

        # 1. 检索文档
        retrieved_docs = self.retrieve_documents(question, document_store)
        print(f"📚 检索到 {len(retrieved_docs)} 个文档")

        # 2. 生成答案
        answer = self.generate_answer(question, retrieved_docs)
        print(f"💡 答案: {answer}")

        # 3. 提取引用（简化版）
        citations = self.extract_citations(answer)
        print(f"📎 引用: {citations}")

        return {
            "question": question,
            "answer": answer,
            "citations": citations,
            "retrieved_docs": retrieved_docs
        }

    def extract_citations(self, answer: str) -> List[str]:
        """从答案中提取引用"""
        import re
        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, answer)
        return matches


# ============================================================================
# 数据生成工具
# ============================================================================

class RaftDataGenerator:
    """RAFT训练数据生成器"""

    def __init__(self):
        pass

    def generate_oracle_answer(self, question: str, relevant_docs: List[Document]) -> str:
        """
        使用Oracle模型生成答案

        实际应用中应该：
        ```python
        import openai
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": f"基于这些文档回答：{relevant_docs}\\n问题：{question}"
            }]
        )
        return response.choices[0].message.content
        ```
        """
        # 简化版：基于相关文档生成简单答案
        if not relevant_docs:
            return "提供的文档中没有包含该问题的答案。"

        citations = ", ".join([f"[{doc.id}]" for doc in relevant_docs])
        return f"根据文档{citations}，这是关于问题的答案。"

    def add_distractor_documents(self, relevant_docs: List[Document],
                                 all_docs: List[Document],
                                 num_distractors: int = 5) -> List[Document]:
        """
        添加干扰文档

        策略：
        1. 随机选择
        2. 困难负样本（语义相似但无关）
        3. 同领域不同主题
        """
        irrelevant_docs = [doc for doc in all_docs if not doc.is_relevant]

        if len(irrelevant_docs) < num_distractors:
            num_distractors = len(irrelevant_docs)

        distractors = random.sample(irrelevant_docs, num_distractors)

        # 合并相关文档和干扰文档
        all_retrieved = relevant_docs + distractors

        # 打乱顺序
        random.shuffle(all_retrieved)

        return all_retrieved

    def create_training_sample(self, question: str, all_docs: List[Document]) -> RaftSample:
        """
        创建单个训练样本

        流程：
        1. 识别相关文档
        2. 使用Oracle生成答案
        3. 添加干扰文档
        4. 返回训练样本
        """
        # 1. 识别相关文档
        relevant_docs = [doc for doc in all_docs if doc.is_relevant]

        # 2. 生成答案
        answer = self.generate_oracle_answer(question, relevant_docs)

        # 3. 添加干扰文档
        all_retrieved = self.add_distractor_documents(relevant_docs, all_docs)

        # 4. 提取引用
        citations = [doc.id for doc in relevant_docs]

        return RaftSample(
            question=question,
            documents=all_retrieved,
            answer=answer,
            citations=citations
        )


# ============================================================================
# 评估工具
# ============================================================================

class RaftEvaluator:
    """RAFT评估器"""

    @staticmethod
    def citation_accuracy(predicted_citations: List[str],
                         gold_citations: List[str]) -> float:
        """
        计算引用准确率
        """
        if not gold_citations:
            return 1.0 if not predicted_citations else 0.0

        overlap = set(predicted_citations) & set(gold_citations)
        return len(overlap) / len(gold_citations)

    @staticmethod
    def distractor_rejection_rate(predicted_citations: List[str],
                                  distractor_ids: List[str]) -> float:
        """
        计算干扰文档排除率
        成功排除干扰文档的比例
        """
        if not distractor_ids:
            return 1.0

        cited = set(predicted_citations)
        distractors = set(distractor_ids)
        wrongly_cited = cited & distractors

        return 1 - (len(wrongly_cited) / len(distractors))

    def evaluate_sample(self, prediction: RaftSample, gold_standard: RaftSample) -> Dict:
        """
        评估单个样本
        """
        citation_acc = self.citation_accuracy(
            prediction.citations,
            gold_standard.citations
        )

        distractor_ids = [doc.id for doc in gold_standard.documents if not doc.is_relevant]
        rejection_rate = self.distractor_rejection_rate(
            prediction.citations,
            distractor_ids
        )

        return {
            "citation_accuracy": citation_acc,
            "distractor_rejection_rate": rejection_rate,
            "overall_score": (citation_acc + rejection_rate) / 2
        }


# ============================================================================
# 主程序演示
# ============================================================================

def main():
    """主程序 - 演示RAFT完整流程"""

    print("=" * 80)
    print("RAFT (Retrieval-Augmented Fine Tuning) 简化版演示")
    print("=" * 80)

    # 1. 准备文档库
    print("\n📚 准备文档库...")
    document_store = create_sample_dataset()
    print(f"✅ 文档库包含 {len(document_store)} 个文档")

    # 2. 创建训练数据
    print("\n📝 创建训练样本...")
    data_generator = RaftDataGenerator()

    questions = [
        "什么是RAG？",
        "RAFT是什么？",
        "什么是向量数据库？"
    ]

    training_samples = []
    for question in questions:
        sample = data_generator.create_training_sample(question, document_store)
        training_samples.append(sample)
        print(f"✅ 创建样本: {question}")

    # 3. 训练模型
    print("\n" + "=" * 80)
    print("🚀 开始训练")
    print("=" * 80)

    trainer = RaftTrainer()
    dataset = trainer.prepare_dataset(training_samples)
    trainer.train(dataset, epochs=2)

    # 4. 保存训练数据（可选）
    print("\n💾 保存训练数据...")
    train_data_json = []
    for sample in training_samples:
        train_data_json.append({
            "question": sample.question,
            "documents": [
                {"id": doc.id, "content": doc.content, "is_relevant": doc.is_relevant}
                for doc in sample.documents
            ],
            "answer": sample.answer,
            "citations": sample.citations
        })

    with open("/Users/zhengnan/Sniper/Developer/github/LearnAgent/ai_learn/rag_high_level_tech/raft_training_data.json", "w", encoding="utf-8") as f:
        json.dump(train_data_json, f, ensure_ascii=False, indent=2)
    print("✅ 训练数据已保存到: raft_training_data.json")

    # 5. 推理演示
    print("\n" + "=" * 80)
    print("🔮 推理演示")
    print("=" * 80)

    inference = RaftInference()

    test_questions = [
        "什么是RAG？",
        "Python是什么？"  # 文档中有相关信息
    ]

    for question in test_questions:
        result = inference.query(question, document_store)

    # 6. 评估演示
    print("\n" + "=" * 80)
    print("📊 评估演示")
    print("=" * 80)

    evaluator = RaftEvaluator()

    # 模拟预测结果
    prediction = training_samples[0]  # 使用第一个样本作为示例

    # 评估
    metrics = evaluator.evaluate_sample(prediction, prediction)

    print(f"\n评估结果:")
    print(f"  引用准确率: {metrics['citation_accuracy']:.2%}")
    print(f"  干扰文档排除率: {metrics['distractor_rejection_rate']:.2%}")
    print(f"  综合得分: {metrics['overall_score']:.2%}")

    print("\n" + "=" * 80)
    print("✨ 演示完成！")
    print("=" * 80)

    print("\n📖 说明:")
    print("1. 这是一个简化的教育性实现，用于理解RAFT的核心概念")
    print("2. 实际应用需要:")
    print("   - 真实的向量检索（Embedding + 向量数据库）")
    print("   - 使用transformers + PEFT进行实际微调")
    print("   - 更大的训练数据集")
    print("   - 使用Oracle模型（如GPT-4）生成高质量答案")
    print("3. 参考文档中的代码注释了解完整的实现细节")


if __name__ == "__main__":
    main()
