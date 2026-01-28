"""
词表示方法对比：One-Hot vs Embedding
通俗易懂地解释为什么需要词嵌入
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("词表示方法对比：One-Hot vs Embedding")
print("="*70)

# ===== 第一部分：One-Hot 编码 =====
print("\n" + "="*70)
print("第一部分：One-Hot 编码（初级方案）")
print("="*70)

# 假设我们有一个小型词汇表
vocabulary = ["我", "爱", "北京", "天安门", "中国", "长城"]

print("\n词汇表:", vocabulary)
print(f"词汇表大小: {len(vocabulary)} 个词")

# 创建 One-Hot 编码器
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
vocabulary_reshaped = np.array(vocabulary).reshape(-1, 1)
encoder.fit(vocabulary_reshaped)

print("\n" + "-"*70)
print("One-Hot 编码结果：")
print("-"*70)

for word in vocabulary:
    one_hot = encoder.transform([[word]])[0]
    print(f"{word:6s} → {one_hot}")

print("\n" + "-"*70)
print("问题分析：")
print("-"*70)

# 计算稀疏度
total_elements = len(vocabulary) * len(vocabulary)
non_zero_elements = len(vocabulary)
sparsity = (total_elements - non_zero_elements) / total_elements * 100

print(f"1. 空间浪费严重：")
print(f"   - 总元素数: {total_elements}")
print(f"   - 非0元素: {non_zero_elements}")
print(f"   - 稀疏度: {sparsity:.1f}% (大部分都是0)")

print(f"\n2. 无法表示相似性：")
print(f"   - '北京' 和 '中国' 的相似度应该很高")
print(f"   - 但 One-Hot 计算相似度 = 0 (因为完全不同)")

# 计算相似度
word1_one_hot = encoder.transform([["北京"]])[0]
word2_one_hot = encoder.transform([["中国"]])[0]
cosine_sim = np.dot(word1_one_hot, word2_one_hot) / (np.linalg.norm(word1_one_hot) * np.linalg.norm(word2_one_hot))

print(f"   - 余弦相似度: {cosine_sim:.2f} (0表示完全不同)")

print(f"\n3. 维度灾难：")
print(f"   - 如果词汇表有 100,000 个词")
print(f"   - 每个词需要 100,000 维向量")
print(f"   - 内存占用巨大，计算效率低")

# ===== 第二部分：Word Embedding =====
print("\n" + "="*70)
print("第二部分：Word Embedding（聪明方案）")
print("="*70)

# 手动构造一个小型词向量（用于演示）
# 在实际中，这些是通过算法（如Word2Vec）从大量文本中学习得到的
embeddings = {
    "我":     np.array([0.8,  0.2,  0.1, -0.5]),
    "爱":     np.array([0.3,  0.7,  0.2, -0.1]),
    "北京":   np.array([0.1, -0.3,  0.9,  0.8]),
    "天安门": np.array([0.2, -0.2,  0.8,  0.9]),
    "中国":   np.array([0.0, -0.4,  0.9,  0.7]),
    "长城":   np.array([-0.1, -0.3,  0.9,  0.8]),
}

print(f"\n使用固定维度的密集向量表示（这里用 4 维演示，实际常用 50-300 维）")
print(f"词汇表大小: {len(vocabulary)} 个词")
print(f"每个词向量维度: 4")

print("\n" + "-"*70)
print("Embedding 编码结果：")
print("-"*70)

for word in vocabulary:
    embedding = embeddings[word]
    print(f"{word:6s} → [{embedding[0]:+.1f}, {embedding[1]:+.1f}, {embedding[2]:+.1f}, {embedding[3]:+.1f}]")

print("\n" + "-"*70)
print("优势分析：")
print("-"*70)

# 计算相似度
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

similarities = [
    ("北京", "中国", cosine_similarity(embeddings["北京"], embeddings["中国"])),
    ("北京", "天安门", cosine_similarity(embeddings["北京"], embeddings["天安门"])),
    ("北京", "长城", cosine_similarity(embeddings["北京"], embeddings["长城"])),
    ("我", "爱", cosine_similarity(embeddings["我"], embeddings["爱"])),
    ("北京", "我", cosine_similarity(embeddings["北京"], embeddings["我"])),
]

print(f"1. 可以表示相似性：")
for word1, word2, sim in similarities:
    bar = "█" * int(sim * 30)
    print(f"   - '{word1}' 和 '{word2}': {sim:.3f} {bar}")

print(f"\n2. 空间效率高：")
print(f"   - 每个词只用 4 个数字（实际常用 50-300 个）")
print(f"   - 密集向量，没有浪费")
print(f"   - 即使词汇表 100,000 个词，每个词还是只用 50-300 维")

print(f"\n3. 语义信息：")
print(f"   - '北京'、'天安门'、'中国'、'长城' 相似度都很高")
print(f"   - 因为它们经常出现在相似的语境中")
print(f"   - 模型自动学习到了这些语义关系")

# ===== 第三部分：可视化对比 =====
print("\n" + "="*70)
print("第三部分：可视化对比")
print("="*70)

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 子图1：One-Hot 可视化
ax1 = axes[0]
one_hot_matrix = encoder.transform(vocabulary_reshaped)
im1 = ax1.imshow(one_hot_matrix, cmap='Blues', aspect='auto')

ax1.set_xticks(range(len(vocabulary)))
ax1.set_yticks(range(len(vocabulary)))
ax1.set_xticklabels(vocabulary, fontsize=10)
ax1.set_yticklabels(vocabulary, fontsize=10)
ax1.set_title('One-Hot Encoding (稀疏矩阵)', fontsize=14, fontweight='bold')
ax1.set_xlabel('词汇位置', fontsize=12)
ax1.set_ylabel('单词', fontsize=12)

# 在每个格子标注0/1
for i in range(len(vocabulary)):
    for j in range(len(vocabulary)):
        text = ax1.text(j, i, f'{int(one_hot_matrix[i, j])}',
                       ha="center", va="center", color="black", fontsize=8)

# 子图2：Embedding 可视化（用前2维）
ax2 = axes[1]

# 提取前2维
words = list(embeddings.keys())
vectors_2d = np.array([embeddings[w][:2] for w in words])

# 绘制散点图
colors = ['red', 'orange', 'blue', 'blue', 'blue', 'blue']
sizes = [100] * len(words)

for i, word in enumerate(words):
    ax2.scatter(vectors_2d[i, 0], vectors_2d[i, 1], c=colors[i], s=sizes[i], alpha=0.6, edgecolors='black')
    ax2.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]),
                xytext=(5, 5), textcoords='offset points', fontsize=11)

# 添加图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='代词'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='动词'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='地名/国家'),
]
ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)

ax2.set_title('Word Embedding (密集向量，仅展示前2维)', fontsize=14, fontweight='bold')
ax2.set_xlabel('维度 1', fontsize=12)
ax2.set_ylabel('维度 2', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('one_hot_vs_embedding.png', dpi=100, bbox_inches='tight')
print("\n✓ 可视化对比图已保存: one_hot_vs_embedding.png")
plt.close()

# ===== 第四部分：实际应用对比 =====
print("\n" + "="*70)
print("第四部分：实际应用 - 句子表示")
print("="*70)

sentence1 = "我爱北京"
sentence2 = "我爱中国"
sentence3 = "我爱天安门"

print(f"\n示例句子：")
print(f"  句子1: {sentence1}")
print(f"  句子2: {sentence2}")
print(f"  句子3: {sentence3}")

print(f"\n{'='*70}")
print("方法1: One-Hot 表示")
print("="*70)

def one_hot_sentence(sentence):
    words = list(sentence)
    vectors = [encoder.transform([[w]])[0] for w in words if w in vocabulary]
    return np.sum(vectors, axis=0)

vec1_onehot = one_hot_sentence(sentence1)
vec2_onehot = one_hot_sentence(sentence2)
vec3_onehot = one_hot_sentence(sentence3)

print(f"\n句子1 向量维度: {len(vec1_onehot)}")
print(f"非零元素数: {np.count_nonzero(vec1_onehot)}")

sim_12_oh = cosine_similarity(vec1_onehot, vec2_onehot)
sim_13_oh = cosine_similarity(vec1_onehot, vec3_onehot)

print(f"\n句子相似度：")
print(f"  句子1 vs 句子2: {sim_12_oh:.3f}")
print(f"  句子1 vs 句子3: {sim_13_oh:.3f}")
print(f"  问题: 句子1和句子3都有'我'和'爱'，相似度应该更高！")

print(f"\n{'='*70}")
print("方法2: Embedding 表示")
print("="*70)

def embedding_sentence(sentence):
    words = list(sentence)
    vectors = [embeddings[w] for w in words if w in embeddings]
    return np.mean(vectors, axis=0)

vec1_emb = embedding_sentence(sentence1)
vec2_emb = embedding_sentence(sentence2)
vec3_emb = embedding_sentence(sentence3)

print(f"\n句子1 向量维度: {len(vec1_emb)}")
print(f"向量类型: 密集向量（所有元素都有意义）")

sim_12_emb = cosine_similarity(vec1_emb, vec2_emb)
sim_13_emb = cosine_similarity(vec1_emb, vec3_emb)

print(f"\n句子相似度：")
print(f"  句子1 vs 句子2: {sim_12_emb:.3f}")
print(f"  句子1 vs 句子3: {sim_13_emb:.3f}")
print(f"  优势: Embedding 能更好地捕捉语义相似性！")

# ===== 第五部分：总结 =====
print("\n" + "="*70)
print("总结：One-Hot vs Embedding")
print("="*70)

comparison = [
    ("维度", f"{len(vocabulary)} (词汇表大小)", f"4 (固定维度)"),
    ("稀疏性", f"{sparsity:.0f}% 是0", "100% 都有值"),
    ("相似性", "无法表示", "自动学习"),
    ("空间效率", "低（大词汇表时）", "高"),
    ("语义理解", "❌", "✅"),
    ("应用场景", "小型词汇表", "实际NLP任务"),
]

print("\n" + "-"*70)
print(f"{'特性':<12} {'One-Hot':<25} {'Embedding':<20}")
print("-"*70)

for feature, oh, emb in comparison:
    print(f"{feature:<12} {oh:<25} {emb:<20}")

print("\n" + "="*70)
print("核心要点")
print("="*70)

print("\n1. One-Hot Encoding:")
print("   优点: 简单、易理解")
print("   缺点: 维度高、稀疏、无法表示相似性")
print("   适用: 词汇表很小的情况（< 100个词）")

print("\n2. Word Embedding:")
print("   优点: 固定维度、密集、能表示语义相似性")
print("   缺点: 需要训练或使用预训练模型")
print("   适用: 几乎所有实际NLP任务")

print("\n3. 关键洞察:")
print("   '意思相近的词，向量也应该相近'")
print("   这是 Embedding 的核心思想！")

print("\n" + "="*70)
print("下一步学习")
print("="*70)

print("\n1. Word2Vec: 如何自动学习词向量？")
print("2. GloVe: 全局矩阵方法")
print("3. FastText: 处理生僻词和子词")
print("4. 预训练词向量: 直接使用别人训练好的模型")

print("\n运行命令：")
print("  python 02_word2vec_demo.py  # Word2Vec 演示")
print("  python 03_sentiment_analysis.py  # 情感分析实战")

print("\n" + "="*70)
print("演示完成！")
print("="*70)
