"""
Word2Vec 实战演示 - 中文词向量训练与使用
边学边练：从原理到实战
"""

import numpy as np
from gensim.models import Word2Vec
import jieba
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("Word2Vec 实战演示：让机器理解词语的含义")
print("="*70)

# ===== 第一部分：准备中文语料 =====
print("\n" + "="*70)
print("第一部分：准备中文语料")
print("="*70)

# 小型中文语料（用于演示）
corpus = [
    ["我", "爱", "北京", "天安门"],
    ["北京", "是", "中国", "的", "首都"],
    ["中国", "有", "五千年", "的", "历史"],
    ["长城", "是", "中国", "的", "象征"],
    ["我", "喜欢", "吃", "北京", "烤鸭"],
    ["北京烤鸭", "非常", "好吃"],
    ["长城", "位于", "北京", "郊区"],
    ["天安门", "广场", "很", "壮观"],
    ["我", "去", "过", "北京", "很多", "次"],
    ["北京", "的", "秋天", "很", "美"],
    ["中国", "的", "经济", "发展", "很快"],
    ["长城", "是", "世界", "文化遗产"],
]

print("\n语料示例（前3句）：")
for i, sent in enumerate(corpus[:3], 1):
    print(f"  {i}. {' '.join(sent)}")

print(f"\n总句子数: {len(corpus)}")
print(f"总词汇数: {sum(len(s) for s in corpus)}")

# ===== 第二部分：训练 Word2Vec 模型 =====
print("\n" + "="*70)
print("第二部分：训练 Word2Vec 模型")
print("="*70)

print("\nWord2Vec 参数说明：")
print("  - vector_size: 词向量维度（默认100）")
print("  - window: 上下文窗口大小（默认5）")
print("  - min_count: 最小词频，低于此值忽略（默认5）")
print("  - sg: 1=Skip-gram, 0=CBOW（默认0）")
print("  - epochs: 训练轮数（默认5）")

# 训练模型
print("\n开始训练...")
model = Word2Vec(
    sentences=corpus,
    vector_size=50,      # 词向量维度
    window=3,            # 上下文窗口
    min_count=1,         # 最小词频（演示用，设为1）
    sg=0,                # 0=CBOW, 1=Skip-gram
    epochs=100,          # 训练轮数（演示用，增加轮数）
    seed=42
)

print("✓ 训练完成！")
print(f"✓ 词汇表大小: {len(model.wv)} 个词")

# 查看词向量
print("\n" + "-"*70)
print("词向量示例（前5个词）：")
print("-"*70)

words = list(model.wv.key_to_index.keys())[:5]
print(f"\n{'词语':<8} {'词向量（前5维）'}")
print("-"*70)

for word in words:
    vector = model.wv[word]
    vector_str = ", ".join([f"{v:.2f}" for v in vector[:5]])
    print(f"{word:<8} [{vector_str}, ...]")

# ===== 第三部分：探索词向量 =====
print("\n" + "="*70)
print("第三部分：探索词向量")
print("="*70)

# 1. 计算词相似度
print("\n" + "-"*70)
print("1. 计算词语相似度")
print("-"*70)

word_pairs = [
    ("北京", "中国"),
    ("北京", "长城"),
    ("北京", "天安门"),
    ("北京", "我"),
    ("长城", "中国"),
    ("我", "喜欢"),
]

for word1, word2 in word_pairs:
    if word1 in model.wv and word2 in model.wv:
        sim = model.wv.similarity(word1, word2)
        bar = "█" * int(sim * 30)
        print(f"  '{word1}' vs '{word2}': {sim:.3f} {bar}")

# 2. 找最相似的词
print("\n" + "-"*70)
print("2. 找最相似的词")
print("-"*70)

query_words = ["北京", "长城", "中国", "吃"]

for word in query_words:
    if word in model.wv:
        similar_words = model.wv.most_similar(word, topn=3)
        print(f"\n  与 '{word}' 最相似的词：")
        for similar_word, score in similar_words:
            bar = "█" * int(score * 30)
            print(f"    {similar_word:<8} {score:.3f} {bar}")

# 3. 词向量类比
print("\n" + "-"*70)
print("3. 词向量类比（类比推理）")
print("-"*70)

print("\n  经典例子：国王 - 男人 + 女人 ≈ 王后")
print("  我们的语料太小，尝试简单类比：")

# 尝试简单类比
print("\n  尝试: 北京 - 中国 + 长城 ≈ ?")
try:
    result = model.wv.most_similar(
        positive=["北京", "长城"],
        negative=["中国"],
        topn=3
    )
    for word, score in result:
        print(f"    {word:<8} {score:.3f}")
except Exception as e:
    print(f"    语料太小，无法完成类比 😅")

print("\n  解释: '北京' 减去 '中国' 的部分特征")
print("        加上 '长城' 的特征")
print("        看看结果接近什么词")

# ===== 第四部分：Word2Vec 两种模式对比 =====
print("\n" + "="*70)
print("第四部分：CBOW vs Skip-gram 对比")
print("="*70)

print("\nCBOW (Continuous Bag-of-Words):")
print("  - 根据周围词预测中心词")
print("  - 例: '今天天气_不错' → 猜'真'")
print("  - 优点: 训练快，适合常见词")
print("  - 缺点: 对生僻词效果差")

print("\nSkip-gram:")
print("  - 根据中心词预测周围词")
print("  - 例: '真' → 猜['今天', '天气', '不错']")
print("  - 优点: 对生僻词效果好，能学到更多信息")
print("  - 缺点: 训练慢")

# 对比训练（小型演示）
print("\n" + "-"*70)
print("对比训练（相同参数，不同模式）")
print("-"*70)

# CBOW
model_cbow = Word2Vec(
    sentences=corpus,
    vector_size=50,
    window=3,
    min_count=1,
    sg=0,  # CBOW
    epochs=50,
    seed=42
)

# Skip-gram
model_sg = Word2Vec(
    sentences=corpus,
    vector_size=50,
    window=3,
    min_count=1,
    sg=1,  # Skip-gram
    epochs=50,
    seed=42
)

print("✓ CBOW 模型训练完成")
print("✓ Skip-gram 模型训练完成")

# 对比相似度
print("\n对比词相似度（以'北京'为例）：")
test_words = ["中国", "长城", "天安门"]
print(f"\n{'对比词':<8} {'CBOW相似度':<15} {'Skip-gram相似度'}")
print("-"*70)

for word in test_words:
    if word in model_cbow.wv and word in model_sg.wv:
        sim_cbow = model_cbow.wv.similarity("北京", word)
        sim_sg = model_sg.wv.similarity("北京", word)
        print(f"{word:<8} {sim_cbow:<15.3f} {sim_sg:.3f}")

print("\n注意: 由于语料很小，差异可能不明显")
print("实际应用中，Skip-gram 通常对生僻词效果更好")

# ===== 第五部分：词向量可视化 =====
print("\n" + "="*70)
print("第五部分：词向量可视化（降维到2D）")
print("="*70)

from sklearn.decomposition import PCA

# 获取所有词向量
words = list(model.wv.key_to_index.keys())
vectors = np.array([model.wv[w] for w in words])

# PCA降维到2维
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# 可视化
plt.figure(figsize=(12, 10))

# 绘制散点图
for i, word in enumerate(words):
    x, y = vectors_2d[i]
    plt.scatter(x, y, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    plt.annotate(word, (x, y), xytext=(5, 5), textcoords='offset points',
                fontsize=10, alpha=0.8)

plt.title('Word2Vec 词向量可视化 (PCA降维)', fontsize=14, fontweight='bold')
plt.xlabel('维度 1 (PCA)', fontsize=12)
plt.ylabel('维度 2 (PCA)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('word2vec_visualization.png', dpi=100, bbox_inches='tight')
print("✓ 可视化图已保存: word2vec_visualization.png")
plt.close()

# ===== 第六部分：实际应用 - 句子相似度 =====
print("\n" + "="*70)
print("第六部分：实际应用 - 计算句子相似度")
print("="*70)

sentences = [
    "我爱北京",
    "我喜欢北京",
    "北京是中国的首都",
    "长城在北京郊区",
]

print("\n示例句子：")
for i, sent in enumerate(sentences, 1):
    print(f"  {i}. {sent}")

# 句子向量化方法：词向量平均
def sentence_vector(sentence, model):
    words = list(jieba.cut(sentence))
    vectors = [model.wv[w] for w in words if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

print("\n" + "-"*70)
print("句子向量化方法：词向量平均")
print("-"*70)

sentence_vectors = [sentence_vector(s, model) for s in sentences]

print("\n句子相似度矩阵：")
print("-"*70)

# 打印表头
print(f"{'':12s}", end="")
for i in range(len(sentences)):
    print(f"句子{i+1:>6d}", end="")
print()

print("-"*70)

# 打印相似度矩阵
for i in range(len(sentences)):
    print(f"句子{i+1:<6d}", end="")
    for j in range(len(sentences)):
        if i == j:
            print(f"  1.000", end="")
        else:
            sim = cosine_similarity([sentence_vectors[i]], [sentence_vectors[j]])[0][0]
            print(f"  {sim:.3f}", end="")
    print()

print("\n观察:")
print("  - 句子1和句子2相似度应该较高（都是表达喜爱）")
print("  - 句子3和句子4相似度也较高（都与北京、中国、长城相关）")

# ===== 第七部分：总结与建议 =====
print("\n" + "="*70)
print("总结：Word2Vec 核心要点")
print("="*70)

print("\n1. 核心思想：")
print("   '一个词的意思由它周围的词决定'")
print("   出现在相似语境中的词，向量应该相似")

print("\n2. 两种模式：")
print("   CBOW: 快、适合常见词")
print("   Skip-gram: 慢、对生僻词效果好")

print("\n3. 参数调优建议：")
print("   vector_size: 小数据50-100，大数据200-300")
print("   window: 一般3-5，根据任务调整")
print("   min_count: 忽略低频词，默认5")
print("   epochs: 小数据多轮次，大数据少轮次")

print("\n4. 实际应用：")
print("   - 情感分析")
print("   - 文档相似度")
print("   - 推荐系统")
print("   - 机器翻译")

print("\n5. 注意事项：")
print("   ⚠️  我们的演示语料太小，实际需要大量文本（至少百万词）")
print("   ⚠️  中文需要分词（jieba、pkuseg等）")
print("   ⚠️  实际项目优先使用预训练词向量")

print("\n" + "="*70)
print("下一步学习")
print("="*70)

print("\n1. 使用更大规模的语料训练")
print("2. 尝试预训练词向量（腾讯、北师大等）")
print("3. 学习 GloVe、FastText")
print("4. 实战：情感分析、文本分类")

print("\n运行命令：")
print("  python 03_sentiment_analysis.py  # 情感分析实战")

print("\n推荐资源：")
print("  - 腾讯词向量: https://ai.tencent.com/ailab/nlp/embedding.html")
print("  - 北京师范大学中文词向量: GitHub搜索 'Chinese-Word-Vectors'")

print("\n" + "="*70)
print("演示完成！")
print("="*70)
