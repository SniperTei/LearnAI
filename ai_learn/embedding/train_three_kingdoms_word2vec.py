"""
基于三国文本训练 Word2Vec 模型
从头开始学习 Embedding
"""

import jieba
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("三国文本 Word2Vec 训练")
print("=" * 70)

# ===== 第一步：加载并预处理文本 =====
print("\n【第一步】加载三国文本...")

with open('three_kingdoms.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"✓ 文本加载成功，总字数: {len(text):,}")

# 分词（按句子分词）
print("\n正在分词...")
sentences = []
for line in text.split('\n'):
    if line.strip():
        words = list(jieba.cut(line))
        # 过滤掉单字和标点，保留有意义的词
        words = [w for w in words if len(w) > 1]
        if words:
            sentences.append(words)

print(f"✓ 分词完成，句子数: {len(sentences):,}")
print(f"\n示例句子（前3句）:")
for i, sent in enumerate(sentences[:3], 1):
    print(f"  {i}. {' '.join(sent[:10])}...")

# ===== 第二步：训练 Word2Vec 模型 =====
print("\n" + "=" * 70)
print("【第二步】训练 Word2Vec 模型")
print("=" * 70)

print("\n训练参数:")
print("  - vector_size: 100 (词向量维度)")
print("  - window: 5     (上下文窗口大小)")
print("  - min_count: 2  (最少出现次数)")
print("  - sg: 0         (使用 CBOW 算法)")
print("  - epochs: 100   (训练轮数)")

print("\n开始训练...")
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    sg=0,
    epochs=100,
    seed=42
)

print(f"✓ 训练完成！词汇表大小: {len(model.wv):,} 个词")

# ===== 第三步：探索词向量 =====
print("\n" + "=" * 70)
print("【第三步】探索词向量")
print("=" * 70)

# 1. 查看词向量
print("\n" + "-" * 70)
print("1. 查看词向量示例")
print("-" * 70)

key_words = ["刘备", "诸葛亮", "曹操", "关羽", "张飞"]
for word in key_words:
    if word in model.wv:
        vector = model.wv[word]
        print(f"\n{word}:")
        print(f"  向量维度: {len(vector)}")
        print(f"  前5维: {vector[:5]}")

# 2. 计算词相似度
print("\n" + "-" * 70)
print("2. 人物相似度")
print("-" * 70)

characters = ["刘备", "关羽", "张飞", "诸葛亮", "曹操", "孙权", "周瑜", "吕布"]
print(f"\n{'人物A':<8} {'人物B':<8} {'相似度':<10} {'可视化'}")
print("-" * 70)

for i in range(len(characters)):
    for j in range(i+1, len(characters)):
        word1, word2 = characters[i], characters[j]
        if word1 in model.wv and word2 in model.wv:
            sim = model.wv.similarity(word1, word2)
            bar = "█" * int(sim * 30)
            print(f"{word1:<8} {word2:<8} {sim:.3f}     {bar}")

# 3. 找最相似的人
print("\n" + "-" * 70)
print("3. 找最相似的人物")
print("-" * 70)

query_people = ["刘备", "诸葛亮", "曹操", "关羽"]

for person in query_people:
    if person in model.wv:
        similar = model.wv.most_similar(person, topn=5)
        print(f"\n与 '{person}' 最相似的人:")
        for name, score in similar:
            bar = "█" * int(score * 30)
            print(f"  {name:<8} {score:.3f} {bar}")

# ===== 第四步：词向量运算 =====
print("\n" + "=" * 70)
print("【第四步】词向量运算")
print("=" * 70)

# 刘备 + 曹操 - 张飞
print("\n运算: 刘备 + 曹操 - 张飞 = ?")
print("-" * 70)

try:
    result = model.wv.most_similar(
        positive=["刘备", "曹操"],
        negative=["张飞"],
        topn=10
    )
    for i, (word, score) in enumerate(result, 1):
        bar = "█" * int(score * 30)
        print(f"  {i:2}. {word:<8} {score:.3f} {bar}")
except Exception as e:
    print(f"  无法完成运算: {e}")

# ===== 第五步：可视化词向量 =====
print("\n" + "=" * 70)
print("【第五步】可视化词向量 (PCA降维)")
print("=" * 70)

# 选择主要人物
main_characters = [
    "刘备", "关羽", "张飞", "诸葛亮", "赵云", "黄忠", "姜维",  # 蜀
    "曹操", "司马懿", "董卓", "袁绍",  # 魏
    "孙权", "周瑜", "陆逊",  # 吴
    "吕布"  # 其他
]

# 提取词向量
vectors = []
valid_chars = []
for char in main_characters:
    if char in model.wv:
        vectors.append(model.wv[char])
        valid_chars.append(char)

vectors = np.array(vectors)

# PCA降维到2维
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# 根据阵营分类
shu = ["刘备", "关羽", "张飞", "诸葛亮", "赵云", "黄忠", "姜维"]
wei = ["曹操", "司马懿", "董卓", "袁绍"]
wu = ["孙权", "周瑜", "陆逊"]
other = ["吕布"]

colors = []
sizes = []
for char in valid_chars:
    if char in shu:
        colors.append('green')
        sizes.append(200)
    elif char in wei:
        colors.append('blue')
        sizes.append(200)
    elif char in wu:
        colors.append('red')
        sizes.append(200)
    else:
        colors.append('gray')
        sizes.append(150)

# 绘图
plt.figure(figsize=(12, 10))

for i, (x, y) in enumerate(vectors_2d):
    plt.scatter(x, y, c=colors[i], s=sizes[i], alpha=0.6,
               edgecolors='black', linewidth=1.5)
    plt.annotate(valid_chars[i], (x, y), xytext=(5, 5),
                textcoords='offset points', fontsize=12, fontweight='bold')

# 添加图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
           markersize=12, label='蜀国'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
           markersize=12, label='魏国'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markersize=12, label='吴国'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=12, label='其他'),
]
plt.legend(handles=legend_elements, loc='best', fontsize=11)

plt.title('三国人物词向量可视化 (Word2Vec + PCA)', fontsize=14, fontweight='bold')
plt.xlabel('维度 1 (PCA)', fontsize=12)
plt.ylabel('维度 2 (PCA)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('word2vec_visualization.png', dpi=100, bbox_inches='tight')
print("\n✓ 可视化图已保存: word2vec_visualization.png")
plt.close()

# ===== 第六步：保存模型 =====
print("\n" + "=" * 70)
print("【第六步】保存模型")
print("=" * 70)

model.save("word2vec.model")
print("✓ 模型已保存: word2vec.model")

print("\n下次加载方法:")
print("  from gensim.models import Word2Vec")
print("  model = Word2Vec.load('word2vec.model')")

# ===== 总结 =====
print("\n" + "=" * 70)
print("训练完成！")
print("=" * 70)

print("\n✓ 完成的工作:")
print("  1. 加载三国文本 (60万+ 字)")
print("  2. 中文分词（jieba）")
print("  3. 训练 Word2Vec 模型 (词汇量: 16,977)")
print("  4. 探索词向量（相似度、类比）")
print("  5. 可视化人物关系")

print("\n💡 你可以继续:")
print("  - 加载模型: model = Word2Vec.load('word2vec.model')")
print("  - 查询相似词: model.wv.most_similar('刘备')")
print("  - 词向量运算: model.wv.most_similar(positive=['刘备','曹操'], negative=['张飞'])")

print("\n" + "=" * 70)
