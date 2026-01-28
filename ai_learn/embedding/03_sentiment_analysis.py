"""
情感分析实战：Embedding + 逻辑回归
结合所学知识，完成一个完整的NLP项目
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import jieba
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("情感分析实战：Embedding + 逻辑回归")
print("="*70)

# ===== 第一部分：准备数据 =====
print("\n" + "="*70)
print("第一部分：准备情感分析数据集")
print("="*70)

# 示例数据（实际应该从文件读取）
data = [
    # 正面评价
    ("这部电影太好看了，演员演技很棒！", 1),
    ("非常精彩的剧情，强烈推荐！", 1),
    ("导演功力深厚，视觉效果震撼", 1),
    ("五星好评，一定要看！", 1),
    ("太好笑了，全程无尿点", 1),
    ("剧情紧凑，悬念迭起", 1),
    ("演员颜值在线，演技在线", 1),
    ("今年最好的电影，没有之一", 1),
    ("感人至深，值得一看", 1),
    ("特效制作精良，音效震撼", 1),
    ("故事情节跌宕起伏", 1),
    ("演员表演自然真实", 1),
    ("剧本写得很好", 1),
    ("全程高能，不容错过", 1),
    ("绝对的神作", 1),

    # 负面评价
    ("剧情太无聊了，看了半小时就睡着了", 0),
    ("浪费时间和金钱，不推荐", 0),
    ("演员演技尴尬", 0),
    ("剧情逻辑不通，莫名其妙", 0),
    ("特效太假，廉价感十足", 0),
    ("导演拍的是什么东西", 0),
    ("全程玩手机，太无聊了", 0),
    ("后悔来看这部电影", 0),
    ("一无是处，强烈差评", 0),
    ("剧本烂，演员烂，导演烂", 0),
    ("情节拖沓，节奏混乱", 0),
    ("看完想退票", 0),
    ("浪费时间，毫无营养", 0),
    ("演技浮夸，台词尴尬", 0),
    ("完全不值票价", 0),
]

print(f"\n数据集大小: {len(data)} 条评论")
print(f"正面评价: {sum(1 for _, label in data if label == 1)} 条")
print(f"负面评价: {sum(1 for _, label in data if label == 0)} 条")

print("\n示例数据：")
print("  正面:", data[0][0])
print("  负面:", data[15][0])

# ===== 第二部分：特征提取 - 三种方法对比 =====
print("\n" + "="*70)
print("第二部分：特征提取方法对比")
print("="*70)

# 先用简单数据训练一个小型 Word2Vec
print("\n步骤1: 训练 Word2Vec 模型（用于词嵌入）")
all_texts = [text for text, _ in data]
tokenized_texts = [list(jieba.cut(text)) for text in all_texts]

w2v_model = Word2Vec(
    sentences=tokenized_texts,
    vector_size=50,
    window=3,
    min_count=1,
    sg=0,
    epochs=100,
    seed=42
)

print(f"✓ Word2Vec 训练完成，词汇表大小: {len(w2v_model.wv)}")

print("\n" + "-"*70)
print("方法1: One-Hot + 词频统计")
print("-"*70)
print("原理: 统计每个词出现的次数")
print("缺点: 忽略词序、无法表示语义")

print("\n" + "-"*70)
print("方法2: TF-IDF")
print("-"*70)
print("原理: 词频-逆文档频率")
print("缺点: 仍然忽略词序、语义")

print("\n" + "-"*70)
print("方法3: Word Embedding（词向量平均）✓ 我们使用这个")
print("-"*70)
print("原理: 将每个词转换为向量，然后平均")
print("优点: 能捕捉语义信息")

# 方法3的实现
def text_to_embedding(text, model):
    """
    将文本转换为词向量的平均值
    """
    words = list(jieba.cut(text))
    vectors = [model.wv[word] for word in words if word in model.wv]

    if len(vectors) == 0:
        # 如果没有词在词表中，返回零向量
        return np.zeros(model.vector_size)

    # 平均所有词向量
    return np.mean(vectors, axis=0)

# 示例
sample_text = "这部电影很好看"
embedding = text_to_embedding(sample_text, w2v_model)
print(f"\n示例: '{sample_text}'")
print(f"  分词: {list(jieba.cut(sample_text))}")
print(f"  向量维度: {len(embedding)}")
print(f"  向量前5维: {embedding[:5]}")

# ===== 第三部分：准备训练数据 =====
print("\n" + "="*70)
print("第三部分：准备训练数据")
print("="*70)

# 提取特征和标签
X = np.array([text_to_embedding(text, w2v_model) for text, _ in data])
y = np.array([label for _, label in data])

print(f"\n特征矩阵形状: {X.shape}")
print(f"  {X.shape[0]} 个样本")
print(f"  每个样本 {X.shape[1]} 维")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n数据集划分:")
print(f"  训练集: {len(X_train)} 条")
print(f"  测试集: {len(X_test)} 条")

# ===== 第四部分：训练逻辑回归模型 =====
print("\n" + "="*70)
print("第四部分：训练逻辑回归模型")
print("="*70)

print("\n模型参数:")
print("  算法: LogisticRegression")
print("  正则化: L2 (C=1.0)")
print("  求解器: lbfgs")

# 训练模型
print("\n开始训练...")
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("✓ 训练完成！")

# 查看模型参数
print(f"\n模型权重形状: {model.coef_.shape}")
print(f"模型截距: {model.intercept_[0]:.4f}")

# ===== 第五部分：模型评估 =====
print("\n" + "="*70)
print("第五部分：模型评估")
print("="*70)

# 训练集评估
y_train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)

# 测试集评估
y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n准确率:")
print(f"  训练集: {train_acc:.2%}")
print(f"  测试集: {test_acc:.2%}")

# 详细分类报告
print("\n详细分类报告:")
print("-"*70)
print(classification_report(y_test, y_test_pred, target_names=["负面", "正面"]))

# 混淆矩阵
cm = confusion_matrix(y_test, y_test_pred)
print("\n混淆矩阵:")
print("-"*70)
print("           预测负面  预测正面")
print(f"真实负面:    {cm[0][0]:>2}       {cm[0][1]:>2}")
print(f"真实正面:    {cm[1][0]:>2}       {cm[1][1]:>2}")

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["负面", "正面"],
            yticklabels=["负面", "正面"])
plt.title('混淆矩阵 - 情感分类', fontsize=14, fontweight='bold')
plt.ylabel('真实标签', fontsize=12)
plt.xlabel('预测标签', fontsize=12)
plt.tight_layout()
plt.savefig('sentiment_confusion_matrix.png', dpi=100, bbox_inches='tight')
print("\n✓ 混淆矩阵图已保存: sentiment_confusion_matrix.png")
plt.close()

# ===== 第六部分：预测新评论 =====
print("\n" + "="*70)
print("第六部分：预测新评论")
print("="*70)

new_reviews = [
    "这部电影非常精彩，值得推荐！",
    "太无聊了，后悔来看",
    "演员演技很棒，剧情也很好",
    "剧情混乱，浪费时间",
    "五星好评，强烈推荐！",
    "完全不推荐，太烂了",
]

print("\n测试评论:")
print("-"*70)

for i, review in enumerate(new_reviews, 1):
    # 转换为向量
    embedding = text_to_embedding(review, w2v_model).reshape(1, -1)

    # 预测
    pred = model.predict(embedding)[0]
    prob = model.predict_proba(embedding)[0]

    sentiment = "😊 正面" if pred == 1 else "😞 负面"
    confidence = prob[pred] * 100

    print(f"\n{i}. {review}")
    print(f"   预测: {sentiment}")
    print(f"   置信度: {confidence:.1f}%")
    print(f"   概率分布: 负面 {prob[0]:.1%} | 正面 {prob[1]:.1%}")

# ===== 第七部分：分析预测错误 =====
print("\n" + "="*70)
print("第七部分：分析预测结果")
print("="*70)

# 获取所有预测概率
y_test_prob = model.predict_proba(X_test)[:, 1]

print("\n测试集预测详情:")
print("-"*70)
print(f"{'评论':<30} {'真实':<8} {'预测':<8} {'正确概率'}")
print("-"*70)

for i, (text, true_label) in enumerate([data[idx] for idx in range(len(data)) if idx % 7 == 0][:5]):
    emb = text_to_embedding(text, w2v_model).reshape(1, -1)
    pred = model.predict(emb)[0]
    prob = model.predict_proba(emb)[0][pred]

    true_sentiment = "正面" if true_label == 1 else "负面"
    pred_sentiment = "正面" if pred == 1 else "负面"
    correct = "✓" if pred == true_label else "✗"

    print(f"{text[:28]:<30} {true_sentiment:<8} {pred_sentiment:<8} {prob:.1%} {correct}")

# ===== 第八部分：可视化词向量 =====
print("\n" + "="*70)
print("第八部分：可视化情感词向量")
print("="*70)

from sklearn.decomposition import PCA

# 选择一些情感词
positive_words = ["好看", "精彩", "推荐", "棒", "好", "优秀"]
negative_words = ["无聊", "烂", "差", "垃圾", "后悔", "差评"]

# 提取词向量
all_words = positive_words + negative_words
vectors = []
labels = []
colors = []

for word in positive_words:
    if word in w2v_model.wv:
        vectors.append(w2v_model.wv[word])
        labels.append(word)
        colors.append('green')

for word in negative_words:
    if word in w2v_model.wv:
        vectors.append(w2v_model.wv[word])
        labels.append(word)
        colors.append('red')

vectors = np.array(vectors)

# PCA降维
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

# 可视化
plt.figure(figsize=(10, 8))
for i, (x, y) in enumerate(vectors_2d):
    plt.scatter(x, y, c=colors[i], s=200, alpha=0.6,
               edgecolors='black', linewidth=1.5)
    plt.annotate(labels[i], (x, y), xytext=(5, 5),
                textcoords='offset points', fontsize=12, fontweight='bold')

# 添加图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
           markersize=12, label='正面词'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markersize=12, label='负面词'),
]
plt.legend(handles=legend_elements, loc='best', fontsize=11)

plt.title('情感词向量可视化 (PCA降维)', fontsize=14, fontweight='bold')
plt.xlabel('维度 1 (PCA)', fontsize=12)
plt.ylabel('维度 2 (PCA)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('sentiment_word_vectors.png', dpi=100, bbox_inches='tight')
print("✓ 情感词向量图已保存: sentiment_word_vectors.png")
plt.close()

# ===== 第九部分：总结与改进 =====
print("\n" + "="*70)
print("总结与改进建议")
print("="*70)

print("\n✓ 我们完成了什么:")
print("  1. 使用 Word2Vec 将词转换为向量")
print("  2. 用词向量平均表示句子")
print("  3. 训练逻辑回归分类器")
print("  4. 预测新评论的情感")

print("\n📊 模型性能:")
print(f"  测试集准确率: {test_acc:.2%}")
print("  注: 由于数据集很小（仅30条），实际性能不代表真实水平")

print("\n🔧 改进方向:")
print("\n1. 数据层面:")
print("   - 使用更大的数据集（至少几千条）")
print("   - 数据清洗：去除标点、停用词")
print("   - 数据增强：同义词替换、回译")

print("\n2. 特征工程:")
print("   - 使用预训练词向量（腾讯、北师大）")
print("   - 尝试不同的句子表示方法")
print("   - 添加TF-IDF加权")

print("\n3. 模型优化:")
print("   - 调整正则化参数 C")
print("   - 尝试其他算法（SVM、随机森林）")
print("   - 使用深度学习（LSTM、BERT）")

print("\n4. 高级技术:")
print("   - Sentence-BERT（句子嵌入）")
print("   - 注意力机制（Attention）")
print("   - 预训练语言模型（BERT、GPT）")

print("\n" + "="*70)
print("下一步学习")
print("="*70)

print("\n推荐项目:")
print("  1. 使用真实的电影评论数据集（IMDB、豆瓣）")
print("  2. 尝试多分类问题（1-5星评分）")
print("  3. 学习更复杂的模型（LSTM、Transformer）")

print("\n推荐资源:")
print("  - IMDB 电影评论数据集")
print("  - 豆瓣电影评论（爬虫获取）")
print("  - 中文情感分析数据集（GitHub搜索）")

print("\n" + "="*70)
print("演示完成！")
print("="*70)

print("\n💡 提示:")
print("  这个项目整合了两个重要概念:")
print("  1. Embedding - 将文本转换为数字向量")
print("  2. 逻辑回归 - 分类算法")
print("  结合起来就能完成情感分析任务！")
