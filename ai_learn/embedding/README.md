# Embedding（词嵌入）入门教程

> 让机器理解语言的魔法 - 从 One-Hot 到 Word2Vec

## 📚 项目简介

Embedding 是现代自然语言处理（NLP）的基石。通过这个项目，你将理解：
- 为什么要用词向量
- Word2Vec、GloVe、FastText 的原理
- 如何用代码实现和应用词向量
- 词向量的可视化与理解

### 学习目标

- ✅ 理解为什么不能直接用文本训练模型
- ✅ 掌握 One-Hot Encoding 的局限性
- ✅ 理解 Word Embedding 的核心思想
- ✅ 学会使用 Word2Vec、GloVe
- ✅ 完成情感分析实战项目

---

## 🎯 核心概念速览

| 概念 | 通俗解释 | 难度 |
|------|---------|------|
| **One-Hot 编码** | 给每个词发一个身份证号 | ⭐ |
| **词嵌入（Embedding）** | 把词压缩成数字列表，意思相近的词数字也相近 | ⭐⭐⭐ |
| **Word2Vec** | 神奇算法，自动学习词的意思 | ⭐⭐⭐⭐ |
| **余弦相似度** | 衡量两个词有多像 | ⭐⭐ |
| **预训练词向量** | 别人已经训练好的词向量字典 | ⭐⭐ |

---

## 📁 文件说明

```
embedding/
├── README.md                    # 本文档
├── 01_one_hot_vs_embedding.py   # One-Hot vs Embedding 对比
├── 02_word2vec_demo.py          # Word2Vec 演示（中文示例）
├── 03_sentiment_analysis.py     # 情感分析实战（Embedding + 逻辑回归）
├── 04_visualization.py          # 词向量可视化（降维到2D）
└── data/                        # 数据文件夹
    ├── small_corpus.txt         # 小型中文语料
    └── movie_reviews.csv        # 电影评论数据集
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install numpy gensim scikit-learn matplotlib jieba
```

### 2. 运行演示

```bash
# One-Hot vs Embedding 对比
python 01_one_hot_vs_embedding.py

# Word2Vec 演示
python 02_word2vec_demo.py

# 词向量可视化
python 04_visualization.py

# 情感分析实战
python 03_sentiment_analysis.py
```

---

## 🎓 核心知识点

### 第一部分：为什么需要 Embedding？

#### 问题1：机器不认识文字

```
输入："这部电影很好看"
机器：??? 这些是什么？

机器只认识数字！
```

#### 方案1：One-Hot 编码（初学者想法）

```
假设我们有 5 个词：
["我", "爱", "电影", "好看", "很"]

给每个词发一个身份证：
我     → [1, 0, 0, 0, 0]
爱     → [0, 1, 0, 0, 0]
电影   → [0, 0, 1, 0, 0]
好看   → [0, 0, 0, 1, 0]
很     → [0, 0, 0, 0, 1]

一句话："我 爱 电影"
        → [1,0,0,0,0] + [0,1,0,0,0] + [0,0,1,0,0]
```

**问题：**
1. ❌ 太稀疏了：词汇表有 10 万个词，每个词要用 10 万个数字表示！
2. ❌ 没有相似性："猫"和"狗"应该很像，但 One-Hot 完全看不出来
3. ❌ 浪费空间：大部分位置都是 0

#### 方案2：Word Embedding（聪明做法）

**核心思想：意思相近的词，数字向量也应该相近**

```
用固定长度的短向量表示每个词（比如 50 个数字）：

"猫"  → [0.8, -0.2, 0.5, 0.9, ...]  (50个数字)
"狗"  → [0.7, -0.1, 0.6, 0.8, ...]  (50个数字)
"汽车" → [-0.5, 0.9, -0.3, 0.2, ...] (50个数字)

观察：
- "猫" 和 "狗" 的向量很接近（都是动物）
- "猫" 和 "汽车" 的向量差很远
```

**类比理解：**

想象你在给商品贴标签：
- 苹果：红色、圆形、甜、水果 → [1, 1, 1, 1]
- 橙子：橙色、圆形、甜、水果 → [0.8, 1, 1, 1]
- 篮球：橙色、圆形、不甜、运动用品 → [0.8, 1, 0, 0]

苹果和橙子的向量很像！这就是 Embedding 的思想。

---

### 第二部分：Word2Vec - 神奇的算法

#### 直观理解

Word2Vec 的核心思想：**"一个词的意思由它周围的词决定"**

```
句子1："今天天气真不错"
句子2："今天天气很好"
句子3："今天天气太棒了"

观察：
"不错"、"好"、"棒" 都出现在相似的语境
→ 它们的向量应该相似
```

#### 两种训练方式

**方式1：CBOW（Continuous Bag-of-Words）**

```
问题：根据周围的词预测中间的词

输入："今天天气_不错"
目标：猜出空格是"真"

学会："真" 经常出现在"今天天气"和"不错"之间
```

**方式2：Skip-gram**

```
问题：根据中间的词预测周围的词

输入："真"
目标：猜出周围可能是 ["今天", "天气", "不错"]

学会："真" 的周围经常出现这些词
```

#### 魔法效果：词向量运算

训练好的词向量可以运算！

```
国王 - 男人 + 女人 ≈ 王后
北京 - 中国 + 法国 ≈ 巴黎

类比：
"国王" 减去 "男人" 的特征（男性）
加上 "女人" 的特征（女性）
得到 "王后"！
```

---

### 第三部分：相似度计算

#### 余弦相似度

**问题：怎么衡量两个词向量有多像？**

```
词A向量：[1, 2, 3]
词B向量：[1, 2, 3]     ← 完全相同，相似度 = 1.0
词C向量：[2, 4, 6]     ← 方向相同，相似度 = 1.0
词D向量：[-1, -2, -3]  ← 方向相反，相似度 = -1.0
词E向量：[3, 1, 2]     ← 不太像，相似度 = 0.7
```

**直觉理解：**
- 两个人朝同一个方向走 → 相似度高
- 两个人朝相反方向走 → 相似度低
- 方向完全相同，距离不管多远，相似度都是 1

**公式（了解即可）：**
```
相似度 = (A·B) / (|A| × |B|)
```

---

## 💡 实际应用

### 应用1：情感分析

```
输入："这部电影太好看了！"
步骤：
1. 分词：["这", "部", "电影", "太", "好看", "了"]
2. 查词向量：每个词 → 50维向量
3. 加平均：得到整句话的向量
4. 分类：用逻辑回归判断 [正面 / 负面]
```

### 应用2：搜索推荐

```
用户搜索："好吃的披萨"
步骤：
1. 找到 "好吃" 和 "披萨" 的词向量
2. 计算与所有文档的相似度
3. 返回最相似的文档
```

### 应用3：机器翻译

```
英文 "cat" → 词向量A → 找到中文中最接近的 → "猫"
```

---

## 🎯 实战项目

### 项目1：Word2Vec 训练中文词向量

```python
from gensim.models import Word2Vec
import jieba

# 准备语料
sentences = [
    ["我", "爱", "北京", "天安门"],
    ["北京", "是", "中国", "的", "首都"],
    ["我", "喜欢", "吃", "北京", "烤鸭"]
]

# 训练模型
model = Word2Vec(sentences, vector_size=50, min_count=1)

# 查看词向量
print(model.wv['北京'])  # 输出50维向量

# 计算相似度
print(model.wv.similarity('北京', '中国'))  # 0.85
print(model.wv.similarity('北京', '吃'))    # 0.12

# 找最相似的词
print(model.wv.most_similar('北京'))
# 输出：[('中国', 0.85), ('首都', 0.78), ...]
```

### 项目2：情感分析（Embedding + 逻辑回归）

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# 数据
reviews = [
    ("这部电影太好看了！", 1),
    ("剧情很无聊", 0),
    ("演员演技很棒", 1),
    ("浪费时间", 0)
]

# 特征提取：用词向量平均表示句子
def get_sentence_vector(sentence):
    words = jieba.lcut(sentence)
    vectors = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vectors, axis=0)

X = [get_sentence_vector(s) for s, _ in reviews]
y = [label for _, label in reviews]

# 训练分类器
clf = LogisticRegression()
clf.fit(X, y)

# 预测
test = "这部电影很棒"
vec = get_sentence_vector(test)
print(clf.predict([vec]))  # 输出：[1] (正面)
```

---

## 🔑 关键概念总结

### One-Hot vs Embedding

| 特性 | One-Hot | Embedding |
|------|---------|-----------|
| **维度** | 词汇表大小（10万+） | 固定（50-300） |
| **稀疏性** | 极度稀疏（大部分是0） | 密集向量 |
| **相似性** | 无法表示 | 自动学习 |
| **空间效率** | 浪费 | 高效 |
| **类比推理** | 不支持 | 支持 |

### 预训练词向量

**什么是预训练？**

别人用海量数据（维基百科、新闻语料）训练好的词向量字典，直接拿来用！

**常用预训练模型：**

| 模型 | 训练数据 | 维度 | 特点 |
|------|---------|------|------|
| **Word2Vec (Google)** | Google News | 300 | 英文，经典 |
| **GloVe (Stanford)** | Wikipedia + Gigaword | 50-300 | 英文，效果好 |
| **FastText (Facebook)** | Wikipedia | 300 | 支持子词，处理OOV |
| **腾讯词向量** | 微信公众号文章 | 200/200 | 中文，推荐 |
| **北师大词向量** | 微博数据 | - | 中文，口语化 |

**使用预训练词向量：**

```python
import gensim.downloader as api

# 下载预训练模型
model = api.load('word2vec-google-news-300')

# 直接使用
print(model.similarity('cat', 'dog'))  # 0.76
print(model.most_similar('cat'))       # [('dog', 0.76), ...]
```

---

## 🤔 常见问题

### Q1: 词向量维度选多少？

**A:**
- 小数据集 / 快速实验：50-100 维
- 一般任务：200-300 维
- 大数据集 / 复杂任务：300-500 维
- BERT 等 Transformer：768-1024 维

**权衡：**
- 维度太低：信息不够
- 维度太高：过拟合，计算慢

### Q2: 如果遇到不在词表里的词怎么办？

**A:**
1. 用 FastText（支持子词）
2. 用随机向量初始化
3. 用 UNK 标记代替
4. 用字符级 Embedding

### Q3: 中文和英文有什么不同？

**A:**
| 特性 | 英文 | 中文 |
|------|------|------|
| 分词 | 空格分开 | 需要分词工具（jieba） |
| 语义单位 | 单词 | 字/词 |
| 预训练模型 | 更多 | 相对少但够用 |

### Q4: Word2Vec vs GloVe vs FastText？

| 特性 | Word2Vec | GloVe | FastText |
|------|----------|-------|----------|
| **训练方式** | 预测 | 共现矩阵 | 预测 + 子词 |
| **速度** | 快 | 慢 | 中等 |
| **OOV处理** | ❌ | ❌ | ✅ |
| **推荐场景** | 通用 | 大语料 | 有生僻词 |

---

## 📈 下一步学习

### 算法路线
1. ✅ 逻辑回归
2. ✅ Embedding ← **当前**
3. ⬜ RNN、LSTM（处理序列）
4. ⬜ Transformer（Attention机制）
5. ⬜ BERT、GPT（大模型时代）

### 技能提升
- ✅ 词向量基础
- ⬜ 句子嵌入（Sentence-BERT）
- ⬜ 文档嵌入（Doc2Vec）
- ⬜ 上下文嵌入（ELMo、BERT）

---

## 📝 总结

### 你已经学会：
- ✅ One-Hot vs Embedding 的区别
- ✅ Word2Vec 的核心思想
- ✅ 词向量的应用场景
- ✅ 如何使用预训练词向量

### 记住：
- 🎯 **词向量 = 词语的数字化表示**
- 🎯 **意思相近的词，向量也相近**
- 🎯 **优先使用预训练模型，别重复造轮子**
- 🎯 **理解比推导重要** - 会用比会推导更实际

### 核心公式速查

```
# 余弦相似度
similarity = cos(A, B) = (A·B) / (|A| × |B|)

# Word2Vec 两种模式
# CBOW: 上下文 → 中心词
# Skip-gram: 中心词 → 上下文
```

---

## 🔗 相关资源

### 推荐阅读
- [Word2Vec 原论文](https://arxiv.org/abs/1301.3781)
- [GloVe 原论文](https://nlp.stanford.edu/pubs/glove.pdf)
- [Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781)

### 在线演示
- [Embedding 可视化](https://projector.tensorflow.org/)
- [词向量类比测试](https://www.tensorflow.org/embedding_vis)

### 中文资源
- [腾讯词向量](https://ai.tencent.com/ailab/nlp/embedding.html)
- [北京师范大学中文词向量](https://github.com/Embedding/Chinese-Word-Vectors)

---

## 📧 反馈与交流

如有问题或建议，欢迎提 Issue 或 Pull Request！

---

## 📄 许可证

MIT License - 自由使用和分享

---

**Happy Learning! 🎉**

> "词向量让机器第一次真正'理解'了语言的意义，而不仅仅是匹配关键词。"
