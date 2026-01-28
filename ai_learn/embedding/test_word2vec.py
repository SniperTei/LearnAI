"""
词向量运算测试
加载训练好的模型进行各种有趣的向量运算
"""

from gensim.models import Word2Vec

# 加载模型
model = Word2Vec.load("word2vec.model")

print("=" * 70)
print("词向量运算测试")
print("=" * 70)

# 运算示例
calculations = [
    {
        "name": "刘备 + 曹操 - 张飞",
        "positive": ["刘备", "曹操"],
        "negative": ["张飞"],
        "desc": "两主公的特征减去武将特征"
    },
    {
        "name": "诸葛亮 + 曹操 - 周瑜",
        "positive": ["诸葛亮", "曹操"],
        "negative": ["周瑜"],
        "desc": "两大谋略家减去中间人"
    },
    {
        "name": "关羽 + 张飞 - 刘备",
        "positive": ["关羽", "张飞"],
        "negative": ["刘备"],
        "desc": "两兄弟减去大哥"
    },
    {
        "name": "孙权 + 刘备 - 曹操",
        "positive": ["孙权", "刘备"],
        "negative": ["曹操"],
        "desc": "吴蜀之和减去魏"
    },
]

for calc in calculations:
    print(f"\n{'=' * 70}")
    print(f"运算: {calc['name']}")
    print(f"解释: {calc['desc']}")
    print(f"{'=' * 70}")

    try:
        result = model.wv.most_similar(
            positive=calc['positive'],
            negative=calc['negative'],
            topn=5
        )

        print("\n结果:")
        for i, (word, score) in enumerate(result, 1):
            bar = "█" * int(score * 30)
            print(f"  {i}. {word:<10} {score:.3f}  {bar}")

        print(f"\n>>> 最接近: {result[0][0]}")

    except Exception as e:
        print(f"\n无法完成运算: {e}")

# 找与结果向量最相似的人物
print(f"\n{'=' * 70}")
print("找运算结果最相似的人物")
print(f"{'=' * 70}")

import numpy as np
from scipy.spatial.distance import cosine

people = ["刘备", "关羽", "张飞", "诸葛亮", "曹操", "孙权", "周瑜",
          "吕布", "赵云", "黄忠", "姜维", "司马懿", "董卓", "袁绍",
          "孙策", "陆逊", "马超", "魏延", "庞统", "法正"]

result_vector = model.wv["刘备"] + model.wv["曹操"] - model.wv["张飞"]

similarities = []
for person in people:
    if person in model.wv:
        sim = 1 - cosine(result_vector, model.wv[person])
        similarities.append((person, sim))

similarities.sort(key=lambda x: x[1], reverse=True)

print("\n'刘备 + 曹操 - 张飞' 的结果向量与人物相似度:")
print("-" * 70)
for i, (person, sim) in enumerate(similarities[:10], 1):
    bar = "█" * int(sim * 30)
    print(f"  {i:2}. {person:<8} {sim:.3f}  {bar}")

print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)
