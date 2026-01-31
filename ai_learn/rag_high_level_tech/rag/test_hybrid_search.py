"""
调试混合检索 - 检查各个组件
"""

import pickle
import numpy as np
import faiss

# 加载数据
print("加载索引...")
with open('threekingdoms_hybrid_chunks.pkl', 'rb') as f:
    data = pickle.load(f)
    chunks = data['chunks']
    embeddings = data['embeddings']

with open('threekingdoms_hybrid_bm25.pkl', 'rb') as f:
    bm25_data = pickle.load(f)

faiss_index = faiss.read_index('threekingdoms_hybrid_faiss.bin')

print(f"✅ 加载完成")
print(f"   文档块数量: {len(chunks)}")
print(f"   向量维度: {embeddings.shape[1]}")
print(f"   BM25词汇量: {len(bm25_data['doc_freqs'])}")

# 检查chunks内容
print(f"\n📝 前3个文档块预览:")
for i in range(min(3, len(chunks))):
    print(f"\n[{i}] {chunks[i]['metadata']}")
    print(f"    内容: {chunks[i]['content'][:100]}...")

# 测试查询
test_queries = [
    "诸葛亮",
    "孔明",
    "扇子",
    "赤壁之战"
]

print(f"\n🔍 测试查询:")
for query in test_queries:
    print(f"\n查询: {query}")

    # 向量检索（使用ollama）
    print(f"  注意：需要Ollama运行才能测试向量检索")

    # BM25检索 - 简单测试
    query_lower = query.lower()
    matches = 0
    for i, chunk in enumerate(chunks[:10]):  # 只检查前10个
        if query_lower in chunk['content'].lower():
            matches += 1
            if matches <= 2:
                print(f"  找到匹配: {chunk['metadata'][:50]}")

    print(f"  前10个文档中匹配数: {matches}")

print("\n✅ 调试完成")
