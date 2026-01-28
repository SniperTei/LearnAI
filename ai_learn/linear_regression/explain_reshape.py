"""
解释 reshape(-1, 1)
"""

import numpy as np

# 原始数据
X = np.array([50, 60, 70, 80, 90, 100])

print("="*60)
print("理解 reshape(-1, 1)")
print("="*60)

print("\n原始数据 X:")
print(X)
print(f"形状: {X.shape}")  # (6,) - 一维数组

print("\n" + "-"*60)
print("reshape 后:")
print("-"*60)

X_reshaped = X.reshape(-1, 1)

print("\nX_reshaped = X.reshape(-1, 1):")
print(X_reshaped)
print(f"形状: {X_reshaped.shape}")  # (6, 2) - 二维数组

print("\n" + "="*60)
print("对比：")
print("="*60)
print(f"原始 X 是 1维数组: {X.shape}")
print(f"  像这样: [50, 60, 70, 80, 90, 100]")
print()
print(f"reshape后是 2维数组: {X_reshaped.shape}")
print(f"  像这样:")
for i in range(len(X_reshaped)):
    print(f"    [{X_reshaped[i][0]}]")

print("\n" + "="*60)
print("为什么要 reshape？")
print("="*60)
print("sklearn 的模型要求 X 必须是二维的！")
print()
print("原因：")
print("1. 支持多特征 - 每行是一个样本，每列是一个特征")
print("2. 统一格式 - 无论几个特征，格式都一样")
print()
print("例子 - 多特征情况:")
print("如果数据是: [面积, 房间数]")
print("reshape后变成:")
print("[[50, 1],")   # 样本1: 50平米, 1房间
print(" [60, 1],")   # 样本2: 60平米, 1房间
print(" [70, 2],")   # 样本3: 70平米, 2房间
print(" ...]")
print()
print("第一列 = 面积特征")
print("第二列 = 房间数特征")

print("\n" + "="*60)
print("reshape(-1, 1) 中的 -1 是什么意思？")
print("="*60)
print("reshape(-1, 1) 表示:")
print("  -1: 自动计算行数（根据数据长度）")
print("   1: 变成1列")
print()
print("等价写法:")
print(f"  reshape(-1, 1)  -> {X.reshape(-1, 1).shape}")
print(f"  reshape(6, 1)   -> {X.reshape(6, 1).shape}")
print(f"  reshape(len(X), 1) -> {X.reshape(len(X), 1).shape}")
print()
print("所以 -1 很方便：让numpy自动计算！")

print("\n" + "="*60)
print("记忆方法：")
print("="*60)
print("reshape(-1, 1) = 把一维数组变成一列")
print()
print("  原来像这样: [50, 60, 70]")
print("  变成这样:")
print("  [[50],")
print("   [60],")
print("   [70]]")
