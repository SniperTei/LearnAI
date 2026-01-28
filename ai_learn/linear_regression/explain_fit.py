"""
解释 model.fit()
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# 数据
X = np.array([50, 60, 70, 80, 90, 100])
y = np.array([150, 180, 210, 240, 270, 300])

# reshape
X_reshaped = X.reshape(-1, 1)

print("="*60)
print("理解 model.fit()")
print("="*60)

print("\nfit 之前（模型是空的）:")
print("-"*60)
model = LinearRegression()
print("model 还没训练，没有 coef_ 和 intercept_ 属性")
print("相当于一个还没学习的小学生")

print("\nfit 之后（模型学到了参数）:")
print("-"*60)
model.fit(X_reshaped, y)
print(f"model.coef_: {model.coef_}")      # [3.] - 学到的权重 w
print(f"model.intercept_: {model.intercept_}")  # 0. - 学到的偏置 b

print("\n" + "="*60)
print("fit() 内部发生了什么？")
print("="*60)
print("""
1. 接收数据 X 和 y
2. 初始化 w 和 b（通常是随机值）
3. 不断尝试不同的 w 和 b
4. 计算每种情况的误差（MSE）
5. 找到让误差最小的 w 和 b
6. 保存这个最优的 w 和 b 到 model.coef_ 和 model.intercept_
""")

print("\n" + "="*60)
print("完整流程演示:")
print("="*60)

print("\n步骤1: 创建模型")
model_new = LinearRegression()
print("  ✓ model = LinearRegression()")

print("\n步骤2: 准备数据")
print(f"  X = {X}")
print(f"  y = {y}")
print(f"  X_reshaped = {X_reshaped.flatten()}")

print("\n步骤3: 训练模型")
model_new.fit(X_reshaped, y)
print("  ✓ model.fit(X_reshaped, y)")
print(f"  ✓ 模型学到了: w={model_new.coef_[0]}, b={model_new.intercept_}")

print("\n步骤4: 做预测")
new_x = [[95]]
prediction = model_new.predict(new_x)
print(f"  预测 95平米的房子: {prediction[0]}万元")

print("\n" + "="*60)
print("类比：小学生学习")
print("="*60)
print("""
model = LinearRegression()  →  小学生入学（大脑是空的）
model.fit(X, y)             →  开始学习（老师教知识）
  - 看 X（题目）
  - 看 y（答案）
  - 找规律（学到了 y = 3x）
model.predict([[95]])        →  考试（做新题目）
  - 题目：95平米的房子多少钱？
  - 根据学的规律：95 × 3 = 285万
""")

print("\n" + "="*60)
print("为什么要 reshape？再强调一次")
print("="*60)
print("""
❌ 错误写法:
  model.fit(X, y)
  报错: Expected 2D array, got 1D array instead

✓ 正确写法:
  model.fit(X.reshape(-1, 1), y)

原因: sklearn 要求 X 必须是二维的
""")
