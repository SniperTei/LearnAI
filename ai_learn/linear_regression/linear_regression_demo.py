"""
线性回归实战：预测房价
边学算法边学数学
"""

import numpy as np
import matplotlib.pyplot as plt

# ===== Data Preparation =====
# 假设我们收集了一些房子的数据
# 面积（平方米）和房价（万元）

# 输入特征 X：房子的面积
X = np.array([50, 60, 70, 80, 90, 100, 110, 120])

# 目标值 y：房价（万元）
y = np.array([150, 180, 210, 240, 270, 300, 330, 360])

print("===== 数据查看 =====")
print(f"房子面积: {X}")
print(f"房价: {y}")
print(f"数据量: {len(X)}套房子")
print()

# 数学概念：均值（平均值）
# 均值 = 所有数的和 / 数的个数
area_mean = np.mean(X)
price_mean = np.mean(y)

print("===== 数学概念1：均值 =====")
print(f"平均面积: {area_mean:.2f} 平方米")
print(f"平均房价: {price_mean:.2f} 万元")
print()

# 数学概念：求和符号 Σ
# Σᵢ₌₁ⁿ xᵢ 表示从第1个数加到第n个数
area_sum = np.sum(X)
price_sum = np.sum(y)

print("===== 数学概念2：求和符号 Σ =====")
print(f"面积总和: {area_sum}")
print(f"房价总和: {price_sum}")
print()

# 可视化数据
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', s=100, label='True Data')
plt.xlabel('Area (sqm)', fontsize=12)
plt.ylabel('Price (10k yuan)', fontsize=12)
plt.title('House Area vs Price', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('/Users/zhengnan/Sniper/Developer/github/LearnAgent/ai_learn/linear_regression/house_prices_scatter.png', dpi=100, bbox_inches='tight')
print("图表已保存到: house_prices_scatter.png")
plt.close()

print("\n观察图表：可以看到面积和房价呈线性关系（近似一条直线）")

# ===== 步骤2：手动尝试不同的w和b =====
print("\n" + "="*60)
print("===== 步骤2：手动尝试不同的直线 =====")
print("="*60)

def predict(x, w, b):
    """
    预测函数：y = wx + b
    """
    return w * x + b

def calculate_loss(y_true, y_pred):
    """
    计算损失（误差）：均方误差 MSE
    MSE = (1/n) * Σ(y_true - y_pred)²
    """
    n = len(y_true)
    # 数学概念：平方运算
    # (y_true - y_pred)² 表示误差的平方
    # 为什么要平方？1) 消除负号 2) 放大误差
    squared_errors = (y_true - y_pred) ** 2
    mse = np.sum(squared_errors) / n
    return mse

# 尝试1：猜测 w=3, b=0
print("\n尝试1：猜测 w=3, b=0")
w1, b1 = 3, 0
y_pred1 = predict(X, w1, b1)
loss1 = calculate_loss(y, y_pred1)
print(f"预测值: {y_pred1}")
print(f"真实值: {y}")
print(f"误差: {y - y_pred1}")
print(f"均方误差 MSE: {loss1:.2f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', s=100, label='True Data', alpha=0.7)
plt.plot(X, y_pred1, color='red', linewidth=2, label=f'Prediction Line (w={w1}, b={b1})')
plt.xlabel('Area (sqm)', fontsize=12)
plt.ylabel('Price (10k yuan)', fontsize=12)
plt.title(f'Manual Guess: w={w1}, b={b1}, MSE={loss1:.2f}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('/Users/zhengnan/Sniper/Developer/github/LearnAgent/ai_learn/linear_regression/house_prices_try1.png', dpi=100, bbox_inches='tight')
print("图表已保存到: house_prices_try1.png")
plt.close()

# 尝试2：调整 w=2.5, b=50
print("\n尝试2：调整 w=2.5, b=50")
w2, b2 = 2.5, 50
y_pred2 = predict(X, w2, b2)
loss2 = calculate_loss(y, y_pred2)
print(f"预测值: {y_pred2}")
print(f"真实值: {y}")
print(f"均方误差 MSE: {loss2:.2f}")
print(f"误差是否变小？{'是' if loss2 < loss1 else '否'}")

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', s=100, label='True Data', alpha=0.7)
plt.plot(X, y_pred2, color='green', linewidth=2, label=f'Prediction Line (w={w2}, b={b2})')
plt.xlabel('Area (sqm)', fontsize=12)
plt.ylabel('Price (10k yuan)', fontsize=12)
plt.title(f'Adjusted: w={w2}, b={b2}, MSE={loss2:.2f}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('/Users/zhengnan/Sniper/Developer/github/LearnAgent/ai_learn/linear_regression/house_prices_try2.png', dpi=100, bbox_inches='tight')
print("图表已保存到: house_prices_try2.png")
plt.close()

print("\n" + "="*60)
print("数学概念3：均方误差 MSE")
print("="*60)
print("MSE = (1/n) × Σ(真实值 - 预测值)²")
print("\n为什么要用均方误差？")
print("1. 平方可以消除负号（误差可能是负的）")
print("2. 平方可以放大大误差（惩罚大的预测错误）")
print("3. 除以n是为了平均（不受数据量影响）")
print("\n目标：找到让MSE最小的 w 和 b！")

# ===== 步骤3：使用sklearn自动训练 =====
print("\n" + "="*60)
print("===== 步骤3：让机器自动找参数 =====")
print("="*60)

from sklearn.linear_model import LinearRegression

# 准备数据：sklearn需要2维数组
X_reshaped = X.reshape(-1, 1)  # 变成 [[50], [60], ...]

# 创建模型
model = LinearRegression()

# 训练模型（自动找最优的w和b）
model.fit(X_reshaped, y)

# 获取训练好的参数
w_best = model.coef_[0]
b_best = model.intercept_

print(f"\n机器找到的最优参数：")
print(f"w（权重/斜率）: {w_best:.2f}")
print(f"b（偏置/截距）: {b_best:.2f}")
print(f"\n预测公式：房价 = {w_best:.2f} × 面积 + {b_best:.2f}")

# 用训练好的模型预测
y_pred_best = model.predict(X_reshaped)
loss_best = calculate_loss(y, y_pred_best)
print(f"\n均方误差 MSE: {loss_best:.4f}")

# 预测新数据
new_house_area = 95  # 95平米的房子
predicted_price = model.predict([[new_house_area]])[0]
print(f"\n预测：{new_house_area}平米的房子价格 = {predicted_price:.2f}万元")

# 可视化最终结果
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', s=100, label='True Data', alpha=0.7)
plt.plot(X, y_pred_best, color='red', linewidth=2, label=f'Best Fit Line: w={w_best:.2f}, b={b_best:.2f}')
plt.scatter([new_house_area], [predicted_price], color='green', s=150, marker='*',
           label=f'Prediction: {predicted_price:.2f}k', zorder=5)
plt.xlabel('Area (sqm)', fontsize=12)
plt.ylabel('Price (10k yuan)', fontsize=12)
plt.title(f'Linear Regression Result (MSE={loss_best:.4f})', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('/Users/zhengnan/Sniper/Developer/github/LearnAgent/ai_learn/linear_regression/house_prices_final.png', dpi=100, bbox_inches='tight')
print("\n最终图表已保存到: house_prices_final.png")
plt.close()

# 数学概念：梯度下降的直观理解
print("\n" + "="*60)
print("数学概念4：梯度下降")
print("="*60)
print("""
想象你在山上（MSE是高度），想下山（最小化MSE）

1. 梯度 = 坡度最大的方向
2. 梯度下降 = 沿着坡度最陡的方向走
3. 学习率 = 每次走多远（步长）

过程：
  随机初始化 w, b
  循环：
    计算梯度（坡度）
    沿梯度的反方向走一步
    更新 w, b
  直到到达最低点（MSE最小）

类比：
- 山的高度 = 损失函数值
- 你的位置 = 当前的 w, b
- 下山的路 = 梯度方向
- 步子大小 = 学习率
""")

print("机器学习就是让计算机自动完成这个过程！")

