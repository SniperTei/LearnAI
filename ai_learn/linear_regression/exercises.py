"""
线性回归练习题
Linear Regression Exercises

巩固你对线性回归的理解
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*70)
print("线性回归练习题 - Linear Regression Exercises")
print("="*70)

# ============================================================================
# 练习1：添加噪声的数据（Exercises 1: Noisy Data）
# ============================================================================
print("\n" + "="*70)
print("练习1: 添加噪声的数据")
print("="*70)
print("\n目标：理解现实中的数据往往不完美，看看模型如何处理噪声数据\n")

# 生成带噪声的数据
np.random.seed(42)  # 设置随机种子，确保结果可复现
X = np.array([50, 60, 70, 80, 90, 100, 110, 120])
y_true = 3 * X + 0  # 真实关系
noise = np.random.normal(0, 15, len(X))  # 添加高斯噪声（均值0，标准差15）
y_noisy = y_true + noise

print("真实关系: y = 3x + 0")
print(f"真实值: {y_true}")
print(f"噪声: {noise.round(1)}")
print(f"观测值（真实值+噪声）: {y_noisy.round(1)}")

# TODO: 训练模型
# 你的任务：使用 X 和 y_noisy 训练一个线性回归模型
model = LinearRegression()
X_reshaped = X.reshape(-1, 1)
model.fit(X_reshaped, y_noisy)

print(f"\n模型学到的参数:")
print(f"w（斜率）: {model.coef_[0]:.2f} (真实值是 3.00)")
print(f"b（截距）: {model.intercept_:.2f} (真实值是 0.00)")
print(f"MSE: {mean_squared_error(y_noisy, model.predict(X_reshaped)):.2f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X, y_noisy, color='blue', s=100, label='Noisy Data', alpha=0.7)
plt.plot(X, y_true, color='green', linewidth=2, linestyle='--', label='True Line (y=3x)')
plt.plot(X, model.predict(X_reshaped), color='red', linewidth=2, label=f'Modeled Line (w={model.coef_[0]:.2f})')
plt.xlabel('Area (sqm)', fontsize=12)
plt.ylabel('Price (10k yuan)', fontsize=12)
plt.title('Exercise 1: Linear Regression with Noisy Data', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/Users/zhengnan/Sniper/Developer/github/LearnAgent/ai_learn/linear_regression/exercise1_noisy_data.png', dpi=100, bbox_inches='tight')
print("\n图表已保存: exercise1_noisy_data.png")
plt.close()

print("\n思考题:")
print("1. 模型找到的w和b与真实值（w=3, b=0）接近吗？")
print("2. 如果噪声更大（比如标准差=30），模型表现会怎样？")
print("3. 如果增加数据量（比如20个样本），模型会变好还是变差？")


# ============================================================================
# 练习2：多特征线性回归（Exercise 2: Multiple Features）
# ============================================================================
print("\n" + "="*70)
print("练习2: 多特征线性回归")
print("="*70)
print("\n目标：处理多个特征（面积 + 房间数）\n")

# 多特征数据
X_multi = np.array([
    [50, 1],   # [面积, 房间数]
    [60, 1],
    [70, 2],
    [80, 2],
    [90, 3],
    [100, 3],
    [110, 4],
    [120, 4]
])
y_multi = np.array([150, 180, 220, 250, 300, 330, 390, 420])

print("数据样例:")
print("前3个样本:")
for i in range(3):
    print(f"  样本{i+1}: 面积={X_multi[i][0]}平米, 房间数={X_multi[i][1]}, 房价={y_multi[i]}万")

# TODO: 训练模型
model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

print(f"\n模型学到的参数:")
print(f"权重: {model_multi.coef_}  [面积权重, 房间数权重]")
print(f"偏置: {model_multi.intercept_:.2f}")

# TODO: 预测新房子
# 预测一个95平米、2个房间的房子
new_house = [[95, 2]]
predicted_price = model_multi.predict(new_house)[0]
print(f"\n预测: 95平米、2个房间的房子价格 = {predicted_price:.2f}万元")

# 对比：只用面积预测
model_single = LinearRegression()
model_single.fit(X_multi[:, [0]], y_multi)
predicted_single = model_single.predict([[95]])[0]
print(f"对比: 只用面积预测95平米的房子价格 = {predicted_single:.2f}万元")
print(f"差异: {predicted_price - predicted_single:.2f}万元（考虑了房间数）")

print("\n思考题:")
print("1. 哪个特征的权重更大？说明什么？")
print("2. 为什么多特征预测比单特征更准确？")


# ============================================================================
# 练习3：训练集/测试集划分（Exercise 3: Train/Test Split）
# ============================================================================
print("\n" + "="*70)
print("练习3: 训练集/测试集划分")
print("="*70)
print("\n目标：理解为什么要划分数据，避免过拟合\n")

# 生成更多数据
np.random.seed(42)
X_larger = np.random.randint(50, 130, 50).reshape(-1, 1)
y_larger = 3 * X_larger.flatten() + np.random.normal(0, 20, 50)

print(f"总数据量: {len(X_larger)}个样本")

# TODO: 划分训练集和测试集
# test_size=0.2 表示20%的数据作为测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_larger, y_larger, test_size=0.2, random_state=42
)

print(f"训练集: {len(X_train)}个样本")
print(f"测试集: {len(X_test)}个样本")

# 在训练集上训练
model_split = LinearRegression()
model_split.fit(X_train, y_train)

# 在训练集和测试集上分别评估
train_mse = mean_squared_error(y_train, model_split.predict(X_train))
test_mse = mean_squared_error(y_test, model_split.predict(X_test))
train_r2 = r2_score(y_train, model_split.predict(X_train))
test_r2 = r2_score(y_test, model_split.predict(X_test))

print(f"\n模型性能:")
print(f"训练集 MSE: {train_mse:.2f}, R²: {train_r2:.4f}")
print(f"测试集 MSE: {test_mse:.2f}, R²: {test_r2:.4f}")

# 可视化
plt.figure(figsize=(12, 5))

# 训练集
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training Data')
plt.plot(X_train, model_split.predict(X_train), color='red', linewidth=2)
plt.xlabel('Area (sqm)')
plt.ylabel('Price (10k yuan)')
plt.title(f'Training Set (MSE={train_mse:.2f})')
plt.legend()
plt.grid(True, alpha=0.3)

# 测试集
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Test Data')
plt.plot(X_test, model_split.predict(X_test), color='red', linewidth=2)
plt.xlabel('Area (sqm)')
plt.ylabel('Price (10k yuan)')
plt.title(f'Test Set (MSE={test_mse:.2f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/zhengnan/Sniper/Developer/github/LearnAgent/ai_learn/linear_regression/exercise3_train_test_split.png', dpi=100, bbox_inches='tight')
print("\n图表已保存: exercise3_train_test_split.png")
plt.close()

print("\n思考题:")
print("1. 训练集和测试集的MSE哪个大？为什么？")
print("2. 如果测试集MSE远大于训练集MSE，说明什么？（过拟合）")
print("3. 为什么要用测试集来评估模型？")


# ============================================================================
# 练习4：模型评估指标（Exercise 4: Evaluation Metrics）
# ============================================================================
print("\n" + "="*70)
print("练习4: 模型评估指标")
print("="*70)
print("\n目标：理解不同评估指标的含义和用途\n")

# 使用练习3的模型
y_pred = model_split.predict(X_test)

# 计算各种评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("评估指标:")
print(f"MSE  (均方误差): {mse:.2f}")
print(f"RMSE (均方根误差): {rmse:.2f}")
print(f"MAE  (平均绝对误差): {mae:.2f}")
print(f"R²   (决定系数): {r2:.4f}")

print("\n各指标的含义:")
print("• MSE  (Mean Squared Error):")
print("    - 误差平方的平均值")
print("    - 优点：放大误差，易于求导")
print("    - 缺点：单位是平方，不易直观理解")
print()
print("• RMSE (Root Mean Squared Error):")
print("    - MSE的平方根")
print("    - 优点：单位和原数据一致，直观")
print("    - 常用！")
print()
print("• MAE  (Mean Absolute Error):")
print("    - 误差绝对值的平均")
print("    - 优点：直观，不受异常值影响")
print()
print("• R²   (R-squared):")
print("    - 0到1之间，越接近1越好")
print("    - 表示模型解释了多少方差")
print("    - 1.0 = 完美拟合，0 = 和均值一样，负数 = 很差")

# 可视化预测 vs 真实值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, s=100)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('True Values (10k yuan)', fontsize=12)
plt.ylabel('Predicted Values (10k yuan)', fontsize=12)
plt.title(f'Prediction vs True Values (R²={r2:.4f})', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/Users/zhengnan/Sniper/Developer/github/LearnAgent/ai_learn/linear_regression/exercise4_evaluation.png', dpi=100, bbox_inches='tight')
print("\n图表已保存: exercise4_evaluation.png")
plt.close()

print("\n思考题:")
print("1. 如果RMSE=20，是什么意思？")
print("2. R²=0.95表示模型很好了吗？")
print("3. 什么时候用MSE，什么时候用MAE？")


# ============================================================================
# 练习5：挑战题 - 实现梯度下降（Exercise 5: Implement Gradient Descent）
# ============================================================================
print("\n" + "="*70)
print("练习5: 挑战题 - 手动实现梯度下降")
print("="*70)
print("\n目标：深入理解梯度下降的工作原理\n")

# 简单数据
X_gd = np.array([1, 2, 3, 4, 5], dtype=float)
y_gd = np.array([2, 4, 6, 8, 10], dtype=float)

# TODO: 实现梯度下降算法
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    梯度下降算法

    参数:
    X, y: 训练数据
    learning_rate: 学习率（步长）
    iterations: 迭代次数

    返回:
    w, b: 学到的参数
    losses: 每次迭代的损失值
    """
    n = len(X)
    w = 0.0  # 初始化权重
    b = 0.0  # 初始化偏置
    losses = []

    for i in range(iterations):
        # 1. 计算预测值
        y_pred = w * X + b

        # 2. 计算损失（MSE）
        loss = np.mean((y - y_pred) ** 2)
        losses.append(loss)

        # 3. 计算梯度（导数）
        # ∂loss/∂w = -2/n * Σ * (y - y_pred)
        # ∂loss/∂b = -2/n * Σ * (y - y_pred)
        dw = -(2/n) * np.sum(X * (y - y_pred))
        db = -(2/n) * np.sum(y - y_pred)

        # 4. 更新参数（沿梯度反方向走）
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 每100次打印一次
        if i % 100 == 0:
            print(f"Iteration {i}: w={w:.4f}, b={b:.4f}, loss={loss:.4f}")

    return w, b, losses

# 运行梯度下降
w_gd, b_gd, losses = gradient_descent(X_gd, y_gd, learning_rate=0.01, iterations=1000)

print(f"\n最终结果:")
print(f"w = {w_gd:.4f} (真实值: 2.0)")
print(f"b = {b_gd:.4f} (真实值: 0.0)")

# 可视化损失下降过程
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss (MSE)')
plt.title('Gradient Descent: Loss over Iterations')
plt.grid(True, alpha=0.3)

# 拟合结果
plt.subplot(1, 2, 2)
plt.scatter(X_gd, y_gd, color='blue', s=100, label='True Data')
plt.plot(X_gd, w_gd * X_gd + b_gd, color='red', linewidth=2, label=f'Gradient Descent (w={w_gd:.2f})')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gradient Descent Result')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/zhengnan/Sniper/Developer/github/LearnAgent/ai_learn/linear_regression/exercise5_gradient_descent.png', dpi=100, bbox_inches='tight')
print("\n图表已保存: exercise5_gradient_descent.png")
plt.close()

print("\n挑战题:")
print("1. 尝试不同的learning_rate（0.01, 0.1, 0.5, 1.0），观察收敛速度")
print("2. 如果learning_rate太大，会发生什么？（发散）")
print("3. 如果learning_rate太小，会发生什么？（收敛慢）")


# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*70)
print("练习完成！")
print("="*70)
print("\n你完成了5个练习:")
print("✓ 练习1: 处理带噪声的数据")
print("✓ 练习2: 多特征线性回归")
print("✓ 练习3: 训练集/测试集划分")
print("✓ 练习4: 模型评估指标")
print("✓ 练习5: 手动实现梯度下降")
print("\n生成的图表:")
print("  - exercise1_noisy_data.png")
print("  - exercise3_train_test_split.png")
print("  - exercise4_evaluation.png")
print("  - exercise5_gradient_descent.png")
print("\n继续加油！你已经掌握了线性回归的核心概念。")
