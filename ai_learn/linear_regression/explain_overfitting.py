"""
演示：为什么要划分训练集和测试集
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("="*70)
print("演示：训练集/测试集划分的重要性")
print("="*70)

# 生成数据
np.random.seed(42)
n_samples = 20
X = np.random.uniform(50, 150, n_samples).reshape(-1, 1)
# 真实关系：y = 3x + 噪声
y = 3 * X.flatten() + np.random.normal(0, 30, n_samples)

print("\n场景1：不划分数据，全量训练")
print("-"*70)

model_all = LinearRegression()
model_all.fit(X, y)
predictions_all = model_all.predict(X)
mse_all = mean_squared_error(y, predictions_all)

print(f"全量训练的MSE: {mse_all:.2f}")
print(f"模型看起来很完美！（但真的吗？）")

# 模拟"新数据"
X_new = np.random.uniform(50, 150, 10).reshape(-1, 1)
y_new = 3 * X_new.flatten() + np.random.normal(0, 30, 10)
predictions_new = model_all.predict(X_new)
mse_new = mean_squared_error(y_new, predictions_new)

print(f"\n在新数据上的MSE: {mse_new:.2f}")
print(f"差距: {abs(mse_new - mse_all):.2f}")
print("\n问题：我们不知道模型在新数据上的真实表现！")

print("\n" + "="*70)
print("场景2：划分数据，有训练集和测试集")
print("-"*70)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model_split = LinearRegression()
model_split.fit(X_train, y_train)

train_mse = mean_squared_error(y_train, model_split.predict(X_train))
test_mse = mean_squared_error(y_test, model_split.predict(X_test))

print(f"训练集MSE: {train_mse:.2f}")
print(f"测试集MSE: {test_mse:.2f}")
print(f"差距: {abs(test_mse - train_mse):.2f}")

print("\n✓ 现在我们可以评估模型了！")
print("✓ 测试集MSE反映了模型在新数据上的真实表现")

print("\n" + "="*70)
print("场景3：演示过拟合")
print("-"*70)

# 使用多项式特征（容易过拟合）
from sklearn.preprocessing import PolynomialFeatures

# 创建一些稍微复杂的数据
np.random.seed(42)
X_complex = np.random.uniform(50, 150, 30).reshape(-1, 1)
y_complex = 3 * X_complex.flatten() + np.random.normal(0, 40, 30)

# 划分数据
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_complex, y_complex, test_size=0.3, random_state=42
)

# 高次多项式（容易过拟合）
poly = PolynomialFeatures(degree=10)  # 10次多项式
X_train_poly = poly.fit_transform(X_train_c)
X_test_poly = poly.transform(X_test_c)

model_overfit = LinearRegression()
model_overfit.fit(X_train_poly, y_train_c)

train_mse_overfit = mean_squared_error(y_train_c, model_overfit.predict(X_train_poly))
test_mse_overfit = mean_squared_error(y_test_c, model_overfit.predict(X_test_poly))

print(f"过拟合模型的训练集MSE: {train_mse_overfit:.2f}")
print(f"过拟合模型的测试集MSE: {test_mse_overfit:.2f}")
print(f"差距: {abs(test_mse_overfit - train_mse_overfit):.2f}")

if test_mse_overfit > train_mse_overfit * 2:
    print("\n⚠️  警告：测试集MSE远大于训练集MSE！")
    print("   这是典型的过拟合现象！")
    print("   模型记住了训练数据，但没有学到真正的规律")

print("\n" + "="*70)
print("可视化对比")
print("-"*70)

# 可视化简单模型 vs 过拟合模型
plt.figure(figsize=(14, 5))

# 左图：简单线性模型
plt.subplot(1, 2, 1)
model_simple = LinearRegression()
model_simple.fit(X_train_c, y_train_c)

X_plot = np.linspace(50, 150, 100).reshape(-1, 1)
plt.scatter(X_train_c, y_train_c, color='blue', alpha=0.5, label='Training Data')
plt.scatter(X_test_c, y_test_c, color='green', alpha=0.5, label='Test Data')
plt.plot(X_plot, model_simple.predict(X_plot), color='red', linewidth=2, label='Simple Model')
plt.xlabel('Area (sqm)')
plt.ylabel('Price (10k yuan)')
plt.title(f'Simple Model (Train MSE={train_mse:.1f}, Test MSE={test_mse:.1f})')
plt.legend()
plt.grid(True, alpha=0.3)

# 右图：过拟合模型
plt.subplot(1, 2, 2)
plt.scatter(X_train_c, y_train_c, color='blue', alpha=0.5, label='Training Data')
plt.scatter(X_test_c, y_test_c, color='green', alpha=0.5, label='Test Data')
plt.plot(X_plot, model_overfit.predict(poly.transform(X_plot)),
         color='red', linewidth=2, label='Overfit Model (degree=10)')
plt.xlabel('Area (sqm)')
plt.ylabel('Price (10k yuan)')
plt.title(f'Overfit Model (Train MSE={train_mse_overfit:.1f}, Test MSE={test_mse_overfit:.1f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/zhengnan/Sniper/Developer/github/LearnAgent/ai_learn/linear_regression/explain_overfitting.png', dpi=100, bbox_inches='tight')
print("\n图表已保存: explain_overfitting.png")
plt.close()

print("\n" + "="*70)
print("总结")
print("="*70)
print("""
1. 全量训练的问题：
   - 无法评估泛化能力
   - 容易过拟合
   - 看起来好，实际可能很差

2. 划分数据的好处：
   - 测试集 = 模拟新数据
   - 能真实评估模型
   - 及时发现过拟合

3. 如何判断过拟合：
   - 测试集MSE >> 训练集MSE
   - 说明模型在死记硬背

4. 常见的划分比例：
   - 训练集 70-80%
   - 测试集 20-30%
   - 数据多时，测试集可以更小（10-20%）
""")
