"""
逻辑回归实战：学生成绩分类
边学算法边学数学 - 从二分类到多分类
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("逻辑回归入门：从二分类到多分类")
print("="*70)

# ===== 第一部分：二分类逻辑回归 =====
print("\n" + "="*70)
print("第一部分：二分类 - 学生是否及格")
print("="*70)

# 生成数据：学习时长 vs 是否及格
# 特征：学习时长
# 标签：0=不及格, 1=及格
np.random.seed(42)

# 不及格的学生：学习时间较短
fail_study = np.random.normal(30, 10, 50)

# 及格的学生：学习时间较长
pass_study = np.random.normal(70, 10, 50)

# 合并数据
X_binary = np.concatenate([fail_study, pass_study])
y_binary = np.concatenate([np.zeros(50), np.ones(50)])

print("\n数据统计：")
print(f"不及格学生人数: {sum(y_binary == 0)}")
print(f"及格学生人数: {sum(y_binary == 1)}")
print(f"不及格学生平均学习时长: {np.mean(fail_study):.2f}小时")
print(f"及格学生平均学习时长: {np.mean(pass_study):.2f}小时")

# 划分训练集和测试集
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary.reshape(-1, 1), y_binary, test_size=0.3, random_state=42
)

# 标准化数据
scaler_bin = StandardScaler()
X_train_bin_scaled = scaler_bin.fit_transform(X_train_bin)
X_test_bin_scaled = scaler_bin.transform(X_test_bin)

# 训练二分类模型
print("\n训练二分类逻辑回归模型...")
model_binary = LogisticRegression(random_state=42)
model_binary.fit(X_train_bin_scaled, y_train_bin)

# 预测
y_pred_bin = model_binary.predict(X_test_bin_scaled)
y_prob_bin = model_binary.predict_proba(X_test_bin_scaled)

print(f"\n模型准确率: {accuracy_score(y_test_bin, y_pred_bin):.2%}")
print(f"模型参数: 权重 w = {model_binary.coef_[0][0]:.4f}, 截距 b = {model_binary.intercept_[0]:.4f}")

# 预测新学生
new_students = np.array([[25], [50], [80]])
new_students_scaled = scaler_bin.transform(new_students)
predictions = model_binary.predict(new_students_scaled)
probabilities = model_binary.predict_proba(new_students_scaled)

print("\n预测新学生：")
for i, (hours, pred, prob) in enumerate(zip(new_students, predictions, probabilities)):
    result = "及格" if pred == 1 else "不及格"
    print(f"  学习{hours[0]}小时: 预测{result} (及格概率: {prob[1]:.2%})")

# 可视化二分类结果
plt.figure(figsize=(12, 5))

# 子图1：数据分布
plt.subplot(1, 2, 1)
plt.scatter(fail_study, np.zeros(50), c='red', alpha=0.6, label='不及格', s=50)
plt.scatter(pass_study, np.ones(50), c='green', alpha=0.6, label='及格', s=50)
plt.xlabel('学习时长（小时）', fontsize=12)
plt.ylabel('是否及格', fontsize=12)
plt.title('二分类数据分布', fontsize=14)
plt.yticks([0, 1], ['不及格', '及格'])
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：Sigmoid函数和预测概率
plt.subplot(1, 2, 2)
# 绘制Sigmoid曲线
x_range = np.linspace(-3, 3, 100).reshape(-1, 1)
# 使用原始尺度
x_original = scaler_bin.inverse_transform(x_range)
prob_curve = model_binary.predict_proba(x_range)[:, 1]

plt.scatter(fail_study, np.zeros(50), c='red', alpha=0.3, s=50)
plt.scatter(pass_study, np.ones(50), c='green', alpha=0.3, s=50)
plt.plot(x_original, prob_curve, 'b-', linewidth=2, label='Sigmoid函数')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='决策边界 (0.5)')
plt.xlabel('学习时长（小时）', fontsize=12)
plt.ylabel('及格概率', fontsize=12)
plt.title('逻辑回归：Sigmoid决策边界', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/zhengnan/Sniper/Developer/github/LearnAgent/ai_learn/logistic_regression/binary_classification.png', dpi=100, bbox_inches='tight')
print("\n图表已保存: binary_classification.png")
plt.close()

print("\n核心概念：Sigmoid函数")
print("σ(z) = 1 / (1 + e^(-z))")
print("作用：将任意实数映射到 [0, 1] 区间，表示概率")
print("决策规则：概率 >= 0.5 → 预测为类别1")

# ===== 第二部分：多分类逻辑回归 =====
print("\n" + "="*70)
print("第二部分：多分类 - 学生成绩评级（优秀/良好/及格/不及格）")
print("="*70)

# 生成数据
# 特征：[学习时长, 作业完成率]
# 标签：0=不及格, 1=及格, 2=良好, 3=优秀

np.random.seed(42)

# 不及格：学习时间短，作业完成率低
poor_study = np.random.normal(20, 5, 40)
poor_homework = np.random.normal(0.3, 0.1, 40)

# 及格：学习时间中等偏短，作业完成率中等
pass_study = np.random.normal(40, 8, 40)
pass_homework = np.random.normal(0.6, 0.1, 40)

# 良好：学习时间中等，作业完成率较高
good_study = np.random.normal(60, 8, 40)
good_homework = np.random.normal(0.8, 0.08, 40)

# 优秀：学习时间长，作业完成率高
excellent_study = np.random.normal(80, 8, 40)
excellent_homework = np.random.normal(0.95, 0.03, 40)

# 合并数据
X_multi = np.column_stack([
    np.concatenate([poor_study, pass_study, good_study, excellent_study]),
    np.concatenate([poor_homework, pass_homework, good_homework, excellent_homework])
])
y_multi = np.concatenate([
    np.zeros(40),  # 不及格
    np.ones(40),   # 及格
    np.full(40, 2), # 良好
    np.full(40, 3)  # 优秀
])

class_names = ['不及格', '及格', '良好', '优秀']

print("\n数据统计：")
for i, name in enumerate(class_names):
    count = sum(y_multi == i)
    print(f"  {name}: {count}人")

# 划分训练集和测试集
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42, stratify=y_multi
)

# 标准化数据
scaler_multi = StandardScaler()
X_train_multi_scaled = scaler_multi.fit_transform(X_train_multi)
X_test_multi_scaled = scaler_multi.transform(X_test_multi)

# 训练多分类模型
print("\n训练多分类逻辑回归模型...")
# 注：sklearn会自动检测多分类问题并使用合适的策略
model_multi = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
model_multi.fit(X_train_multi_scaled, y_train_multi)

# 预测
y_pred_multi = model_multi.predict(X_test_multi_scaled)
y_prob_multi = model_multi.predict_proba(X_test_multi_scaled)

print(f"\n模型准确率: {accuracy_score(y_test_multi, y_pred_multi):.2%}")

print("\n详细分类报告:")
print(classification_report(y_test_multi, y_pred_multi, target_names=class_names))

# 预测新学生
print("\n预测新学生成绩等级：")
new_students_multi = np.array([
    [15, 0.2],   # 学习15小时，作业20%完成
    [35, 0.5],   # 学习35小时，作业50%完成
    [55, 0.75],  # 学习55小时，作业75%完成
    [85, 0.98],  # 学习85小时，作业98%完成
    [50, 0.85],  # 边界案例
])

new_students_multi_scaled = scaler_multi.transform(new_students_multi)
predictions_multi = model_multi.predict(new_students_multi_scaled)
probabilities_multi = model_multi.predict_proba(new_students_multi_scaled)

for i, (features, pred, probs) in enumerate(zip(new_students_multi, predictions_multi, probabilities_multi)):
    print(f"\n学生{i+1}: 学习{features[0]}小时, 作业完成率{features[1]*100:.0f}%")
    print(f"  预测等级: {class_names[int(pred)]}")
    print(f"  各等级概率:")
    for j, (cls_name, prob) in enumerate(zip(class_names, probs)):
        bar = "█" * int(prob * 30)
        print(f"    {cls_name}: {prob:.2%} {bar}")

# 可视化多分类决策边界
print("\n生成多分类决策边界可视化...")

# 创建网格
x_min, x_max = X_multi[:, 0].min() - 5, X_multi[:, 0].max() + 5
y_min, y_max = X_multi[:, 1].min() - 0.1, X_multi[:, 1].max() + 0.1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.5),
    np.arange(y_min, y_max, 0.01)
)

# 预测网格中每个点的类别
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_scaled = scaler_multi.transform(grid_points)
Z = model_multi.predict(grid_points_scaled)
Z = Z.reshape(xx.shape)

# 绘制
plt.figure(figsize=(12, 8))

# 定义颜色
colors = ['red', 'yellow', 'lightblue', 'green']

# 绘制决策边界
from matplotlib.colors import ListedColormap
cmap_background = ListedColormap(['#fff5f5', '#fffff0', '#f0f8ff', '#f0fff0'])
plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.8)

# 绘制数据点
for i, (name, color) in enumerate(zip(class_names, colors)):
    mask = y_multi == i
    plt.scatter(X_multi[mask, 0], X_multi[mask, 1], c=color, label=name,
               s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

# 标记新学生
markers = ['o', 's', '^', 'D', '*']
for i, (features, pred) in enumerate(zip(new_students_multi, predictions_multi)):
    plt.scatter(features[0], features[1], c='black', marker=markers[4],
               s=200, edgecolors='red', linewidths=2, zorder=5)

plt.xlabel('学习时长（小时）', fontsize=12)
plt.ylabel('作业完成率', fontsize=12)
plt.title('逻辑回归多分类：学生成绩评级决策边界', fontsize=14)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/zhengnan/Sniper/Developer/github/LearnAgent/ai_learn/logistic_regression/multiclass_decision_boundary.png', dpi=100, bbox_inches='tight')
print("决策边界图已保存: multiclass_decision_boundary.png")
plt.close()

# ===== 第三部分：核心概念总结 =====
print("\n" + "="*70)
print("核心概念总结")
print("="*70)

print("\n1. 逻辑回归 vs 线性回归")
print("   - 线性回归：预测连续值（如房价、温度）")
print("   - 逻辑回归：预测类别（如及格/不及格，优秀/良好/一般）")

print("\n2. Sigmoid函数（二分类）")
print("   σ(z) = 1 / (1 + e^(-z))")
print("   - 将线性输出映射到 [0, 1] 区间")
print("   - 输出解释为属于正类的概率")
print("   - 决策边界通常是 0.5")

print("\n3. Softmax函数（多分类）")
print("   σ(z)_i = e^z_i / Σ(e^z_j)")
print("   - 将输出转换为概率分布")
print("   - 所有类别概率之和为 1")
print("   - 选择概率最大的类别作为预测")

print("\n4. 损失函数")
print("   - 二分类：二元交叉熵损失")
print("   - 多分类：分类交叉熵损失")
print("   - 目标：最小化预测概率与真实标签的差异")

print("\n5. 多分类策略")
print("   - One-vs-Rest (OvR): 训练多个二分类器")
print("   - Softmax (Multinomial): 直接多分类，推荐使用")

print("\n6. 数据标准化的重要性")
print("   - 逻辑回归对特征尺度敏感")
print("   - 标准化后收敛更快，性能更好")
print("   - 使用 StandardScaler 或 MinMaxScaler")

print("\n" + "="*70)
print("实战技巧")
print("="*70)

print("\n1. 查看概率而不只是类别标签")
print("   proba = model.predict_proba(X)")
print("   这能告诉你模型的置信度")

print("\n2. 调整决策阈值")
print("   默认阈值是 0.5，但可以根据需求调整")
print("   例如：医疗诊断可能需要更低的阈值（宁可误报，不可漏报）")

print("\n3. 处理类别不平衡")
print("   - 使用 class_weight='balanced'")
print("   - 或使用过采样/欠采样技术")

print("\n4. 正则化防止过拟合")
print("   - L1正则化（Lasso）：产生稀疏解，特征选择")
print("   - L2正则化（Ridge）：防止过拟合，默认使用")
print("   - 调整参数 C（C越小，正则化越强）")

print("\n" + "="*70)
print("下一步学习建议")
print("="*70)

print("\n1. 尝试不同的数据集")
print("   - 鸢尾花数据集（经典多分类）")
print("   - 泰坦尼克号数据集（二分类）")

print("\n2. 深入理解数学原理")
print("   - 交叉熵损失的推导")
print("   - 梯度下降的更新公式")
print("   - 正则化的作用机制")

print("\n3. 学习更多分类算法")
print("   - 决策树")
print("   - 支持向量机（SVM）")
print("   - 随机森林")
print("   - 神经网络")

print("\n" + "="*70)
print("演示完成！")
print("="*70)
