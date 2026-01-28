"""
泰坦尼克号生存预测 - 逻辑回归实战
基于真实数据集的二分类问题
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# 设置图表风格
sns.set_style("whitegrid")

print("="*70)
print("泰坦尼克号生存预测 - 逻辑回归实战")
print("="*70)

# ===== 第一部分：数据加载 =====
print("\n正在加载泰坦尼克号数据集...")

# 本地数据文件路径
local_data_file = '/Users/zhengnan/Sniper/Developer/github/LearnAgent/ai_learn/logistic_regression/titanic_survival_data.csv'

# 检查本地是否存在数据文件
import os
if os.path.exists(local_data_file):
    print(f"✓ 从本地加载数据: {local_data_file}")
    df = pd.read_csv(local_data_file)
    print(f"✓ 数据加载成功！数据形状: {df.shape}")
else:
    print("本地数据文件不存在，从网络下载...")
    # 使用Stanford的泰坦尼克号数据集
    url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"

    try:
        df = pd.read_csv(url)
        print(f"✓ 数据下载成功！数据形状: {df.shape}")

        # 保存到本地文件
        df.to_csv(local_data_file, index=False)
        print(f"✓ 数据已保存到本地: {local_data_file}")

    except Exception as e:
        print(f"✗ 从网络加载失败，尝试使用备用方案...")
        # 备用方案：使用seaborn内置数据集
        import seaborn as sns
        df = sns.load_dataset('titanic')
        # 重命名列以保持一致
        df = df.rename(columns={
            'pclass': 'Pclass',
            'sex': 'Sex',
            'age': 'Age',
            'sibsp': 'Siblings/Spouses Aboard',
            'parch': 'Parents/Children Aboard',
            'fare': 'Fare',
            'survived': 'Survived'
        })
        print(f"✓ 使用seaborn内置数据集！数据形状: {df.shape}")

        # 保存到本地文件
        df.to_csv(local_data_file, index=False)
        print(f"✓ 数据已保存到本地: {local_data_file}")

print("\n数据前5行:")
print(df.head())

print("\n数据信息:")
print(df.info())

print("\n数据统计摘要:")
print(df.describe())

print("\n缺失值统计:")
print(df.isnull().sum())

# ===== 第二部分：数据探索性分析 =====
print("\n" + "="*70)
print("数据探索性分析")
print("="*70)

# 生存人数统计
print("\n生存统计:")
survived_counts = df['Survived'].value_counts()
print(survived_counts)
print(f"生存率: {survived_counts[1] / len(df):.2%}")

# 按性别统计生存率
print("\n按性别统计生存率:")
survival_by_sex = df.groupby('Sex')['Survived'].mean()
print(survival_by_sex)

# 按客舱等级统计生存率
if 'Pclass' in df.columns:
    print("\n按客舱等级统计生存率:")
    survival_by_class = df.groupby('Pclass')['Survived'].mean()
    print(survival_by_class)

# 创建特征分析可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1：生存人数分布
ax1 = axes[0, 0]
survival_counts = df['Survived'].value_counts()
ax1.bar(['Not Survived', 'Survived'], [survival_counts[0], survival_counts[1]],
        color=['#ff6b6b', '#51cf66'], alpha=0.8)
ax1.set_title('Survival Count Distribution', fontsize=14, fontweight='bold')
ax1.set_ylabel('Count', fontsize=12)
for i, v in enumerate([survival_counts[0], survival_counts[1]]):
    ax1.text(i, v + 10, str(v), ha='center', fontsize=12, fontweight='bold')

# 图2：按性别的生存率
ax2 = axes[0, 1]
survival_by_sex.plot(kind='bar', ax=ax2, color=['#ff8787', '#74c0fc'], alpha=0.8)
ax2.set_title('Survival Rate by Sex', fontsize=14, fontweight='bold')
ax2.set_ylabel('Survival Rate', fontsize=12)
ax2.set_xlabel('Sex', fontsize=12)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
# 添加数值标签
for p in ax2.patches:
    ax2.annotate(f'{p.get_height():.2%}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# 图3：按客舱等级的生存率
if 'Pclass' in df.columns:
    ax3 = axes[1, 0]
    survival_by_class.plot(kind='bar', ax=ax3, color=['#ffd43b', '#fab005', '#fd7e14'], alpha=0.8)
    ax3.set_title('Survival Rate by Passenger Class', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Survival Rate', fontsize=12)
    ax3.set_xlabel('Passenger Class', fontsize=12)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
    for p in ax3.patches:
        ax3.annotate(f'{p.get_height():.2%}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# 图4：年龄分布
ax4 = axes[1, 1]
if 'Age' in df.columns:
    ax4.hist(df[df['Survived'] == 0]['Age'].dropna(), bins=20, alpha=0.6,
             label='Not Survived', color='#ff6b6b', density=True)
    ax4.hist(df[df['Survived'] == 1]['Age'].dropna(), bins=20, alpha=0.6,
             label='Survived', color='#51cf66', density=True)
    ax4.set_title('Age Distribution by Survival Status', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Age', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/zhengnan/Sniper/Developer/github/LearnAgent/ai_learn/logistic_regression/titanic_eda.png',
            dpi=100, bbox_inches='tight')
print("\n✓ 探索性分析图表已保存: titanic_eda.png")
plt.close()

# ===== 第三部分：数据预处理 =====
print("\n" + "="*70)
print("数据预处理")
print("="*70)

# 选择特征
features_to_use = []
if 'Pclass' in df.columns:
    features_to_use.append('Pclass')
if 'Sex' in df.columns:
    features_to_use.append('Sex')
if 'Age' in df.columns:
    features_to_use.append('Age')
if 'Siblings/Spouses Aboard' in df.columns:
    features_to_use.append('Siblings/Spouses Aboard')
if 'Parents/Children Aboard' in df.columns:
    features_to_use.append('Parents/Children Aboard')
if 'Fare' in df.columns:
    features_to_use.append('Fare')

print(f"\n使用的特征: {features_to_use}")

# 创建特征矩阵
X = df[features_to_use].copy()
y = df['Survived']

print(f"\n原始特征形状: {X.shape}")
print(f"标签形状: {y.shape}")

# 处理缺失值
print("\n处理缺失值...")
if X['Age'].isnull().sum() > 0:
    # 用中位数填充年龄
    age_median = X['Age'].median()
    X['Age'] = X['Age'].fillna(age_median)
    print(f"  年龄缺失值用中位数 {age_median:.1f} 填充")

# 编码类别变量
if 'Sex' in X.columns:
    # 性别编码：male -> 0, female -> 1
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
    print("  性别编码: male=0, female=1")

print("\n预处理后的特征:")
print(X.head())

print("\n特征统计:")
print(X.describe())

# ===== 第四部分：划分训练集和测试集 =====
print("\n" + "="*70)
print("划分数据集")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
print(f"\n训练集生存率: {y_train.mean():.2%}")
print(f"测试集生存率: {y_test.mean():.2%}")

# 标准化特征
print("\n标准化特征...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ 特征标准化完成")

# ===== 第五部分：训练逻辑回归模型 =====
print("\n" + "="*70)
print("训练逻辑回归模型")
print("="*70)

# 训练模型
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

print("\n✓ 模型训练完成")

# 模型参数
print("\n模型参数:")
feature_names = X.columns
coefficients = pd.DataFrame({
    '特征': feature_names,
    '系数': model.coef_[0]
})
coefficients = coefficients.sort_values('系数', ascending=False)
print(coefficients)

print(f"\n截距 (bias): {model.intercept_[0]:.4f}")

# 解释重要特征
print("\n特征重要性分析:")
for _, row in coefficients.iterrows():
    feature = row['特征']
    coef = row['系数']
    if coef > 0:
        print(f"  {feature}: 系数={coef:.4f} -> Positive impact on survival")
    else:
        print(f"  {feature}: 系数={coef:.4f} -> Negative impact on survival")

# ===== 第六部分：模型评估 =====
print("\n" + "="*70)
print("模型评估")
print("="*70)

# 预测
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# 准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.2%}")

# 详细分类报告
print("\n详细分类报告:")
print(classification_report(y_test, y_pred,
                          target_names=['Not Survived', 'Survived'],
                          digits=4))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵:")
print(cm)
print(f"True Negative (TN): {cm[0, 0]} - Correctly predicted Not Survived")
print(f"False Positive (FP): {cm[0, 1]} - Incorrectly predicted as Survived")
print(f"False Negative (FN): {cm[1, 0]} - Incorrectly predicted as Not Survived")
print(f"True Positive (TP): {cm[1, 1]} - Correctly predicted Survived")

# 计算更多指标
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # 召回率
specificity = tn / (tn + fp)  # 特异性
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

print(f"\n敏感性 (召回率): {sensitivity:.2%} - 实际存活者中被正确预测的比例")
print(f"特异性: {specificity:.2%} - 实际未存活者中被正确预测的比例")
print(f"精确率: {precision:.2%} - 预测为存活者中真正存活的比例")

# ===== 第七部分：ROC曲线和精确率-召回率曲线 =====
print("\n生成ROC曲线和精确率-召回率曲线...")

# 计算ROC曲线
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# 计算精确率-召回率曲线
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)

# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC曲线
ax1 = axes[0]
ax1.plot(fpr, tpr, color='#339af0', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random Classifier')
ax1.fill_between(fpr, tpr, alpha=0.3, color='#339af0')
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curve - Titanic Survival Prediction', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1.05])

# 精确率-召回率曲线
ax2 = axes[1]
ax2.plot(recall_curve, precision_curve, color='#ff6b6b', linewidth=2, label='PR Curve')
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax2.legend(loc='lower left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1.05])

# 标记最佳阈值点
# Youden's J统计量：最大化 (TPR - FPR)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds_roc[best_idx]
ax1.scatter(fpr[best_idx], tpr[best_idx], c='red', s=100, zorder=5,
           marker='o', label=f'Best Threshold = {best_threshold:.3f}')
ax1.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('/Users/zhengnan/Sniper/Developer/github/LearnAgent/ai_learn/logistic_regression/titanic_roc_pr.png',
            dpi=100, bbox_inches='tight')
print("✓ ROC和PR曲线已保存: titanic_roc_pr.png")
plt.close()

# ===== 第八部分：预测案例 =====
print("\n" + "="*70)
print("预测案例")
print("="*70)

# 创建几个测试案例
test_passengers = pd.DataFrame({
    'Pclass': [1, 3, 2, 3, 1],
    'Sex': [1, 0, 1, 0, 0],  # 1=female, 0=male
    'Age': [25, 30, 50, 5, 60],
    'Siblings/Spouses Aboard': [0, 1, 1, 0, 0],
    'Parents/Children Aboard': [0, 0, 0, 2, 0],
    'Fare': [100, 10, 25, 30, 50]
})

# 标准化
test_passengers_scaled = scaler.transform(test_passengers)

# 预测
predictions = model.predict(test_passengers_scaled)
probabilities = model.predict_proba(test_passengers_scaled)

print("\n预测结果:")
print("-" * 80)
for i, (passenger, pred, prob) in enumerate(zip(test_passengers.values, predictions, probabilities)):
    print(f"\nPassenger {i+1}:")
    print(f"  Passenger Class: {int(passenger[0])}")
    print(f"  Sex: {'Female' if passenger[1] == 1 else 'Male'}")
    print(f"  Age: {passenger[2]:.0f}")
    print(f"  Siblings/Spouses Aboard: {int(passenger[3])}")
    print(f"  Parents/Children Aboard: {int(passenger[4])}")
    print(f"  Fare: ${passenger[5]:.1f}")
    print(f"  Prediction: {'✓ Survived' if pred == 1 else '✗ Not Survived'}")
    print(f"  Survival Probability: {prob[1]:.2%}")

# ===== 第九部分：总结和学习要点 =====
print("\n" + "="*70)
print("核心学习要点")
print("="*70)

print("\n1. 逻辑回归在真实数据集上的应用")
print("   - 泰坦尼克号数据集是一个经典的二分类问题")
print("   - 目标：根据乘客特征预测是否存活")

print("\n2. 数据预处理的重要性")
print("   - 处理缺失值（年龄字段）")
print("   - 类别变量编码（性别）")
print("   - 特征标准化（提升模型性能）")

print("\n3. 特征工程")
print("   - 选择相关特征：客舱等级、性别、年龄等")
print("   - 理解每个特征对预测的影响方向和程度")

print("\n4. 模型评估")
print("   - 准确率：整体预测正确的比例")
print("   - 精确率：预测为存活中真正存活的比例")
print("   - 召回率：实际存活中被正确预测的比例")
print("   - ROC-AUC：模型区分能力")

print("\n5. 模型解释")
print("   - 正向特征（如女性、高舱位）提高存活概率")
print("   - 负向特征（如男性、低舱位）降低存活概率")
print("   - 系数大小表示特征的重要性")

print("\n6. 实战技巧")
print("   - 查看概率值而不只是类别标签")
print("   - 使用ROC曲线选择最佳阈值")
print("   - 关注混淆矩阵，理解假阳性和假阴性")
print("   - 对不平衡数据使用不同的评估指标")

print("\n" + "="*70)
print("下一步改进建议")
print("="*70)

print("\n1. 特征工程")
print("   - 创建新特征：家庭规模 = 配偶兄弟姐妹 + 父母子女")
print("   - 提取称谓（Mr., Mrs., Miss等）")
print("   - 年龄分段（儿童、成人、老人）")

print("\n2. 模型优化")
print("   - 尝试不同的正则化强度（C参数）")
print("   - 使用交叉验证选择最优参数")
print("   - 尝试其他分类算法（随机森林、XGBoost）")

print("\n3. 处理类别不平衡")
print("   - 使用类别权重：class_weight='balanced'")
print("   - 过采样或欠采样技术")

print("\n4. 深入分析")
print("   - 分析预测错误的案例")
print("   - 特征交互作用分析")
print("   - 部分依赖图可视化")

print("\n" + "="*70)
print("演示完成！")
print("="*70)
