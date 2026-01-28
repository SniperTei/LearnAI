import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

file_path = 'insurance.csv'
df = pd.read_csv(file_path)

print("=" * 60)
print("保险费用预测数据集 - 这是一个回归任务！")
print("=" * 60)

print("\n" + "=" * 60)
print("前10行数据:")
print("=" * 60)
print(df.head(10))

print("\n" + "=" * 60)
print("数据基本信息:")
print("=" * 60)
print(df.info())

print("\n" + "=" * 60)
print("数据统计描述:")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("分类变量分布:")
print("=" * 60)
print(f"\n性别分布:\n{df['sex'].value_counts()}")
print(f"\n是否吸烟:\n{df['smoker'].value_counts()}")
print(f"\n地区分布:\n{df['region'].value_counts()}")

print("\n" + "=" * 60)
print("目标变量: charges (医疗费用)")
print("=" * 60)
print(f"平均费用: ${df['charges'].mean():,.2f}")
print(f"最低费用: ${df['charges'].min():,.2f}")
print(f"最高费用: ${df['charges'].max():,.2f}")
print(f"中位数费用: ${df['charges'].median():,.2f}")

print("\n" + "=" * 60)
print("重要说明: 这是一个回归任务，不是分类任务！")
print("=" * 60)
print("分类任务: 预测类别（如：会续约/不会续约，0/1）")
print("回归任务: 预测连续数值（如：医疗费用 $10,000）")
print(f"\n本任务: 根据客户的特征预测其医疗费用")

print("\n" + "=" * 60)
print("开始数据可视化...")
print("=" * 60)

fig = plt.figure(figsize=(18, 12))

plt.subplot(3, 3, 1)
sns.histplot(df['charges'], bins=30, kde=True)
plt.title('医疗费用分布', fontsize=12, fontweight='bold')
plt.xlabel('医疗费用 ($)')
plt.ylabel('频数')

plt.subplot(3, 3, 2)
sns.boxplot(x='smoker', y='charges', data=df)
plt.title('吸烟者 vs 费用', fontsize=12, fontweight='bold')
plt.xlabel('是否吸烟')
plt.ylabel('医疗费用 ($)')

plt.subplot(3, 3, 3)
sns.boxplot(x='sex', y='charges', data=df)
plt.title('性别 vs 费用', fontsize=12, fontweight='bold')
plt.xlabel('性别')
plt.ylabel('医疗费用 ($)')

plt.subplot(3, 3, 4)
sns.scatterplot(x='age', y='charges', data=df, alpha=0.5)
plt.title('年龄 vs 费用', fontsize=12, fontweight='bold')
plt.xlabel('年龄')
plt.ylabel('医疗费用 ($)')

plt.subplot(3, 3, 5)
sns.scatterplot(x='bmi', y='charges', data=df, alpha=0.5)
plt.title('BMI vs 费用', fontsize=12, fontweight='bold')
plt.xlabel('BMI')
plt.ylabel('医疗费用 ($)')

plt.subplot(3, 3, 6)
sns.boxplot(x='children', y='charges', data=df)
plt.title('孩子数量 vs 费用', fontsize=12, fontweight='bold')
plt.xlabel('孩子数量')
plt.ylabel('医疗费用 ($)')

plt.subplot(3, 3, 7)
sns.boxplot(x='region', y='charges', data=df)
plt.title('地区 vs 费用', fontsize=12, fontweight='bold')
plt.xlabel('地区')
plt.ylabel('医疗费用 ($)')
plt.xticks(rotation=45)

plt.subplot(3, 3, 8)
sns.boxplot(x='region', y='charges', hue='smoker', data=df)
plt.title('地区 vs 费用 (按吸烟分组)', fontsize=12, fontweight='bold')
plt.xlabel('地区')
plt.ylabel('医疗费用 ($)')
plt.xticks(rotation=45)

plt.subplot(3, 3, 9)
numeric_cols = ['age', 'bmi', 'children', 'charges']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('数值型变量相关性热力图', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('insurance_data_visualization.png', dpi=300, bbox_inches='tight')
print("数据可视化已保存为: insurance_data_visualization.png")

print("\n" + "=" * 60)
print("开始数据预处理...")
print("=" * 60)

df_processed = df.copy()

le_sex = LabelEncoder()
df_processed['sex_encoded'] = le_sex.fit_transform(df_processed['sex'])

le_smoker = LabelEncoder()
df_processed['smoker_encoded'] = le_smoker.fit_transform(df_processed['smoker'])

region_dummies = pd.get_dummies(df_processed['region'], prefix='region')
df_processed = pd.concat([df_processed, region_dummies], axis=1)

X = df_processed.drop(['sex', 'smoker', 'region', 'charges'], axis=1)
y = df_processed['charges']

print(f"\n特征数量: {X.shape[1]}")
print(f"特征列表: {list(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\n训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")

print("\n" + "=" * 60)
print("开始模型训练 - 线性回归...")
print("=" * 60)

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

y_pred_linear = linear_reg.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("\n" + "=" * 60)
print("线性回归模型性能评估")
print("=" * 60)
print(f"均方误差 (MSE): {mse_linear:,.2f}")
print(f"均方根误差 (RMSE): {rmse_linear:,.2f}")
print(f"平均绝对误差 (MAE): {mae_linear:,.2f}")
print(f"R²分数: {r2_linear:.4f}")

coefficients = pd.DataFrame({
    'feature': X.columns,
    'coefficient': linear_reg.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print("\n" + "=" * 60)
print("线性回归系数 (按绝对值排序)")
print("=" * 60)
print(coefficients.to_string(index=False))

print("\n" + "=" * 60)
print("开始模型训练 - 决策树回归...")
print("=" * 60)

decision_tree_reg = DecisionTreeRegressor(max_depth=4, random_state=42)
decision_tree_reg.fit(X_train, y_train)

y_pred_tree = decision_tree_reg.predict(X_test)

mse_tree = mean_squared_error(y_test, y_pred_tree)
rmse_tree = np.sqrt(mse_tree)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print("\n" + "=" * 60)
print("决策树回归模型性能评估")
print("=" * 60)
print(f"均方误差 (MSE): {mse_tree:,.2f}")
print(f"均方根误差 (RMSE): {rmse_tree:,.2f}")
print(f"平均绝对误差 (MAE): {mae_tree:,.2f}")
print(f"R²分数: {r2_tree:.4f}")

feature_importance_tree = pd.DataFrame({
    'feature': X.columns,
    'importance': decision_tree_reg.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 60)
print("决策树特征重要性")
print("=" * 60)
print(feature_importance_tree.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("开始模型训练 - 随机森林回归...")
print("=" * 60)

random_forest_reg = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
random_forest_reg.fit(X_train, y_train)

y_pred_rf = random_forest_reg.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n" + "=" * 60)
print("随机森林回归模型性能评估")
print("=" * 60)
print(f"均方误差 (MSE): {mse_rf:,.2f}")
print(f"均方根误差 (RMSE): {rmse_rf:,.2f}")
print(f"平均绝对误差 (MAE): {mae_rf:,.2f}")
print(f"R²分数: {r2_rf:.4f}")

feature_importance_rf = pd.DataFrame({
    'feature': X.columns,
    'importance': random_forest_reg.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 60)
print("随机森林特征重要性")
print("=" * 60)
print(feature_importance_rf.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("模型对比分析...")
print("=" * 60)

models_comparison_reg = pd.DataFrame({
    'Model': ['线性回归', '决策树回归', '随机森林回归'],
    'RMSE': [rmse_linear, rmse_tree, rmse_rf],
    'MAE': [mae_linear, mae_tree, mae_rf],
    'R²': [r2_linear, r2_tree, r2_rf]
})

print("\n" + "=" * 60)
print("模型性能对比表")
print("=" * 60)
print(models_comparison_reg.to_string(index=False))

best_model_reg = models_comparison_reg.loc[models_comparison_reg['RMSE'].idxmin()]

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.scatter(y_test, y_pred_linear, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('线性回归: 预测值 vs 真实值', fontsize=12, fontweight='bold')
plt.xlabel('真实值')
plt.ylabel('预测值')

plt.subplot(2, 3, 2)
plt.scatter(y_test, y_pred_tree, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('决策树: 预测值 vs 真实值', fontsize=12, fontweight='bold')
plt.xlabel('真实值')
plt.ylabel('预测值')

plt.subplot(2, 3, 3)
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('随机森林: 预测值 vs 真实值', fontsize=12, fontweight='bold')
plt.xlabel('真实值')
plt.ylabel('预测值')

plt.subplot(2, 3, 4)
plt.bar(models_comparison_reg['Model'], models_comparison_reg['RMSE'], color=['#3498db', '#2ecc71', '#9b59b6'])
plt.title('RMSE对比 (越小越好)', fontsize=12, fontweight='bold')
plt.ylabel('RMSE')

plt.subplot(2, 3, 5)
plt.bar(models_comparison_reg['Model'], models_comparison_reg['MAE'], color=['#3498db', '#2ecc71', '#9b59b6'])
plt.title('MAE对比 (越小越好)', fontsize=12, fontweight='bold')
plt.ylabel('MAE')

plt.subplot(2, 3, 6)
plt.bar(models_comparison_reg['Model'], models_comparison_reg['R²'], color=['#3498db', '#2ecc71', '#9b59b6'])
plt.title('R²对比 (越大越好)', fontsize=12, fontweight='bold')
plt.ylabel('R²')
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('insurance_models_comparison.png', dpi=300, bbox_inches='tight')
print("模型对比图已保存为: insurance_models_comparison.png")

print("\n" + "=" * 60)
print("所有任务完成!")
print("=" * 60)
print("\n生成的文件:")
print("1. insurance_data_visualization.png - 数据可视化")
print("2. insurance_models_comparison.png - 模型对比图")

print("\n" + "=" * 60)
print("回归任务 vs 分类任务对比")
print("=" * 60)
print("分类任务 (之前的lesson_04):")
print("- 目标: 预测类别 (续约/不续约)")
print("- 评估指标: 准确率、精确率、召回率、F1分数")
print("- 输出: 0 或 1")
print("\n回归任务 (当前的insurance):")
print("- 目标: 预测连续数值 (医疗费用)")
print("- 评估指标: RMSE、MAE、R²")
print("- 输出: $12,345.67 (具体金额)")
print("\n关键区别:")
print("- 分类: 判断是A还是B")
print("- 回归: 预测具体数值")
