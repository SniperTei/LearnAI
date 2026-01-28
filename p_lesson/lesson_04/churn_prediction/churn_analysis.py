import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

file_path = 'customer_churn.csv'
df = pd.read_csv(file_path)

print("=" * 60)
print("客户流失预测 - 分类任务")
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
print("目标变量分布 (流失情况):")
print("=" * 60)
print(df['churn'].value_counts())
print(f"流失率: {df['churn'].mean():.2%}")

print("\n" + "=" * 60)
print("分类变量分布:")
print("=" * 60)
print(f"\n合同类型分布:\n{df['contract_type'].value_counts()}")
print(f"\n互联网服务分布:\n{df['internet_service'].value_counts()}")
print(f"\n在线安全分布:\n{df['online_security'].value_counts()}")
print(f"\n技术支持分布:\n{df['tech_support'].value_counts()}")
print(f"\n无纸账单分布:\n{df['paperless_billing'].value_counts()}")
print(f"\n支付方式分布:\n{df['payment_method'].value_counts()}")

print("\n" + "=" * 60)
print("开始数据可视化...")
print("=" * 60)

fig = plt.figure(figsize=(20, 14))

plt.subplot(3, 3, 1)
sns.histplot(data=df, x='age', bins=20, kde=True)
plt.title('客户年龄分布', fontsize=12, fontweight='bold')
plt.xlabel('年龄')
plt.ylabel('频数')

plt.subplot(3, 3, 2)
sns.histplot(data=df, x='tenure', bins=20, kde=True)
plt.title('入网时长分布', fontsize=12, fontweight='bold')
plt.xlabel('入网时长（月）')
plt.ylabel('频数')

plt.subplot(3, 3, 3)
sns.histplot(data=df, x='monthly_charges', bins=20, kde=True)
plt.title('月消费金额分布', fontsize=12, fontweight='bold')
plt.xlabel('月消费金额（$）')
plt.ylabel('频数')

plt.subplot(3, 3, 4)
df['churn'].value_counts().plot(kind='bar', color=['#3498db', '#e74c3c'])
plt.title('流失情况分布', fontsize=12, fontweight='bold')
plt.xlabel('流失 (0=不流失, 1=流失)')
plt.ylabel('客户数')
plt.xticks(rotation=0)

plt.subplot(3, 3, 5)
df['contract_type'].value_counts().plot(kind='bar')
plt.title('合同类型分布', fontsize=12, fontweight='bold')
plt.xlabel('合同类型')
plt.ylabel('客户数')
plt.xticks(rotation=45)

plt.subplot(3, 3, 6)
df['internet_service'].value_counts().plot(kind='bar')
plt.title('互联网服务类型分布', fontsize=12, fontweight='bold')
plt.xlabel('互联网服务')
plt.ylabel('客户数')
plt.xticks(rotation=45)

plt.subplot(3, 3, 7)
sns.boxplot(x='churn', y='age', data=df)
plt.title('年龄 vs 流失', fontsize=12, fontweight='bold')
plt.xlabel('流失 (0=不流失, 1=流失)')
plt.ylabel('年龄')

plt.subplot(3, 3, 8)
sns.boxplot(x='churn', y='monthly_charges', data=df)
plt.title('月消费金额 vs 流失', fontsize=12, fontweight='bold')
plt.xlabel('流失 (0=不流失, 1=流失)')
plt.ylabel('月消费金额（$）')

plt.subplot(3, 3, 9)
churn_by_contract = df.groupby('contract_type')['churn'].mean().sort_values(ascending=True)
churn_by_contract.plot(kind='barh')
plt.title('各合同类型流失率', fontsize=12, fontweight='bold')
plt.xlabel('流失率')
plt.ylabel('合同类型')
plt.axvline(x=df['churn'].mean(), color='red', linestyle='--', label='平均流失率')
plt.legend()

plt.tight_layout()
plt.savefig('churn_data_visualization.png', dpi=300, bbox_inches='tight')
print("数据可视化已保存为: churn_data_visualization.png")

plt.figure(figsize=(16, 8))

plt.subplot(2, 4, 1)
churn_by_payment = df.groupby('payment_method')['churn'].mean()
churn_by_payment.plot(kind='bar', color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
plt.title('支付方式 vs 流失率', fontsize=12, fontweight='bold')
plt.xlabel('支付方式')
plt.ylabel('流失率')
plt.xticks(rotation=45)
plt.axhline(y=df['churn'].mean(), color='red', linestyle='--', label='平均流失率')
plt.legend()

plt.subplot(2, 4, 2)
churn_by_internet = df.groupby('internet_service')['churn'].mean()
churn_by_internet.plot(kind='bar', color=['#3498db', '#e74c3c', '#2ecc71'])
plt.title('互联网服务 vs 流失率', fontsize=12, fontweight='bold')
plt.xlabel('互联网服务')
plt.ylabel('流失率')
plt.xticks(rotation=45)
plt.axhline(y=df['churn'].mean(), color='red', linestyle='--', label='平均流失率')
plt.legend()

plt.subplot(2, 4, 3)
churn_by_security = df.groupby('online_security')['churn'].mean()
churn_by_security.plot(kind='bar', color=['#3498db', '#e74c3c', '#2ecc71'])
plt.title('在线安全 vs 流失率', fontsize=12, fontweight='bold')
plt.xlabel('在线安全')
plt.ylabel('流失率')
plt.xticks(rotation=45)
plt.axhline(y=df['churn'].mean(), color='red', linestyle='--', label='平均流失率')
plt.legend()

plt.subplot(2, 4, 4)
churn_by_support = df.groupby('tech_support')['churn'].mean()
churn_by_support.plot(kind='bar', color=['#3498db', '#e74c3c', '#2ecc71'])
plt.title('技术支持 vs 流失率', fontsize=12, fontweight='bold')
plt.xlabel('技术支持')
plt.ylabel('流失率')
plt.xticks(rotation=45)
plt.axhline(y=df['churn'].mean(), color='red', linestyle='--', label='平均流失率')
plt.legend()

plt.subplot(2, 4, 5)
sns.boxplot(x='churn', y='tenure', data=df)
plt.title('入网时长 vs 流失', fontsize=12, fontweight='bold')
plt.xlabel('流失 (0=不流失, 1=流失)')
plt.ylabel('入网时长（月）')

plt.subplot(2, 4, 6)
sns.boxplot(x='churn', y='total_charges', data=df)
plt.title('总消费金额 vs 流失', fontsize=12, fontweight='bold')
plt.xlabel('流失 (0=不流失, 1=流失)')
plt.ylabel('总消费金额（$）')

plt.subplot(2, 4, 7)
df['paperless_billing'].value_counts().plot(kind='bar', color=['#3498db', '#e74c3c'])
plt.title('无纸账单分布', fontsize=12, fontweight='bold')
plt.xlabel('无纸账单 (No/Yes)')
plt.ylabel('客户数')
plt.xticks(rotation=0)

plt.subplot(2, 4, 8)
churn_by_paperless = df.groupby('paperless_billing')['churn'].mean()
churn_by_paperless.plot(kind='bar', color=['#3498db', '#e74c3c'])
plt.title('无纸账单 vs 流失率', fontsize=12, fontweight='bold')
plt.xlabel('无纸账单 (No/Yes)')
plt.ylabel('流失率')
plt.xticks(rotation=0)
plt.axhline(y=df['churn'].mean(), color='red', linestyle='--', label='平均流失率')
plt.legend()

plt.tight_layout()
plt.savefig('churn_analysis.png', dpi=300, bbox_inches='tight')
print("流失分析图表已保存为: churn_analysis.png")

print("\n" + "=" * 60)
print("开始数据预处理和模型训练...")
print("=" * 60)

df_processed = df.copy()

le_contract = LabelEncoder()
df_processed['contract_type_encoded'] = le_contract.fit_transform(df_processed['contract_type'])

le_internet = LabelEncoder()
df_processed['internet_service_encoded'] = le_internet.fit_transform(df_processed['internet_service'])

le_online_security = LabelEncoder()
df_processed['online_security_encoded'] = le_online_security.fit_transform(df_processed['online_security'])

le_tech_support = LabelEncoder()
df_processed['tech_support_encoded'] = le_tech_support.fit_transform(df_processed['tech_support'])

le_paperless = LabelEncoder()
df_processed['paperless_billing_encoded'] = le_paperless.fit_transform(df_processed['paperless_billing'])

le_payment = LabelEncoder()
df_processed['payment_method_encoded'] = le_payment.fit_transform(df_processed['payment_method'])

X = df_processed.drop(['customer_id', 'contract_type', 'internet_service', 'online_security', 
                        'tech_support', 'paperless_billing', 'payment_method', 'churn'], axis=1)
y = df_processed['churn']

print(f"\n特征数量: {X.shape[1]}")
print(f"特征列表: {list(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\n训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")

print("\n" + "=" * 60)
print("开始逻辑回归模型训练...")
print("=" * 60)

logistic_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
logistic_model.fit(X_train, y_train)

y_pred_lr = logistic_model.predict(X_test)
y_pred_proba_lr = logistic_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("逻辑回归模型性能评估")
print("=" * 60)
print(f"准确率: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"精确率: {precision_score(y_test, y_pred_lr):.4f}")
print(f"召回率: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1分数: {f1_score(y_test, y_pred_lr):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_lr))

cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('逻辑回归混淆矩阵', fontsize=14, fontweight='bold')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.savefig('churn_logistic_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("混淆矩阵已保存为: churn_logistic_confusion_matrix.png")

print("\n" + "=" * 60)
print("开始决策树模型训练...")
print("=" * 60)

decision_tree = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
decision_tree.fit(X_train, y_train)

y_pred_dt = decision_tree.predict(X_test)

print("\n" + "=" * 60)
print("决策树模型性能评估")
print("=" * 60)
print(f"准确率: {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"精确率: {precision_score(y_test, y_pred_dt):.4f}")
print(f"召回率: {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1分数: {f1_score(y_test, y_pred_dt):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('决策树混淆矩阵', fontsize=14, fontweight='bold')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.savefig('churn_dt_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("混淆矩阵已保存为: churn_dt_confusion_matrix.png")

feature_importance_dt = pd.DataFrame({
    'feature': X.columns,
    'importance': decision_tree.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 60)
print("决策树特征重要性")
print("=" * 60)
print(feature_importance_dt.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("开始随机森林模型训练...")
print("=" * 60)

random_forest = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, class_weight='balanced')
random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_test)

print("\n" + "=" * 60)
print("随机森林模型性能评估")
print("=" * 60)
print(f"准确率: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"精确率: {precision_score(y_test, y_pred_rf):.4f}")
print(f"召回率: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1分数: {f1_score(y_test, y_pred_rf):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Purples', cbar=False)
plt.title('随机森林混淆矩阵', fontsize=14, fontweight='bold')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.savefig('churn_rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("混淆矩阵已保存为: churn_rf_confusion_matrix.png")

feature_importance_rf = pd.DataFrame({
    'feature': X.columns,
    'importance': random_forest.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "=" * 60)
print("随机森林特征重要性")
print("=" * 60)
print(feature_importance_rf.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("模型对比分析...")
print("=" * 60)

models_comparison = pd.DataFrame({
    'Model': ['逻辑回归', '决策树', '随机森林'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_rf)
    ],
    'Precision': [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_dt),
        precision_score(y_test, y_pred_rf)
    ],
    'Recall': [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_dt),
        recall_score(y_test, y_pred_rf)
    ],
    'F1': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_dt),
        f1_score(y_test, y_pred_rf)
    ]
})

print("\n" + "=" * 60)
print("模型性能对比表")
print("=" * 60)
print(models_comparison.to_string(index=False))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(models_comparison['Model'], models_comparison['Accuracy'], color=['#3498db', '#2ecc71', '#9b59b6'])
plt.title('准确率对比', fontsize=12, fontweight='bold')
plt.ylabel('准确率')
plt.ylim(0, 1)
for i, v in enumerate(models_comparison['Accuracy']):
    plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)

plt.subplot(1, 2, 2)
x = np.arange(len(models_comparison['Model']))
width = 0.2
plt.bar(x - width, models_comparison['Precision'], width, label='精确率', color='#3498db')
plt.bar(x, models_comparison['Recall'], width, label='召回率', color='#2ecc71')
plt.bar(x + width, models_comparison['F1'], width, label='F1分数', color='#9b59b6')
plt.title('多指标对比', fontsize=12, fontweight='bold')
plt.ylabel('分数')
plt.xticks(x, models_comparison['Model'])
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('churn_models_comparison.png', dpi=300, bbox_inches='tight')
print("模型对比图已保存为: churn_models_comparison.png")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
top_features_dt = feature_importance_dt.head(8)
plt.barh(range(len(top_features_dt)), top_features_dt['importance'][::-1])
plt.yticks(range(len(top_features_dt)), top_features_dt['feature'][::-1])
plt.xlabel('特征重要性', fontsize=12)
plt.title('决策树 - Top 8 特征重要性', fontsize=12, fontweight='bold')

plt.subplot(1, 2, 2)
top_features_rf = feature_importance_rf.head(8)
plt.barh(range(len(top_features_rf)), top_features_rf['importance'][::-1])
plt.yticks(range(len(top_features_rf)), top_features_rf['feature'][::-1])
plt.xlabel('特征重要性', fontsize=12)
plt.title('随机森林 - Top 8 特征重要性', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('churn_feature_importance.png', dpi=300, bbox_inches='tight')
print("特征重要性对比图已保存为: churn_feature_importance.png")

best_model = models_comparison.loc[models_comparison['Accuracy'].idxmax()]

print("\n" + "=" * 60)
print("客户流失预测分析完成!")
print("=" * 60)
print(f"\n最佳模型: {best_model['Model']} (准确率: {best_model['Accuracy']:.4f})")
print("\n主要发现:")
print(f"1. 客户流失率: {df['churn'].mean():.2%}")
print(f"2. 最重要的流失影响因素: {feature_importance_rf.iloc[0]['feature']}")
print(f"3. 第二重要因素: {feature_importance_rf.iloc[1]['feature']}")
print(f"4. 第三重要因素: {feature_importance_rf.iloc[2]['feature']}")
print("\n业务建议:")
print(f"- 重点监控{feature_importance_rf.iloc[0]['feature']}指标")
print(f"- 对高风险客户提前采取挽留措施")
print(f"- 优化{feature_importance_rf.iloc[1]['feature']}相关服务")

print("\n" + "=" * 60)
print("生成的文件:")
print("=" * 60)
print("1. churn_data_visualization.png - 数据可视化")
print("2. churn_analysis.png - 流失分析图表")
print("3. churn_logistic_confusion_matrix.png - 逻辑回归混淆矩阵")
print("4. churn_dt_confusion_matrix.png - 决策树混淆矩阵")
print("5. churn_rf_confusion_matrix.png - 随机森林混淆矩阵")
print("6. churn_models_comparison.png - 模型对比图")
print("7. churn_feature_importance.png - 特征重要性对比图")
