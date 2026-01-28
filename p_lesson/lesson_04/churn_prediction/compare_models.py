import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
df = pd.read_csv('customer_churn.csv')

# 2. 数据预处理 - 特征编码
le = LabelEncoder()
df['contract_type_encoded'] = le.fit_transform(df['contract_type'])
df['internet_service_encoded'] = le.fit_transform(df['internet_service'])
df['online_security_encoded'] = le.fit_transform(df['online_security'])
df['tech_support_encoded'] = le.fit_transform(df['tech_support'])
df['paperless_billing_encoded'] = le.fit_transform(df['paperless_billing'])
df['payment_method_encoded'] = le.fit_transform(df['payment_method'])

# 3. 准备特征和目标变量
feature_columns = ['age', 'tenure', 'contract_type_encoded', 'internet_service_encoded', 
                    'online_security_encoded', 'tech_support_encoded', 
                    'paperless_billing_encoded', 'payment_method_encoded', 
                    'monthly_charges', 'total_charges']

x = df[feature_columns]
y = df['churn']

# 4. 数据标准化
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled, columns=feature_columns)

# 5. 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

print("=" * 60)
print("数据集信息")
print("=" * 60)
print(f"总样本数: {len(df)}")
print(f"训练集: {len(x_train)}")
print(f"测试集: {len(x_test)}")
print(f"特征数: {len(feature_columns)}")
print()

# 6. 训练决策树模型
print("=" * 60)
print("训练决策树模型")
print("=" * 60)
dt_model = DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced')
dt_model.fit(x_train, y_train)
dt_pred = dt_model.predict(x_test)

# 评估决策树
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)

print(f"准确率 (Accuracy): {dt_accuracy:.4f}")
print(f"召回率 (Recall): {dt_recall:.4f}")
print(f"精确率 (Precision): {dt_precision:.4f}")
print(f"F1分数 (F1 Score): {dt_f1:.4f}")
print("\n分类报告:\n", classification_report(y_test, dt_pred))

# 7. 训练随机森林模型
print("\n" + "=" * 60)
print("训练随机森林模型")
print("=" * 60)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced')
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)

# 评估随机森林
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(f"准确率 (Accuracy): {rf_accuracy:.4f}")
print(f"召回率 (Recall): {rf_recall:.4f}")
print(f"精确率 (Precision): {rf_precision:.4f}")
print(f"F1分数 (F1 Score): {rf_f1:.4f}")
print("\n分类报告:\n", classification_report(y_test, rf_pred))

# 8. 模型对比
print("\n" + "=" * 60)
print("模型性能对比")
print("=" * 60)
comparison_df = pd.DataFrame({
    '模型': ['逻辑回归', '决策树', '随机森林'],
    '准确率': [0.5950, dt_accuracy, rf_accuracy],
    '召回率': [0.6380, dt_recall, rf_recall],
    '精确率': [0.3611, dt_precision, rf_precision],
    'F1分数': [0.4612, dt_f1, rf_f1]
})

print(comparison_df.to_string(index=False))

# 9. 可视化模型性能对比
print("\n" + "=" * 60)
print("生成可视化图表...")
print("=" * 60)

# 创建图表布局
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('模型性能对比分析', fontsize=16, fontweight='bold')

# 准备数据
models = comparison_df['模型'].values
metrics = ['准确率', '召回率', '精确率', 'F1分数']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

# 图表1: 各指标对比柱状图
x = np.arange(len(models))
width = 0.2

for i, metric in enumerate(metrics):
    axes[0, 0].bar(x + i*width, comparison_df[metric].values, width, 
                   label=metric, color=colors[i], alpha=0.8)

axes[0, 0].set_xlabel('模型', fontsize=11)
axes[0, 0].set_ylabel('分数', fontsize=11)
axes[0, 0].set_title('各模型性能指标对比', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(x + width * 1.5)
axes[0, 0].set_xticklabels(models, fontsize=10)
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].set_ylim(0, 1.0)

# 图表2: F1分数对比
axes[0, 1].bar(models, comparison_df['F1分数'].values, color='#9b59b6', alpha=0.8)
axes[0, 1].set_xlabel('模型', fontsize=11)
axes[0, 1].set_ylabel('F1分数', fontsize=11)
axes[0, 1].set_title('F1分数对比', fontsize=12, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)
axes[0, 1].set_ylim(0, 1.0)

# 在柱子上添加数值标签
for i, v in enumerate(comparison_df['F1分数'].values):
    axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# 图表3: 准确率 vs 召回率散点图
axes[1, 0].scatter(comparison_df['准确率'], comparison_df['召回率'], 
                  s=300, alpha=0.6, c=range(len(models)), cmap='viridis')
for i, model in enumerate(models):
    axes[1, 0].annotate(model, 
                      (comparison_df['准确率'][i], comparison_df['召回率'][i]),
                      xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

axes[1, 0].set_xlabel('准确率', fontsize=11)
axes[1, 0].set_ylabel('召回率', fontsize=11)
axes[1, 0].set_title('准确率 vs 召回率权衡', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_xlim(0, 1.0)
axes[1, 0].set_ylim(0, 1.0)

# 图表4: 综合雷达图
categories = ['准确率', '召回率', '精确率', 'F1分数']
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax_radar = axes[1, 1]
ax_radar = plt.subplot(2, 2, 4, projection='polar')

for idx, model in enumerate(models):
    values = comparison_df.iloc[idx][metrics].values.tolist()
    values += values[:1]
    ax_radar.plot(angles, values, 'o-', linewidth=2, label=model)
    ax_radar.fill(angles, values, alpha=0.15)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(categories, fontsize=9)
ax_radar.set_ylim(0, 1.0)
ax_radar.set_title('模型性能雷达图', fontsize=12, fontweight='bold', pad=20)
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax_radar.grid(True)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("可视化图表已保存为 'model_comparison.png'")
plt.show()

# 额外生成一个单独的详细对比表格图
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

table_data = comparison_df.values
table = ax.table(cellText=table_data, colLabels=comparison_df.columns, 
                cellLoc='center', loc='center',
                colColours=['#3498db']*len(comparison_df.columns))

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

for i in range(len(table_data)):
    for j in range(len(table_data[0])):
        if j > 0:
            value = table_data[i][j]
            if value > 0.6:
                table[(i+1, j)].set_facecolor('#d5f5e3')
            elif value > 0.4:
                table[(i+1, j)].set_facecolor('#fdebd0')
            else:
                table[(i+1, j)].set_facecolor('#fadbd8')

table[(0, 0)].set_facecolor('#3498db')
for j in range(1, len(comparison_df.columns)):
    table[(0, j)].set_facecolor('#3498db')
    table[(0, j)].get_text().set_color('white')

plt.title('模型性能对比表', fontsize=14, fontweight='bold', pad=10)
plt.savefig('comparison_table.png', dpi=300, bbox_inches='tight')
print("对比表格已保存为 'comparison_table.png'")
plt.close()
