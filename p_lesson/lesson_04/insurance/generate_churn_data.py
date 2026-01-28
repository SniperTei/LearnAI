import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 2000

age = np.random.randint(18, 80, n_samples)
tenure = np.random.randint(1, 10, n_samples)
monthly_charges = np.random.uniform(20, 120, n_samples)
total_charges = tenure * monthly_charges * np.random.uniform(0.9, 1.1, n_samples)

contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2])
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2])
online_security = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.6, 0.1])
tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.6, 0.1])
paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4])
payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples, p=[0.4, 0.2, 0.2, 0.2])

churn_prob = (
    0.1 +
    (age - 18) / 62 * -0.05 +
    (tenure - 1) / 9 * -0.15 +
    (monthly_charges - 20) / 100 * 0.2 +
    (np.where(contract_type == 'Month-to-month', 0.15, 0)) +
    (np.where(internet_service == 'Fiber optic', 0.1, 0)) +
    (np.where(payment_method == 'Electronic check', 0.08, 0))
)

churn_prob = np.clip(churn_prob, 0.05, 0.7)
churn = np.random.binomial(1, churn_prob)

df = pd.DataFrame({
    'customer_id': range(1, n_samples + 1),
    'age': age,
    'tenure': tenure,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'contract_type': contract_type,
    'internet_service': internet_service,
    'online_security': online_security,
    'tech_support': tech_support,
    'paperless_billing': paperless_billing,
    'payment_method': payment_method,
    'churn': churn
})

output_path = 'customer_churn.csv'
df.to_csv(output_path, index=False)

print(f"客户流失预测数据集已生成，共 {n_samples} 条记录")
print(f"保存路径: {output_path}")
print(f"\n流失率: {df['churn'].mean():.2%}")
print("\n前5行数据:")
print(df.head())

print("\n数据集说明:")
print("这是一个经典的分类任务")
print("目标变量: churn (0=不流失, 1=流失)")
print("任务: 根据客户特征预测其是否会流失")
