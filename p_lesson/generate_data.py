import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

np.random.seed(42)

n_samples = 1000

age = np.random.randint(18, 80, n_samples)
gender = np.random.choice(['男', '女'], n_samples)
income = np.random.normal(50000, 20000, n_samples)
income = np.maximum(income, 5000)
income = np.minimum(income, 200000)

provinces = ['北京', '上海', '广东', '浙江', '江苏', '四川', '湖北', '山东', '河南', '福建']
province = np.random.choice(provinces, n_samples)

has_children = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
had_accident = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])

years_insured = np.random.randint(1, 10, n_samples)
premium_amount = np.random.normal(3000, 1000, n_samples)

renewal_prob = (
    0.3 + 
    (age - 18) / 62 * 0.15 +
    (income - 5000) / 195000 * 0.1 +
    has_children * 0.1 +
    years_insured / 10 * 0.2 -
    had_accident * 0.15
)

renewal_prob = np.clip(renewal_prob, 0.1, 0.9)
renewal = np.random.binomial(1, renewal_prob)

df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'income': income,
    'province': province,
    'has_children': has_children,
    'had_accident': had_accident,
    'years_insured': years_insured,
    'premium_amount': premium_amount,
    'renewal': renewal
})

output_path = '/Users/zhengnan/Sniper/Developer/github/LearnAgent/p_lesson/lesson_04_data.xlsx'
df.to_excel(output_path, index=False)

print(f"模拟数据已生成，共 {n_samples} 条记录")
print(f"保存路径: {output_path}")
print("\n数据统计:")
print(f"续约率: {df['renewal'].mean():.2%}")
print("\n前5行数据:")
print(df.head())
