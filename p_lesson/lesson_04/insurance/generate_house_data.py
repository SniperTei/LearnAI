import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 1500

square_feet = np.random.randint(500, 5000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
floors = np.random.randint(1, 3, n_samples)
waterfront = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
view = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.5, 0.25, 0.15, 0.07, 0.03])
condition = np.random.randint(1, 6, n_samples)
grade = np.random.randint(3, 13, n_samples)
sqft_above = square_feet * np.random.uniform(0.7, 0.95, n_samples)
sqft_basement = square_feet - sqft_above
yr_built = np.random.randint(1950, 2024, n_samples)
yr_renovated = np.zeros(n_samples, dtype=int)
yr_renovated_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
yr_renovated[yr_renovated_indices] = np.random.randint(2000, 2024, int(n_samples * 0.15))

price = (
    50000 +
    square_feet * 150 +
    bedrooms * 15000 +
    bathrooms * 20000 +
    floors * 25000 +
    waterfront * 150000 +
    view * 30000 +
    condition * 20000 +
    grade * 40000 +
    (yr_built - 1950) * 800 +
    np.where(yr_renovated > 0, (yr_renovated - 2000) * 1000, 0)
) * np.random.uniform(0.8, 1.2, n_samples)

price = np.maximum(price, 50000)

df = pd.DataFrame({
    'id': range(1, n_samples + 1),
    'price': price,
    'square_feet': square_feet,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'floors': floors,
    'waterfront': waterfront,
    'view': view,
    'condition': condition,
    'grade': grade,
    'sqft_above': sqft_above,
    'sqft_basement': sqft_basement,
    'yr_built': yr_built,
    'yr_renovated': yr_renovated
})

output_path = 'house_prices.csv'
df.to_csv(output_path, index=False)

print(f"房价预测数据集已生成，共 {n_samples} 条记录")
print(f"保存路径: {output_path}")
print(f"\n平均房价: ${df['price'].mean():,.0f}")
print(f"最低房价: ${df['price'].min():,.0f}")
print(f"最高房价: ${df['price'].max():,.0f}")
print("\n前5行数据:")
print(df.head())

print("\n数据集说明:")
print("这是一个经典的回归任务")
print("目标变量: price (房价，连续数值)")
print("任务: 根据房屋特征预测其售价")
