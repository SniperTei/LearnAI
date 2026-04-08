"""合成银行客户数据生成器"""
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_CUSTOMERS, RANDOM_SEED, PRODUCTS, CUSTOMER_TIERS, HIGH_VALUE_THRESHOLD, DATA_PATH

np.random.seed(RANDOM_SEED)


def generate_customers(n=NUM_CUSTOMERS):
    """生成模拟银行客户数据"""
    customers = []

    for i in range(n):
        cust_id = f"C{10000 + i}"

        # 基本信息
        age = int(np.clip(np.random.normal(45, 15), 18, 80))
        gender = np.random.choice(["男", "女"], p=[0.55, 0.45])
        city = np.random.choice(
            ["北京", "上海", "广州", "深圳", "杭州", "成都", "南京", "武汉", "重庆", "苏州"],
            p=[0.15, 0.15, 0.10, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08, 0.10],
        )
        occupation = np.random.choice(
            ["企业主", "高管", "专业人士", "公务员", "退休", "白领", "自由职业"],
            p=[0.15, 0.10, 0.15, 0.10, 0.15, 0.20, 0.15],
        )

        # 资产类数据 - 使用对数正态分布模拟真实财富分布
        base_wealth = np.random.lognormal(mean=14.5, sigma=1.2)
        # 企业主和高管资产更高
        if occupation in ["企业主", "高管"]:
            base_wealth *= np.random.uniform(1.5, 3.0)
        # 年龄因素：中年客户资产更高
        age_factor = 1.0 + 0.3 * np.sin(np.pi * (age - 30) / 60) if 30 < age < 60 else 0.8
        base_wealth *= age_factor

        aum = round(base_wealth, 2)
        deposit = round(aum * np.random.uniform(0.05, 0.4), 2)
        wealth_mgmt = round(aum * np.random.uniform(0, 0.3), 2)
        fund = round(aum * np.random.uniform(0, 0.25), 2)
        insurance = round(aum * np.random.uniform(0, 0.15), 2)
        trust = round(aum * np.random.uniform(0, 0.2), 2) if aum > 1_000_000 else 0

        # 交易行为
        monthly_tx_count = int(np.clip(np.random.lognormal(2, 1), 1, 50))
        monthly_tx_amount = round(aum * np.random.uniform(0.01, 0.15), 2)

        # 客户等级
        if aum >= 10_000_000:
            tier = "钻石客户"
        elif aum >= 5_000_000:
            tier = "白金客户"
        elif aum >= 1_000_000:
            tier = "金卡客户"
        elif aum >= 300_000:
            tier = "银卡客户"
        else:
            tier = "普通客户"

        # 产品持有（用于关联分析）
        owned_products = []
        owned_products.append("存款")
        if wealth_mgmt > 0:
            owned_products.append("理财产品")
        if fund > 0:
            owned_products.append("基金")
        if insurance > 0:
            owned_products.append("保险")
        if trust > 0:
            owned_products.append("信托")
        # 额外随机产品
        for p in ["贵金属", "外汇", "国债"]:
            if np.random.random() < 0.15 + 0.1 * (aum / 5_000_000):
                owned_products.append(p)

        product_count = len(owned_products)

        # 近12个月资产趋势（用于时间序列）
        trend_base = aum * np.random.uniform(0.6, 1.0)
        trend_growth = np.random.uniform(-0.02, 0.05)
        trend_volatility = np.random.uniform(0.01, 0.08)
        monthly_assets = []
        for m in range(12):
            val = trend_base * (1 + trend_growth) ** m + np.random.normal(0, trend_base * trend_volatility)
            monthly_assets.append(round(max(val, 10000), 2))

        # 高价值标签
        is_high_value = 1 if aum >= HIGH_VALUE_THRESHOLD else 0

        customers.append({
            "客户ID": cust_id,
            "年龄": age,
            "性别": gender,
            "城市": city,
            "职业": occupation,
            "资产总额(AUM)": aum,
            "存款": deposit,
            "理财": wealth_mgmt,
            "基金": fund,
            "保险": insurance,
            "信托": trust,
            "月均交易次数": monthly_tx_count,
            "月均交易金额": monthly_tx_amount,
            "客户等级": tier,
            "持有产品数": product_count,
            "持有产品": "|".join(owned_products),
            "近12月资产": ",".join(str(x) for x in monthly_assets),
            "是否高价值客户": is_high_value,
        })

    return pd.DataFrame(customers)


def main():
    df = generate_customers()

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")

    print(f"生成 {len(df)} 条客户数据 -> {DATA_PATH}")
    print(f"高价值客户: {df['是否高价值客户'].sum()} ({df['是否高价值客户'].mean()*100:.1f}%)")
    print(f"AUM 范围: {df['资产总额(AUM)'].min():,.0f} ~ {df['资产总额(AUM)'].max():,.0f}")
    print(f"\n客户等级分布:")
    print(df["客户等级"].value_counts().to_string())
    print(f"\n职业分布:")
    print(df["职业"].value_counts().to_string())


if __name__ == "__main__":
    main()
