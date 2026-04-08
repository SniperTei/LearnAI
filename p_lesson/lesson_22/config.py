"""全局配置"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "customers.csv")

# 数据生成参数
NUM_CUSTOMERS = 2000
RANDOM_SEED = 42

# 产品列表
PRODUCTS = ["存款", "理财产品", "基金", "保险", "信托", "贵金属", "外汇", "国债"]

# 客户等级
CUSTOMER_TIERS = ["普通客户", "银卡客户", "金卡客户", "白金客户", "钻石客户"]

# 高价值客户阈值（AUM > 500万）
HIGH_VALUE_THRESHOLD = 5_000_000
