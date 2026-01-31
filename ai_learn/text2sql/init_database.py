"""
初始化保险场景示例数据库
包含：客户表、保单表、理赔表、产品表
"""

import sqlite3
from datetime import datetime, timedelta
import random

def create_database():
    """创建数据库和表结构"""

    conn = sqlite3.connect('insurance.db')
    cursor = conn.cursor()

    # 创建产品表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            product_type TEXT,
            premium_range_min REAL,
            premium_range_max REAL,
            coverage_amount REAL
        )
    """)

    # 创建客户表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            phone TEXT,
            city TEXT,
            occupation TEXT
        )
    """)

    # 创建保单表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS policies (
            policy_id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id INTEGER,
            start_date DATE,
            end_date DATE,
            premium REAL,
            status TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )
    """)

    # 创建理赔表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            claim_id INTEGER PRIMARY KEY,
            policy_id INTEGER,
            claim_date DATE,
            claim_amount REAL,
            status TEXT,
            description TEXT,
            FOREIGN KEY (policy_id) REFERENCES policies(policy_id)
        )
    """)

    conn.commit()
    print("✓ 数据库表创建成功")

    return conn

def insert_sample_data(conn):
    """插入示例数据"""

    cursor = conn.cursor()

    # 插入产品数据
    products = [
        (1, '平安无忧终身寿险', '寿险', 5000, 20000, 500000),
        (2, '健康守护医疗保险', '医疗险', 2000, 8000, 100000),
        (3, '意外伤害保险', '意外险', 500, 2000, 200000),
        (4, '重大疾病保险', '重疾险', 3000, 15000, 300000),
        (5, '少儿教育金保险', '教育险', 3000, 10000, 200000),
    ]
    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)", products)

    # 插入客户数据
    customers = []
    names = ['张伟', '李娜', '王芳', '刘洋', '陈静', '杨明', '赵丽', '黄强', '周杰', '吴芳',
             '徐明', '孙娜', '马强', '朱丽', '胡杰', '郭芳', '林明', '何娜', '高强', '罗丽']

    for i, name in enumerate(names):
        customer = (
            i + 1,
            name,
            random.randint(25, 55),
            random.choice(['男', '女']),
            f'138{random.randint(10000000, 99999999)}',
            random.choice(['北京', '上海', '深圳', '广州', '杭州']),
            random.choice(['工程师', '教师', '医生', '销售', '财务', '自由职业'])
        )
        customers.append(customer)

    cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?)", customers)

    # 插入保单数据
    policies = []
    base_date = datetime.now()

    for customer_id in range(1, 21):
        # 每个客户1-3份保单
        num_policies = random.randint(1, 3)

        for _ in range(num_policies):
            product_id = random.randint(1, 5)
            product = [p for p in products if p[0] == product_id][0]

            start_date = base_date - timedelta(days=random.randint(30, 365))
            end_date = start_date + timedelta(days=365)

            premium = random.uniform(product[3], product[4])

            policy = (
                len(policies) + 1,
                customer_id,
                product_id,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                round(premium, 2),
                random.choices(['有效', '已失效', '已注销'], weights=[0.7, 0.2, 0.1])[0]
            )
            policies.append(policy)

    cursor.executemany("INSERT INTO policies VALUES (?, ?, ?, ?, ?, ?, ?)", policies)

    # 插入理赔数据
    claims = []
    active_policies = [p for p in policies if p[6] == '有效']

    for _ in range(30):
        policy = random.choice(active_policies)
        claim_date = datetime.strptime(policy[3], '%Y-%m-%d') + timedelta(days=random.randint(1, 180))

        claim = (
            len(claims) + 1,
            policy[0],
            claim_date.strftime('%Y-%m-%d'),
            random.uniform(1000, 50000),
            random.choices(['已批准', '审核中', '已拒绝'], weights=[0.6, 0.3, 0.1])[0],
            random.choice(['意外医疗', '疾病住院', '重大疾病', '意外伤害'])
        )
        claims.append(claim)

    cursor.executemany("INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?)", claims)

    conn.commit()
    print("✓ 示例数据插入成功")
    print(f"  - 产品: {len(products)} 条")
    print(f"  - 客户: {len(customers)} 条")
    print(f"  - 保单: {len(policies)} 条")
    print(f"  - 理赔: {len(claims)} 条")

def main():
    print("=" * 50)
    print("初始化保险场景数据库")
    print("=" * 50)

    conn = create_database()
    insert_sample_data(conn)

    print("\n✓ 数据库初始化完成！")
    print(f"  数据库文件: insurance.db")
    print("\n可以使用以下命令查看数据:")
    print("  sqlite3 insurance.db")
    print("  sqlite> .tables")
    print("  sqlite> SELECT * FROM policies LIMIT 5;")

    conn.close()

if __name__ == "__main__":
    main()
