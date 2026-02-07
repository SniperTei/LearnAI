# PostgreSQL 操作指南

## 目录
1. [安装与启动](#安装与启动)
2. [连接方式](#连接方式)
3. [基础命令](#基础命令)
4. [数据库操作](#数据库操作)
5. [表操作](#表操作)
6. [数据操作](#数据操作)
7. [高级功能](#高级功能)
8. [常见问题](#常见问题)

---

## 安装与启动

### macOS 安装
```bash
# 通过 Homebrew 安装
brew install postgresql@16

# 启动服务
brew services start postgresql@16

# 停止服务
brew services stop postgresql@16

# 重启服务
brew services restart postgresql@16

# 查看服务状态
brew services list | grep postgresql
```

### 验证安装
```bash
# 检查版本
psql --version

# 或使用完整路径
/opt/homebrew/opt/postgresql@16/bin/psql --version
```

---

## 连接方式

### 1. 命令行连接 (psql)
```bash
# 连接到默认数据库
psql postgres

# 连接到指定数据库
psql -d insurance

# 指定用户名连接
psql -U username -d database_name

# 指定主机和端口
psql -h localhost -p 5432 -d postgres

# 使用完整路径（如果 PATH 未配置）
/opt/homebrew/opt/postgresql@16/bin/psql postgres
```

### 2. GUI 客户端

#### DBeaver (推荐 - 免费开源)
```bash
# 安装
brew install --cask dbeaver-community

# 连接配置:
# Host: localhost
# Port: 5432
# Database: postgres
# Username: (你的 macOS 用户名)
# Password: (默认为空，直接回车)
```

#### TablePlus (付费 - UI 精美)
```bash
# 安装
brew install --cask tableplus

# 免费版限制: 2 个连接
```

#### pgAdmin 4 (官方工具)
```bash
# 安装
brew install --cask pgadmin4
```

---

## 基础命令

### psql 元命令
```sql
-- 帮助
\?              -- 显示所有 psql 命令
\h              -- 显示 SQL 命令帮助
\h CREATE TABLE -- 显示特定命令的帮助

-- 退出
\q              -- 退出 psql

-- 清屏
\! clear        -- macOS/Linux

-- 执行 shell 命令
\! ls           -- 列出当前目录文件
```

### 信息查询
```sql
-- 列出所有数据库
\l

-- 列出当前数据库的所有表
\dt

-- 列出所有表（包括系统表）
\dt+

-- 描述表结构
\d table_name
\d customers

-- 显示所有索引
\di

-- 显示所有用户
\du

-- 显示当前数据库
\conninfo

-- 显示当前用户
\conninfo
SELECT current_user;

-- 显示表大小
\dt+

-- 显示函数
\df
```

---

## 数据库操作

### 创建数据库
```sql
-- 创建数据库
CREATE DATABASE insurance;

-- 创建数据库并指定编码
CREATE DATABASE insurance
  ENCODING 'UTF8'
  LC_COLLATE = 'en_US.UTF-8'
  LC_CTYPE = 'en_US.UTF-8';

-- 查看数据库详情
SELECT * FROM pg_database WHERE datname = 'insurance';
```

### 切换数据库
```sql
-- 在 psql 中切换数据库
\c insurance
\c insurance username

-- 或者退出后重新连接
\q
psql insurance
```

### 删除数据库
```sql
-- 删除数据库（不能在有连接时删除）
DROP DATABASE insurance;

-- 强制断开所有连接后删除
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'insurance';

DROP DATABASE insurance;
```

### 修改数据库
```sql
-- 重命名数据库
ALTER DATABASE insurance RENAME TO insurance_db;

-- 修改数据库所有者
ALTER DATABASE insurance OWNER TO new_user;

```

---

## 表操作

### 创建表
```sql
-- 创建客户表
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INTEGER,
    gender VARCHAR(10),
    city VARCHAR(100),
    email VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建产品表
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(200) NOT NULL,
    product_type VARCHAR(50),
    min_premium DECIMAL(10, 2),
    max_premium DECIMAL(10, 2),
    description TEXT
);

-- 创建保单表
CREATE TABLE policies (
    policy_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    product_id INTEGER REFERENCES products(product_id),
    start_date DATE,
    end_date DATE,
    premium DECIMAL(10, 2),
    status VARCHAR(20) CHECK (status IN ('active', 'expired', 'cancelled'))
);

-- 创建理赔表
CREATE TABLE claims (
    claim_id SERIAL PRIMARY KEY,
    policy_id INTEGER REFERENCES policies(policy_id),
    claim_date DATE,
    amount DECIMAL(10, 2),
    status VARCHAR(20) DEFAULT 'pending',
    description TEXT
);
```

### 查看表结构
```sql
-- 查看表定义
\d table_name
\d customers

-- 查看表的创建语句
SELECT
    'CREATE TABLE ' || table_name || ' (' ||
    string_agg(
        column_name || ' ' ||
        data_type ||
        CASE WHEN character_maximum_length IS NOT NULL
             THEN '(' || character_maximum_length || ')'
             ELSE '' END ||
        CASE WHEN is_nullable = 'NO' THEN ' NOT NULL' ELSE '' END ||
        CASE WHEN column_default IS NOT NULL
             THEN ' DEFAULT ' || column_default
             ELSE '' END,
        ', '
    ) || ');'
FROM information_schema.columns
WHERE table_name = 'customers'
GROUP BY table_name;
```

### 修改表
```sql
-- 添加列
ALTER TABLE customers ADD COLUMN phone VARCHAR(20);

-- 删除列
ALTER TABLE customers DROP COLUMN phone;

-- 重命名列
ALTER TABLE customers RENAME COLUMN name TO full_name;

-- 修改列类型
ALTER TABLE customers ALTER COLUMN age TYPE SMALLINT;

-- 添加约束
ALTER TABLE customers ADD CONSTRAINT email_check
  CHECK (email ~* '^[A-Za-z0-9._%-]+@[A-Za-z0-9.-]+[.][A-Za-z]+$');

-- 添加外键
ALTER TABLE policies
  ADD CONSTRAINT fk_customer
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id);

-- 重命名表
ALTER TABLE customers RENAME TO clients;
```

### 删除表
```sql
-- 删除表
DROP TABLE customers;

-- 删除表及其依赖关系
DROP TABLE customers CASCADE;

-- 清空表（保留表结构）
TRUNCATE TABLE customers;
TRUNCATE TABLE customers RESTART IDENTITY CASCADE; -- 重置自增序列
```

---

## 数据操作

### 插入数据
```sql
-- 插入单条记录
INSERT INTO customers (name, age, gender, city, email)
VALUES ('张三', 30, '男', '北京', 'zhangsan@example.com');

-- 插入多条记录
INSERT INTO customers (name, age, gender, city) VALUES
    ('李四', 25, '女', '上海'),
    ('王五', 35, '男', '广州'),
    ('赵六', 28, '女', '深圳');

-- 从其他表插入数据
INSERT INTO customers_archive
SELECT * FROM customers WHERE age > 30;

-- 插入并返回数据
INSERT INTO customers (name, age) VALUES ('测试', 20)
RETURNING *;
```

### 查询数据
```sql
-- 基础查询
SELECT * FROM customers;
SELECT name, email FROM customers;

-- 条件查询
SELECT * FROM customers WHERE age > 30;
SELECT * FROM customers WHERE city = '北京' AND gender = '女';

-- 排序
SELECT * FROM customers ORDER BY age DESC;
SELECT * FROM customers ORDER BY city, age DESC;

-- 限制结果数量
SELECT * FROM customers LIMIT 10;
SELECT * FROM customers LIMIT 10 OFFSET 20;

-- 去重
SELECT DISTINCT city FROM customers;

-- 聚合查询
SELECT COUNT(*) FROM customers;
SELECT AVG(age) FROM customers;
SELECT MAX(age), MIN(age) FROM customers;
SELECT city, COUNT(*) as count
FROM customers
GROUP BY city
HAVING COUNT(*) > 5;

-- 连接查询
SELECT c.name, p.product_name
FROM customers c
JOIN policies po ON c.customer_id = po.customer_id
JOIN products p ON po.product_id = p.product_id;
```

### 更新数据
```sql
-- 更新单条记录
UPDATE customers SET age = 31 WHERE customer_id = 1;

-- 更新多条记录
UPDATE customers SET city = '北京' WHERE city = 'Beijing';

-- 更新并返回
UPDATE customers SET age = age + 1 WHERE customer_id = 1
RETURNING *;
```

### 删除数据
```sql
-- 删除单条记录
DELETE FROM customers WHERE customer_id = 1;

-- 删除多条记录
DELETE FROM customers WHERE age < 18;

-- 删除并返回
DELETE FROM customers WHERE city = '未知'
RETURNING *;
```

---

## 高级功能

### 索引
```sql
-- 创建索引
CREATE INDEX idx_customer_email ON customers(email);
CREATE INDEX idx_customer_age ON customers(age);
CREATE INDEX idx_customer_city_age ON customers(city, age);

-- 创建唯一索引
CREATE UNIQUE INDEX idx_unique_email ON customers(email);

-- 查看索引
\di
SELECT * FROM pg_indexes WHERE tablename = 'customers';

-- 删除索引
DROP INDEX idx_customer_email;
```

### 视图
```sql
-- 创建视图
CREATE VIEW customer_summary AS
SELECT
    c.city,
    COUNT(*) as customer_count,
    AVG(c.age) as avg_age
FROM customers c
GROUP BY c.city;

-- 查询视图
SELECT * FROM customer_summary;

-- 删除视图
DROP VIEW customer_summary;
```

### 存储过程
```sql
-- 创建函数
CREATE OR REPLACE FUNCTION get_customer_count(city_name VARCHAR)
RETURNS INTEGER AS $$
BEGIN
    RETURN (
        SELECT COUNT(*) FROM customers
        WHERE city = city_name
    );
END;
$$ LANGUAGE plpgsql;

-- 调用函数
SELECT get_customer_count('北京');

-- 删除函数
DROP FUNCTION get_customer_count(VARCHAR);
```

### 触发器
```sql
-- 创建触发器函数
CREATE OR REPLACE FUNCTION update_modified_time()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 创建触发器
CREATE TRIGGER update_customer_timestamp
BEFORE UPDATE ON customers
FOR EACH ROW
EXECUTE FUNCTION update_modified_time();
```

### 事务
```sql
-- 开始事务
BEGIN;

-- 执行多个操作
INSERT INTO customers (name, age) VALUES ('测试', 20);
UPDATE customers SET age = 21 WHERE name = '测试';

-- 提交事务
COMMIT;

-- 或回滚事务
ROLLBACK;
```

---

## 从 SQLite 迁移到 PostgreSQL

### 方法 1: 使用 pgloader (推荐)
```bash
# 安装 pgloader
brew install pgloader

# 迁移整个数据库
pgloader sqlite:///path/to/insurance.db postgresql://localhost/insurance

# 查看迁移日志
pgloader --verbose sqlite:///path/to/insurance.db postgresql://localhost/insurance
```

### 方法 2: 手动导出导入
```bash
# 从 SQLite 导出数据
sqlite3 insurance.db .dump > dump.sql

# 修改 dump.sql 中的 SQLite 特定语法
# - INTEGER PRIMARY KEY → SERIAL PRIMARY KEY
# - AUTOINCREMENT → SERIAL
# - datetime('now') → CURRENT_TIMESTAMP
# - 删除 SQLite 特有命令

# 导入到 PostgreSQL
psql insurance < dump.sql
```

### 方法 3: 使用 Python 脚本
```python
import sqlite3
import psycopg2
from psycopg2.extras import execute_batch

# 连接 SQLite
sqlite_conn = sqlite3.connect('insurance.db')
sqlite_cursor = sqlite_conn.cursor()

# 连接 PostgreSQL
pg_conn = psycopg2.connect(
    host="localhost",
    database="insurance",
    user="your_username"
)
pg_cursor = pg_conn.cursor()

# 迁移数据示例
def migrate_table(table_name):
    sqlite_cursor.execute(f"SELECT * FROM {table_name}")
    rows = sqlite_cursor.fetchall()

    columns = [desc[0] for desc in sqlite_cursor.description]
    placeholders = ', '.join(['%s'] * len(columns))
    query = f"INSERT INTO {table_name} VALUES ({placeholders})"

    execute_batch(pg_cursor, query, rows)
    pg_conn.commit()

# 迁移所有表
tables = ['customers', 'products', 'policies', 'claims']
for table in tables:
    migrate_table(table)
    print(f"Migrated {table}")
```

---

## 配置与优化

### 连接配置文件
```bash
# ~/.pgpass 文件（自动密码登录）
localhost:5432:insurance:username:password
chmod 600 ~/.pgpass

# ~/.psqlrc 文件（psql 个人配置）
\set AUTOCOMMIT off
\timing on  -- 显示查询时间
\pset null 'NULL'  -- 显示 NULL 值
```

### 性能优化
```sql
-- VACUUM - 回收空间
VACUUM;
VACUUM ANALYZE customers;

-- ANALYZE - 更新统计信息
ANALYZE;

-- REINDEX - 重建索引
REINDEX DATABASE insurance;
REINDEX TABLE customers;

-- 查看慢查询
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- 查看表大小
SELECT
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

---

## 备份与恢复

### 备份
```bash
# 备份单个数据库
pg_dump insurance > insurance_backup.sql

# 压缩备份
pg_dump insurance | gzip > insurance_backup.sql.gz

# 备份所有数据库
pg_dumpall > all_databases.sql

# 自定义格式备份（更快，更灵活）
pg_dump -Fc insurance -f insurance_backup.dump
```

### 恢复
```bash
# 恢复 SQL 备份
psql insurance < insurance_backup.sql

# 恢复压缩备份
gunzip -c insurance_backup.sql.gz | psql insurance

# 恢复自定义格式备份
pg_restore -d insurance insurance_backup.dump

# 从备份恢复特定表
pg_restore -t customers -d insurance insurance_backup.dump
```

---

## 常见问题

### 连接问题
```bash
# 无法连接到服务器
psql: could not connect to server: Connection refused

# 解决方法
brew services start postgresql@16

# 检查端口是否被占用
lsof -i :5432

# 检查 PostgreSQL 状态
brew services list | grep postgresql
```

### 权限问题
```sql
-- 授予权限
GRANT ALL PRIVILEGES ON DATABASE insurance TO user_name;
GRANT ALL PRIVILEGES ON TABLE customers TO user_name;
GRANT SELECT, INSERT ON TABLE customers TO user_name;

-- 创建用户
CREATE USER new_user WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE insurance TO new_user;
```

### 密码认证
```bash
# 修改 pg_hba.conf 文件
# 位置: /opt/homebrew/var/postgresql@16/pg_hba.conf

# 将此行:
#   local   all             all                                     md5
# 改为:
#   local   all             all                                     trust

# 然后重启 PostgreSQL
brew services restart postgresql@16
```

### 性能问题
```sql
-- 查看当前连接
SELECT * FROM pg_stat_activity;

-- 查看表大小
SELECT
    relname AS table_name,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;

-- 查看索引使用情况
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan AS index_scans,
    idx_tup_read AS tuples_read,
    idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC;
```

---

## Python 集成

### 使用 psycopg2
```python
import psycopg2
from psycopg2.extras import RealDictCursor

# 连接数据库
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="insurance",
    user="your_username"
)

# 使用字典游标
cursor = conn.cursor(cursor_factory=RealDictCursor)

# 执行查询
cursor.execute("SELECT * FROM customers WHERE age > %s", (30,))
results = cursor.fetchall()

# 插入数据
cursor.execute(
    "INSERT INTO customers (name, age) VALUES (%s, %s)",
    ("张三", 30)
)
conn.commit()

# 关闭连接
cursor.close()
conn.close()
```

### 使用 SQLAlchemy
```python
from sqlalchemy import create_engine, text

# 创建引擎
engine = create_engine('postgresql://user@localhost/insurance')

# 执行查询
with engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM customers"))
    for row in result:
        print(row)

# 使用 ORM
from sqlalchemy.orm import Session, sessionmaker

Session = sessionmaker(bind=engine)
session = Session()

customers = session.query(Customer).filter(Customer.age > 30).all()
```

---

## 资源链接

- **官方文档**: https://www.postgresql.org/docs/
- **pgvector (AI 向量扩展)**: https://github.com/pgvector/pgvector
- **psql 命令速查**: https://www.postgresql.org/docs/current/app-psql.html
- **SQL 教程**: https://www.postgresql.org/docs/current/sql-commands.html
