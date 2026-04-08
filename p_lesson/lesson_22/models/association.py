"""关联分析：产品推荐 (Apriori)"""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def prepare_transactions(df):
    """将产品持有转换为事务格式"""
    transactions = df["持有产品"].str.split("|").tolist()
    return transactions


def run_apriori(df, min_support=0.05, min_threshold=1.0):
    """运行 Apriori 关联分析"""
    transactions = prepare_transactions(df)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # 频繁项集
    frequent_items = apriori(df_encoded, min_support=min_support, use_colnames=True)
    frequent_items["项集长度"] = frequent_items["itemsets"].apply(len)

    # 关联规则
    rules = association_rules(
        frequent_items, metric="confidence", min_threshold=min_threshold
    )
    rules = rules.sort_values("lift", ascending=False)

    # 生成推荐建议
    recommendations = _generate_recommendations(rules)

    return frequent_items, rules, recommendations


def _generate_recommendations(rules):
    """基于关联规则生成推荐建议"""
    recs = []
    seen = set()
    for _, row in rules.head(20).iterrows():
        antecedent = ", ".join(row["antecedents"])
        consequent = ", ".join(row["consequents"])
        key = f"{antecedent}->{consequent}"
        if key not in seen:
            seen.add(key)
            recs.append({
                "前提产品": antecedent,
                "推荐产品": consequent,
                "支持度": round(row["support"], 4),
                "置信度": round(row["confidence"], 4),
                "提升度": round(row["lift"], 4),
            })
    return pd.DataFrame(recs)


def get_product_co_occurrence(df):
    """产品共现矩阵（用于网络图）"""
    from config import PRODUCTS
    products = [p for p in PRODUCTS]
    co_matrix = pd.DataFrame(0, index=products, columns=products)

    for _, row in df.iterrows():
        owned = row["持有产品"].split("|")
        for i, p1 in enumerate(owned):
            for p2 in owned[i + 1:]:
                if p1 in co_matrix.index and p2 in co_matrix.columns:
                    co_matrix.loc[p1, p2] += 1
                    co_matrix.loc[p2, p1] += 1

    return co_matrix
