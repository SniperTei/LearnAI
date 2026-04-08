"""产品推荐页面"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH
from models.association import run_apriori, get_product_co_occurrence

st.set_page_config(page_title="产品推荐", page_icon="💎", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.title("💎 产品关联分析与推荐")

# 侧边栏参数
st.sidebar.header("Apriori 参数")
min_support = st.sidebar.slider("最小支持度", 0.01, 0.2, 0.05, 0.01)
min_confidence = st.sidebar.slider("最小置信度", 0.1, 1.0, 0.5, 0.1)

with st.spinner("运行 Apriori 关联分析..."):
    frequent_items, rules, recommendations = run_apriori(
        df, min_support=min_support, min_threshold=min_confidence
    )

# 频繁项集
st.subheader("频繁项集 TOP 15")
fi_display = frequent_items.sort_values("support", ascending=False).head(15).copy()
fi_display["itemsets"] = fi_display["itemsets"].apply(lambda x: ", ".join(x))
fi_display = fi_display.rename(columns={"itemsets": "项集", "support": "支持度"})
st.dataframe(fi_display, use_container_width=True)

st.markdown("---")

# 关联规则网络图 (用共现矩阵热力图代替)
st.subheader("产品共现热力图")
co_matrix = get_product_co_occurrence(df)
fig = px.imshow(
    co_matrix.values,
    x=co_matrix.columns, y=co_matrix.index,
    text_auto=True, aspect="auto",
    title="产品共现次数",
    color_continuous_scale="YlOrRd",
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 关联规则散点图
if len(rules) > 0:
    st.subheader("关联规则分布")
    rules_display = rules.copy()
    rules_display["antecedents"] = rules_display["antecedents"].apply(lambda x: ", ".join(x))
    rules_display["consequents"] = rules_display["consequents"].apply(lambda x: ", ".join(x))

    fig = px.scatter(
        rules_display, x="support", y="confidence",
        size="lift", color="lift",
        hover_data=["antecedents", "consequents"],
        title="关联规则 (气泡大小=提升度)",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

# 推荐列表
st.subheader("产品推荐建议")
if len(recommendations) > 0:
    st.dataframe(recommendations, use_container_width=True)
else:
    st.info("当前参数下未发现关联规则，请尝试降低最小支持度或置信度。")

st.markdown("---")

# 产品持有率
st.subheader("各产品持有率")
from config import PRODUCTS
product_rates = {}
for p in PRODUCTS:
    product_rates[p] = df["持有产品"].str.contains(p).mean() * 100

rate_df = pd.DataFrame({"产品": list(product_rates.keys()), "持有率(%)": list(product_rates.values())})
fig = px.bar(rate_df, x="产品", y="持有率(%)", title="各产品客户持有率", color="持有率(%)")
st.plotly_chart(fig, use_container_width=True)
