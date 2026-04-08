"""银行高净值客户 AI 运营助手 - Streamlit 主入口"""
import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PATH

st.set_page_config(
    page_title="银行高净值客户 AI 运营助手",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏦 银行高净值客户 AI 运营助手")
st.markdown("---")

# 加载数据
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.warning("数据文件不存在，请先运行 `python data/generate_data.py`")
        st.stop()
    return pd.read_csv(DATA_PATH)

df = load_data()

# 首页 KPI 卡片
st.header("核心指标概览")
col1, col2, col3, col4, col5 = st.columns(5)

total_customers = len(df)
total_aum = df["资产总额(AUM)"].sum()
high_value_count = df["是否高价值客户"].sum()
avg_aum = df["资产总额(AUM)"].mean()
avg_products = df["持有产品数"].mean()

col1.metric("客户总数", f"{total_customers:,}")
col2.metric("总AUM", f"{total_aum/1e8:.2f}亿")
col3.metric("高价值客户", f"{high_value_count}", f"{high_value_count/total_customers*100:.1f}%")
col4.metric("户均AUM", f"{avg_aum/1e4:.1f}万")
col5.metric("户均产品数", f"{avg_products:.1f}")

st.markdown("---")

# 两栏布局
left, right = st.columns(2)

with left:
    st.subheader("客户等级分布")
    tier_counts = df["客户等级"].value_counts()
    st.bar_chart(tier_counts)

with right:
    st.subheader("职业分布")
    occ_counts = df["职业"].value_counts()
    st.bar_chart(occ_counts)

st.markdown("---")
st.info("👈 请使用左侧导航栏访问各分析模块：客户分群、高价值预测、资产预测、产品推荐")
