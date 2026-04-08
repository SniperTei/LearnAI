"""数据总览大屏"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH

st.set_page_config(page_title="数据总览", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.title("📊 数据总览大屏")

# KPI 行
col1, col2, col3, col4 = st.columns(4)
col1.metric("客户总数", f"{len(df):,}")
col2.metric("总AUM", f"{df['资产总额(AUM)'].sum()/1e8:.2f}亿")
col3.metric("高价值客户占比", f"{df['是否高价值客户'].mean()*100:.1f}%")
col4.metric("平均年龄", f"{df['年龄'].mean():.0f}岁")

st.markdown("---")

# 资产分布
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("AUM 分布 (对数刻度)")
    fig = px.histogram(
        df, x="资产总额(AUM)", nbins=50,
        title="客户资产分布",
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_xaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("客户等级占比")
    tier_counts = df["客户等级"].value_counts()
    fig = px.pie(
        values=tier_counts.values, names=tier_counts.index,
        title="客户等级分布",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 城市和职业
col_city, col_occ = st.columns(2)

with col_city:
    st.subheader("城市 AUM 分布")
    city_aum = df.groupby("城市")["资产总额(AUM)"].mean().sort_values(ascending=True)
    fig = px.bar(
        x=city_aum.values / 1e4, y=city_aum.index, orientation="h",
        labels={"x": "平均AUM (万)", "y": "城市"},
        title="各城市平均AUM",
        color=city_aum.values,
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, use_container_width=True)

with col_occ:
    st.subheader("职业客户数分布")
    occ_counts = df["职业"].value_counts()
    fig = px.bar(
        x=occ_counts.index, y=occ_counts.values,
        labels={"x": "职业", "y": "客户数"},
        title="各职业客户数",
        color=occ_counts.values,
        color_continuous_scale="Greens",
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 数据预览
st.subheader("数据预览")
st.dataframe(df.head(20), use_container_width=True)
