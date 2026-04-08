"""客户分群页面"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH
from models.clustering import find_optimal_k, run_kmeans, run_dbscan, get_radar_data, prepare_features

st.set_page_config(page_title="客户分群", page_icon="👥", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.title("👥 客户分群分析")

# 侧边栏参数
st.sidebar.header("聚类参数")
method = st.sidebar.radio("聚类方法", ["K-Means", "DBSCAN"])

X_scaled, feature_cols, _ = prepare_features(df)

if method == "K-Means":
    # 手肘法
    with st.spinner("计算最优K值..."):
        k_range, inertias, silhouettes = find_optimal_k(X_scaled)

    col_elbow, col_sil = st.columns(2)
    with col_elbow:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(k_range), y=inertias, mode="lines+markers"))
        fig.update_layout(title="手肘法 - 惯性随K变化", xaxis_title="K", yaxis_title="惯性")
        st.plotly_chart(fig, use_container_width=True)

    with col_sil:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(k_range), y=silhouettes, mode="lines+markers"))
        fig.update_layout(title="轮廓系数随K变化", xaxis_title="K", yaxis_title="轮廓系数")
        st.plotly_chart(fig, use_container_width=True)

    n_clusters = st.sidebar.slider("聚类数量 K", 2, 8, 4)

    df_result, profiles, profile_names, X_pca, labels = run_kmeans(df, n_clusters=n_clusters)

else:
    eps = st.sidebar.slider("DBSCAN eps", 0.5, 3.0, 1.5, 0.1)
    min_samples = st.sidebar.slider("最小样本数", 3, 30, 10)

    labels, X_pca, n_clusters, n_noise = run_dbscan(df, eps=eps, min_samples=min_samples)
    df_result = df.copy()
    df_result["聚类标签"] = labels

    st.metric("发现聚类数", n_clusters)
    st.metric("噪声点数", n_noise)

st.markdown("---")

# 散点图
st.subheader("聚类散点图 (PCA降维)")
scatter_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "聚类": labels.astype(str)})
fig = px.scatter(
    scatter_df, x="PC1", y="PC2", color="聚类",
    title="客户聚类结果", opacity=0.6,
    color_discrete_sequence=px.colors.qualitative.Set1,
)
st.plotly_chart(fig, use_container_width=True)

if method == "K-Means":
    # 雷达图
    st.subheader("各客群画像 (雷达图)")
    radar = get_radar_data(df_result, feature_cols)

    fig = go.Figure()
    for cluster_id in radar.index:
        fig.add_trace(go.Scatterpolar(
            r=radar.loc[cluster_id].values,
            theta=feature_cols,
            fill="toself",
            name=profile_names.get(cluster_id, f"群{cluster_id}"),
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # 客群统计
    st.subheader("客群画像描述")
    for cluster_id, name in profile_names.items():
        cluster_data = df_result[df_result["聚类标签"] == cluster_id]
        st.markdown(f"**{name}** ({len(cluster_data)}人)")
        col1, col2, col3 = st.columns(3)
        col1.metric("平均AUM", f"{cluster_data['资产总额(AUM)'].mean()/1e4:.1f}万")
        col2.metric("平均年龄", f"{cluster_data['年龄'].mean():.0f}岁")
        col3.metric("平均产品数", f"{cluster_data['持有产品数'].mean():.1f}")
        st.markdown("")

st.markdown("---")
st.subheader("聚类结果数据")
st.dataframe(df_result[["客户ID", "客户等级", "资产总额(AUM)", "持有产品数", "聚类标签"] + (["客群名称"] if method == "K-Means" else [])].head(30), use_container_width=True)
