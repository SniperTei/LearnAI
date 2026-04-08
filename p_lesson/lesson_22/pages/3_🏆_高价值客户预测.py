"""高价值客户预测页面"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH
from models.classification import train_models

st.set_page_config(page_title="高价值客户预测", page_icon="🏆", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def get_model_results(df):
    return train_models(df)

df = load_data()

st.title("🏆 高价值客户预测")

with st.spinner("训练模型中（逻辑回归 / 决策树 / GBDT / 随机森林）..."):
    results_df, roc_data, feature_importance, trained = get_model_results(df)

# 模型对比表格
st.subheader("模型性能对比")
st.dataframe(results_df.style.highlight_max(subset=["准确率", "精确率", "召回率", "F1分数"], axis=0), use_container_width=True)

st.markdown("---")

# ROC 曲线
col_roc, col_fi = st.columns(2)

with col_roc:
    st.subheader("ROC 曲线")
    fig = go.Figure()
    colors = {"逻辑回归": "blue", "决策树": "green", "GBDT": "red", "随机森林": "orange"}
    for name, data in roc_data.items():
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"],
            mode="lines", name=f"{name} (AUC={data['auc']:.3f})",
            line=dict(color=colors.get(name, "gray")),
        ))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="随机基线", line=dict(dash="dash", color="gray")))
    fig.update_layout(xaxis_title="FPR", yaxis_title="TPR", title="ROC 曲线对比")
    st.plotly_chart(fig, use_container_width=True)

with col_fi:
    st.subheader("特征重要性对比")
    selected_model = st.selectbox("选择模型", list(feature_importance.keys()), index=2)
    fi = feature_importance[selected_model]
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    fig = px.bar(
        x=list(fi_sorted.values()), y=list(fi_sorted.keys()), orientation="h",
        title=f"{selected_model} 特征重要性",
        color=list(fi_sorted.values()),
        color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 模型比较柱状图
st.subheader("各指标模型对比")
fig = go.Figure()
metrics = ["准确率", "精确率", "召回率", "F1分数"]
for _, row in results_df.iterrows():
    fig.add_trace(go.Bar(
        name=row["模型"],
        x=metrics,
        y=[row[m] for m in metrics],
    ))
fig.update_layout(barmode="group", title="各模型指标对比", yaxis=dict(tickformat=".2%"))
st.plotly_chart(fig, use_container_width=True)
