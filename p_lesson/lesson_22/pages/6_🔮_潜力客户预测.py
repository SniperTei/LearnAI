"""潜力客户预测页面：预测非高价值客户中谁将在未来3个月升级为高价值客户"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH, HIGH_VALUE_THRESHOLD
from models.potential_customer import generate_future_labels, train_potential_model, predict_potential_customers

st.set_page_config(page_title="潜力客户预测", page_icon="🔮", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.title("🔮 潜力客户预测")
st.markdown("通过客户资产趋势 + 逻辑回归模型，预测非高价值客户中谁将在 **未来3个月** 升级为高价值客户（AUM ≥ 500万）")

st.markdown("---")

# Step 1: 生成标签
with st.spinner("分析客户资产趋势，生成潜力标签..."):
    df_labeled = generate_future_labels(df)

non_hv = df_labeled[df_labeled["是否当前高价值"] == 0]
potential_count = non_hv["是否潜力客户"].sum()
total_non_hv = len(non_hv)

col1, col2, col3 = st.columns(3)
col1.metric("非高价值客户数", f"{total_non_hv}")
col2.metric("预测将升级客户数", f"{potential_count}")
col3.metric("升级比例", f"{potential_count / total_non_hv * 100:.1f}%")

if potential_count == 0:
    st.warning("当前数据中没有预测会升级的客户。这可能是因为合成数据的趋势模拟导致升级客户较少。")
    st.stop()

st.markdown("---")

# Step 2: 训练模型
with st.spinner("训练逻辑回归模型..."):
    model, scaler, le_gender, le_occ, feature_cols, report, fpr, tpr, roc_auc, fi = train_potential_model(df_labeled)

# 模型性能
st.subheader("模型性能")
col_perf1, col_perf2 = st.columns(2)

with col_perf1:
    target_report = report.get("1", report.get("weighted avg", {}))
    st.metric("准确率", f"{report['accuracy']:.1%}")
    st.metric("AUC", f"{roc_auc:.3f}")

with col_perf2:
    if "1" in report:
        st.metric("精确率", f"{report['1']['precision']:.1%}")
        st.metric("召回率", f"{report['1']['recall']:.1%}")
        st.metric("F1分数", f"{report['1']['f1-score']:.1%}")

# ROC 曲线
fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"逻辑回归 (AUC={roc_auc:.3f})"))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="随机基线", line=dict(dash="dash", color="gray")))
fig.update_layout(title="ROC 曲线", xaxis_title="FPR", yaxis_title="TPR")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Step 3: 特征重要性
st.subheader("特征重要性（影响升级的关键因素）")
fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
fig = px.bar(
    x=list(fi_sorted.values()), y=list(fi_sorted.keys()), orientation="h",
    color=list(fi_sorted.values()), color_continuous_scale="Viridis",
)
fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Step 4: 预测结果
st.subheader("潜力客户排行榜 TOP 30")
predictions = predict_potential_customers(df_labeled, model, scaler, le_gender, le_occ, feature_cols)

# 高亮预测为潜力的客户
top30 = predictions.head(30).copy()
top30["AUM(万)"] = top30["AUM"].apply(lambda x: f"{x/1e4:.1f}")
top30["预测AUM(万)"] = top30["预测AUM"].apply(lambda x: f"{x/1e4:.1f}")
top30["潜力概率"] = top30["潜力概率"].apply(lambda x: f"{x:.1%}")
top30["增长率"] = top30["增长率"].apply(lambda x: f"{x:.1%}")
top30["近3月增长率"] = top30["近3月增长率"].apply(lambda x: f"{x:.1%}")

st.dataframe(
    top30[["客户ID", "客户等级", "AUM(万)", "预测AUM(万)", "增长率", "近3月增长率", "持有产品数", "潜力概率"]],
    use_container_width=True,
)

st.markdown("---")

# 潜力客户分布
st.subheader("潜力客户分析")
pred_potential = predictions[predictions["预测结果"] == 1]

if len(pred_potential) > 0:
    col_dist1, col_dist2 = st.columns(2)

    with col_dist1:
        tier_counts = pred_potential["客户等级"].value_counts()
        fig = px.pie(values=tier_counts.values, names=tier_counts.index,
                     title="潜力客户等级分布", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col_dist2:
        fig = px.histogram(
            predictions, x="潜力概率", nbins=30,
            title="非高价值客户潜力概率分布",
            color_discrete_sequence=["#636EFA"],
        )
        st.plotly_chart(fig, use_container_width=True)
