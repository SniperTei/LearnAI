"""资产变动预测页面"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH
from models.timeseries import forecast_all_customers, get_aggregate_trend

st.set_page_config(page_title="资产变动预测", page_icon="📈", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.title("📈 资产变动预测")

# 汇总趋势
st.subheader("全体客户平均资产趋势")
months, avg_monthly = get_aggregate_trend(df)
avg_wan = [v / 1e4 for v in avg_monthly]

fig = go.Figure()
fig.add_trace(go.Scatter(x=months, y=avg_wan, mode="lines+markers", name="平均AUM (万)"))
fig.update_layout(title="近12个月全体客户平均AUM趋势", xaxis_title="月份", yaxis_title="AUM (万)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 客户级别预测
st.subheader("客户资产预测（抽样）")
with st.spinner("进行资产预测..."):
    forecasts = forecast_all_customers(df, sample_size=50, forecast_periods=3)

# 趋势统计
trend_counts = {"上升": 0, "下降": 0}
for f in forecasts:
    trend_counts[f["趋势"]] += 1

col1, col2, col3 = st.columns(3)
col1.metric("预测上升客户", trend_counts["上升"])
col2.metric("预测下降客户", trend_counts["下降"])
col3.metric("平均增长率", f"{np.mean([f['增长率'] for f in forecasts]):.1f}%")

st.markdown("---")

# 选择客户查看详情
customer_ids = [f["客户ID"] for f in forecasts]
selected_idx = st.selectbox("选择客户查看详细预测", range(len(customer_ids)),
                            format_func=lambda i: f"{forecasts[i]['客户ID']} ({forecasts[i]['客户等级']}, {forecasts[i]['当前AUM']/1e4:.0f}万)")

f = forecasts[selected_idx]
history_wan = [v / 1e4 for v in f["历史数据"]]
ma_wan = [v / 1e4 for v in f["MA预测"]]
arima_wan = [v / 1e4 for v in f["ARIMA预测"]]

all_months = [f"M{i+1}" for i in range(12)] + ["M13", "M14", "M15"]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=all_months[:12], y=history_wan, mode="lines+markers", name="历史数据"
))
fig.add_trace(go.Scatter(
    x=all_months[11:], y=[history_wan[-1]] + ma_wan, mode="lines+markers",
    name="移动平均预测", line=dict(dash="dash"),
))
fig.add_trace(go.Scatter(
    x=all_months[11:], y=[history_wan[-1]] + arima_wan, mode="lines+markers",
    name="ARIMA预测", line=dict(dash="dot"),
))

# 置信区间
if f["置信区间"] is not None:
    ci = np.array(f["置信区间"])
    lower = [v / 1e4 for v in ci[:, 0]]
    upper = [v / 1e4 for v in ci[:, 1]]
    fig.add_trace(go.Scatter(
        x=all_months[12:], y=upper, mode="lines",
        line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=all_months[12:], y=lower, mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor="rgba(255,0,0,0.1)",
        name="95% 置信区间",
    ))

fig.update_layout(
    title=f"客户 {f['客户ID']} 资产预测",
    xaxis_title="月份", yaxis_title="AUM (万)",
)
st.plotly_chart(fig, use_container_width=True)

# 预测详情表格
st.subheader("预测结果汇总")
summary = []
for f in forecasts:
    summary.append({
        "客户ID": f["客户ID"],
        "等级": f["客户等级"],
        "当前AUM(万)": round(f["当前AUM"] / 1e4, 1),
        "趋势": f["趋势"],
        "增长率": f"{f['增长率']}%",
        "ARIMA预测M13(万)": round(f["ARIMA预测"][0] / 1e4, 1),
    })
st.dataframe(pd.DataFrame(summary), use_container_width=True)
