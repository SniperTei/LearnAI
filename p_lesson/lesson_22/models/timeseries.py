"""时间序列：资产变动预测"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def prepare_ts_data(df):
    """准备时间序列数据"""
    ts_data = []
    for _, row in df.iterrows():
        monthly = [float(x) for x in row["近12月资产"].split(",")]
        ts_data.append({
            "客户ID": row["客户ID"],
            "客户等级": row["客户等级"],
            "AUM": row["资产总额(AUM)"],
            "月度资产": monthly,
        })
    return ts_data


def moving_average_forecast(data, window=3, forecast_periods=3):
    """移动平均预测"""
    history = data.copy()
    forecasts = []
    for _ in range(forecast_periods):
        ma = np.mean(history[-window:])
        forecasts.append(ma)
        history.append(ma)
    return forecasts


def arima_forecast(data, order=(1, 1, 1), forecast_periods=3):
    """ARIMA 预测"""
    try:
        model = ARIMA(data, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=forecast_periods)
        # 获取置信区间
        conf_int = fitted.get_forecast(steps=forecast_periods).conf_int()
        return forecast.tolist(), conf_int
    except Exception:
        # ARIMA 失败时回退到移动平均
        forecasts = moving_average_forecast(data, forecast_periods=forecast_periods)
        return forecasts, None


def forecast_all_customers(df, sample_size=50, forecast_periods=3):
    """对抽样客户进行资产预测"""
    ts_data = prepare_ts_data(df)

    # 按客户等级分层抽样
    sample_df = df.groupby("客户等级", group_keys=False).apply(
        lambda x: x.sample(min(len(x), sample_size // 5), random_state=42)
    )
    sample_ids = sample_df["客户ID"].tolist()

    results = []
    for record in ts_data:
        if record["客户ID"] not in sample_ids:
            continue

        monthly = record["月度资产"]

        # 移动平均
        ma_forecast = moving_average_forecast(monthly, forecast_periods=forecast_periods)

        # ARIMA
        arima_result, conf_int = arima_forecast(monthly, forecast_periods=forecast_periods)

        # 计算趋势
        trend = "上升" if monthly[-1] > monthly[0] else "下降"
        growth_rate = (monthly[-1] - monthly[0]) / monthly[0] * 100

        results.append({
            "客户ID": record["客户ID"],
            "客户等级": record["客户等级"],
            "当前AUM": record["AUM"],
            "历史数据": monthly,
            "MA预测": ma_forecast,
            "ARIMA预测": arima_result if isinstance(arima_result, list) else arima_result.tolist() if hasattr(arima_result, 'tolist') else list(arima_result),
            "置信区间": conf_int,
            "趋势": trend,
            "增长率": round(growth_rate, 2),
        })

    return results


def get_aggregate_trend(df):
    """获取全体客户的汇总资产趋势"""
    ts_data = prepare_ts_data(df)
    monthly_totals = [0] * 12
    monthly_counts = [0] * 12

    for record in ts_data:
        for i, val in enumerate(record["月度资产"]):
            monthly_totals[i] += val
            monthly_counts[i] += 1

    avg_monthly = [t / c if c > 0 else 0 for t, c in zip(monthly_totals, monthly_counts)]
    months = [f"M{i+1}" for i in range(12)]

    return months, avg_monthly
