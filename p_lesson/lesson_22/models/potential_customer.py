"""潜力高价值客户预测：预测非高价值客户中谁将在未来3个月升级"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
from statsmodels.tsa.arima.model import ARIMA


def generate_future_labels(df, threshold=5_000_000):
    """根据12个月资产趋势预测未来3个月AUM，生成潜力标签"""
    records = []

    for _, row in df.iterrows():
        monthly = [float(x) for x in row["近12月资产"].split(",")]
        current_aum = monthly[-1]
        is_current_hv = row["是否高价值客户"]

        if is_current_hv == 1:
            # 已是高价值客户，标记为 0（不参与"潜力"预测）
            future_label = 0
            projected_aum = current_aum
        else:
            # 用最近趋势外推3个月
            try:
                model = ARIMA(monthly, order=(1, 1, 1))
                fitted = model.fit()
                forecast = fitted.forecast(steps=3)
                projected_aum = float(forecast.iloc[-1]) if hasattr(forecast, 'iloc') else float(forecast[-1])
            except Exception:
                # 回退：线性回归外推
                x = np.arange(12)
                coeffs = np.polyfit(x, monthly, 1)
                projected_aum = coeffs[0] * 14 + coeffs[1]

            future_label = 1 if projected_aum >= threshold else 0

        # 增长率特征
        growth_rate = (monthly[-1] - monthly[0]) / monthly[0] if monthly[0] > 0 else 0
        # 近3个月增长率
        recent_growth = (monthly[-1] - monthly[-3]) / monthly[-3] if monthly[-3] > 0 else 0
        # 波动率
        volatility = np.std(monthly) / np.mean(monthly) if np.mean(monthly) > 0 else 0

        records.append({
            "客户ID": row["客户ID"],
            "年龄": row["年龄"],
            "城市": row["城市"],
            "职业": row["职业"],
            "性别": row["性别"],
            "AUM": row["资产总额(AUM)"],
            "存款": row["存款"],
            "理财": row["理财"],
            "基金": row["基金"],
            "保险": row["保险"],
            "信托": row["信托"],
            "月均交易次数": row["月均交易次数"],
            "月均交易金额": row["月均交易金额"],
            "持有产品数": row["持有产品数"],
            "客户等级": row["客户等级"],
            "增长率": growth_rate,
            "近3月增长率": recent_growth,
            "波动率": volatility,
            "预测AUM": round(projected_aum, 2),
            "是否当前高价值": is_current_hv,
            "是否潜力客户": future_label,
        })

    return pd.DataFrame(records)


def train_potential_model(df_labeled):
    """训练逻辑回归模型预测潜力客户"""
    # 只用非高价值客户训练
    train_df = df_labeled[df_labeled["是否当前高价值"] == 0].copy()

    feature_cols = ["年龄", "月均交易次数", "月均交易金额", "持有产品数",
                    "存款", "理财", "基金", "保险", "信托",
                    "增长率", "近3月增长率", "波动率"]

    # 编码分类变量
    le_gender = LabelEncoder()
    le_occ = LabelEncoder()
    train_df["性别_enc"] = le_gender.fit_transform(train_df["性别"])
    train_df["职业_enc"] = le_occ.fit_transform(train_df["职业"])
    feature_cols += ["性别_enc", "职业_enc"]

    X = train_df[feature_cols]
    y = train_df["是否潜力客户"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 0 else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    feature_importance = dict(zip(feature_cols, np.abs(model.coef_[0])))

    return model, scaler, le_gender, le_occ, feature_cols, report, fpr, tpr, roc_auc, feature_importance


def predict_potential_customers(df_labeled, model, scaler, le_gender, le_occ, feature_cols):
    """用训练好的模型预测所有非高价值客户的潜力概率"""
    non_hv = df_labeled[df_labeled["是否当前高价值"] == 0].copy()

    non_hv["性别_enc"] = le_gender.transform(non_hv["性别"])
    non_hv["职业_enc"] = le_occ.transform(non_hv["职业"])

    X = non_hv[feature_cols]
    X_scaled = scaler.transform(X)

    non_hv["潜力概率"] = model.predict_proba(X_scaled)[:, 1]
    non_hv["预测结果"] = model.predict(X_scaled)

    # 按概率排序
    result = non_hv.sort_values("潜力概率", ascending=False)

    return result[["客户ID", "客户等级", "AUM", "预测AUM", "增长率", "近3月增长率",
                   "持有产品数", "月均交易次数", "潜力概率", "预测结果"]]
