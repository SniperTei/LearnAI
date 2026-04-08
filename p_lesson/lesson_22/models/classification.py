"""分类模型：预测高价值客户"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


def prepare_features(df):
    """准备分类特征"""
    feature_cols = ["年龄", "月均交易次数", "月均交易金额", "持有产品数",
                    "存款", "理财", "基金", "保险", "信托"]
    X = df[feature_cols].copy()

    # 编码分类变量
    le_gender = LabelEncoder()
    le_city = LabelEncoder()
    le_occ = LabelEncoder()

    X["性别"] = le_gender.fit_transform(df["性别"])
    X["城市"] = le_city.fit_transform(df["城市"])
    X["职业"] = le_occ.fit_transform(df["职业"])

    feature_cols += ["性别", "城市", "职业"]

    y = df["是否高价值客户"]
    return X, y, feature_cols


def train_models(df):
    """训练多个分类模型并返回结果"""
    X, y, feature_cols = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
        "决策树": DecisionTreeClassifier(max_depth=8, random_state=42),
        "GBDT": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "随机森林": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42
        ),
    }

    results = []
    trained = {}
    roc_data = {}
    feature_importance = {}

    for name, model in models.items():
        if name == "逻辑回归":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            imp = np.abs(model.coef_[0])
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            imp = model.feature_importances_

        results.append({
            "模型": name,
            "准确率": round(accuracy_score(y_test, y_pred), 4),
            "精确率": round(precision_score(y_test, y_pred), 4),
            "召回率": round(recall_score(y_test, y_pred), 4),
            "F1分数": round(f1_score(y_test, y_pred), 4),
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}

        trained[name] = model
        feature_importance[name] = dict(zip(feature_cols, imp))

    results_df = pd.DataFrame(results)
    return results_df, roc_data, feature_importance, trained
