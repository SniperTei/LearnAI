"""聚类分析：客户分群"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


def prepare_features(df):
    """准备聚类特征"""
    feature_cols = ["资产总额(AUM)", "月均交易次数", "月均交易金额", "持有产品数",
                    "存款", "理财", "基金", "保险", "信托"]
    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, feature_cols, scaler


def find_optimal_k(X, k_range=range(2, 9)):
    """手肘法 + 轮廓系数确定最优K"""
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    return list(k_range), inertias, silhouettes


def run_kmeans(df, n_clusters=4):
    """运行 K-Means 聚类"""
    X_scaled, feature_cols, scaler = prepare_features(df)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    df_result = df.copy()
    df_result["聚类标签"] = labels

    # 客群画像
    profiles = df_result.groupby("聚类标签")[
        ["资产总额(AUM)", "月均交易次数", "持有产品数", "年龄"]
    ].mean()

    profile_names = _generate_profile_names(profiles)
    df_result["客群名称"] = df_result["聚类标签"].map(profile_names)

    # 用于可视化的 PCA 降维
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    return df_result, profiles, profile_names, X_pca, labels


def run_dbscan(df, eps=1.5, min_samples=10):
    """运行 DBSCAN 聚类"""
    X_scaled, feature_cols, scaler = prepare_features(df)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    return labels, X_pca, n_clusters, n_noise


def get_radar_data(df_result, feature_cols):
    """获取雷达图数据（各聚类中心标准化）"""
    scaler = StandardScaler()
    X = df_result[feature_cols]
    X_scaled = scaler.fit_transform(X)

    df_temp = df_result.copy()
    df_temp[feature_cols] = X_scaled

    radar = df_temp.groupby("聚类标签")[feature_cols].mean()
    return radar


def _generate_profile_names(profiles):
    """根据客群特征生成描述性名称"""
    names = {}
    for idx, row in profiles.iterrows():
        aum = row["资产总额(AUM)"]
        tx = row["月均交易次数"]
        products = row["持有产品数"]

        if aum > 8_000_000:
            label = "高净值活跃群"
        elif aum > 3_000_000:
            label = "中产成长群" if tx > 10 else "中产稳健群"
        elif aum > 500_000:
            label = "大众潜力群" if products > 3 else "大众基础群"
        else:
            label = "长尾客户群"

        names[idx] = label
    return names
