import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans


BASE_PATH = r"Your File Path"
INPUT_FILE = os.path.join(BASE_PATH, "First League _ Since 2000.xlsx")
OUTPUT_FILE = os.path.join(BASE_PATH, "league_advanced_analytics.csv")


df = pd.read_excel(INPUT_FILE)


gf_ga = df["Goals Scored: Goals Conceded"].str.split(":", expand=True)
df["GoalsFor"] = gf_ga[0].astype(int)
df["GoalsAgainst"] = gf_ga[1].astype(int)


df["Games"] = df["Wins"] + df["Draws"] + df["Defeats"]


df["IsCurrentSeason"] = df["Season"] == "25/26"


df["GF_per_match"] = df["GoalsFor"] / df["Games"]
df["GA_per_match"] = df["GoalsAgainst"] / df["Games"]



completed_mask = (~df["IsCurrentSeason"]) & df["Games"].isin([30, 34])
train = df.loc[completed_mask].copy()

features = train[["GF_per_match", "GA_per_match"]]
target = train["Points"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(
        n_estimators=500,
        random_state=42
    ))
])


scores = cross_val_score(
    model,
    features,
    target,
    cv=5,
    scoring="neg_mean_absolute_error"
)
print(f"CV MAE (points): {-scores.mean():.2f} ± {scores.std():.2f}")


model.fit(features, target)


features_all = df[["GF_per_match", "GA_per_match"]]
df["ExpectedPoints"] = model.predict(features_all)


is_completed_season = (~df["IsCurrentSeason"]) & df["Games"].isin([30, 34])

df["OverPerformance"] = np.where(
    is_completed_season,
    df["Points"] - df["ExpectedPoints"],
    np.nan
)


cluster_features = train[["GF_per_match", "GA_per_match", "Points"]]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
train["Cluster"] = kmeans.fit_predict(cluster_features)


cluster_order = (
    train.groupby("Cluster")["Points"]
    .mean()
    .sort_values()
    .index
    .tolist()
)

cluster_labels = {
    cluster_order[0]: "Below Model Prediction",
    cluster_order[1]: "Within Model Range",
    cluster_order[2]: "Above Model Prediction",
}

train["ClusterLabel"] = train["Cluster"].map(cluster_labels)


df = df.merge(
    train[["Season", "ClusterLabel"]],
    on="Season",
    how="left"
)


df_out = df[[
    "Season",
    "Games",
    "GoalsFor",
    "GoalsAgainst",
    "GF_per_match",
    "GA_per_match",
    "Points",
    "ExpectedPoints",
    "OverPerformance",
    "IsCurrentSeason",
    "ClusterLabel"
]].copy()


df_out["GF_per_match"] = df_out["GF_per_match"].round(3)
df_out["GA_per_match"] = df_out["GA_per_match"].round(3)
df_out["ExpectedPoints"] = df_out["ExpectedPoints"].round(2)
df_out["OverPerformance"] = df_out["OverPerformance"].round(2)


df_out = df_out.sort_values("Season")


df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"Saved: {OUTPUT_FILE}")
