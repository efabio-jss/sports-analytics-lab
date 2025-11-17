import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


BASE_PATH = r"Your File Path"
INPUT_FILE = os.path.join(BASE_PATH, "First League _ Since 2000.xlsx")
OUTPUT_FILE = os.path.join(BASE_PATH, "season_25_26_forecast.csv")

N_SIM = 10_000
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

df = pd.read_excel(INPUT_FILE)

gf_ga = df["Goals Scored: Goals Conceded"].str.split(":", expand=True)
df["GoalsFor"] = gf_ga[0].astype(int)
df["GoalsAgainst"] = gf_ga[1].astype(int)

df["Games"] = df["Wins"] + df["Draws"] + df["Defeats"]

df["IsCurrentSeason"] = df["Season"] == "25/26"


df["GF_per_match"] = df["GoalsFor"] / df["Games"]
df["GA_per_match"] = df["GoalsAgainst"] / df["Games"]
df["Points_per_match"] = df["Points"] / df["Games"]
df["GoalDiff_per_match"] = (df["GoalsFor"] - df["GoalsAgainst"]) / df["Games"]


is_completed = (~df["IsCurrentSeason"]) & df["Games"].isin([30, 34])
hist = df.loc[is_completed].copy()


hist["FinalPoints"] = hist["Points"]
hist["FinalGoalsFor"] = hist["GoalsFor"]
hist["FinalGoalsAgainst"] = hist["GoalsAgainst"]


feature_cols = ["GF_per_match", "GA_per_match", "Points_per_match", "GoalDiff_per_match"]
X = hist[feature_cols]

y_points = hist["FinalPoints"]
y_gf = hist["FinalGoalsFor"]
y_ga = hist["FinalGoalsAgainst"]


rf_points = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE)
rf_gf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE)
rf_ga = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE)

rf_points.fit(X, y_points)
rf_gf.fit(X, y_gf)
rf_ga.fit(X, y_ga)


hist["PredPoints_hist"] = rf_points.predict(X)
hist["PredGF_hist"] = rf_gf.predict(X)
hist["PredGA_hist"] = rf_ga.predict(X)

res_points = hist["FinalPoints"] - hist["PredPoints_hist"]
res_gf = hist["FinalGoalsFor"] - hist["PredGF_hist"]
res_ga = hist["FinalGoalsAgainst"] - hist["PredGA_hist"]

sigma_points = res_points.std(ddof=1)
sigma_gf = res_gf.std(ddof=1)
sigma_ga = res_ga.std(ddof=1)


if sigma_points == 0:
    sigma_points = 1.0
if sigma_gf == 0:
    sigma_gf = 1.0
if sigma_ga == 0:
    sigma_ga = 1.0


current = df[df["Season"] == "25/26"].copy()
if current.empty:
    raise ValueError("Season '25/26' not found in the dataset.")

X_current = current[feature_cols]

pred_points_mean = float(rf_points.predict(X_current)[0])
pred_gf_mean = float(rf_gf.predict(X_current)[0])
pred_ga_mean = float(rf_ga.predict(X_current)[0])

MAX_GAMES = 34
MAX_POINTS = MAX_GAMES * 3

pred_points_mean = max(0.0, min(pred_points_mean, MAX_POINTS))
pred_gf_mean = max(0.0, pred_gf_mean)
pred_ga_mean = max(0.0, pred_ga_mean)


points_samples = np.random.normal(loc=pred_points_mean, scale=sigma_points, size=N_SIM)
gf_samples = np.random.normal(loc=pred_gf_mean, scale=sigma_gf, size=N_SIM)
ga_samples = np.random.normal(loc=pred_ga_mean, scale=sigma_ga, size=N_SIM)


points_samples = np.clip(points_samples, 0, MAX_POINTS)
gf_samples = np.clip(gf_samples, 0, None)
ga_samples = np.clip(ga_samples, 0, None)


def summary_stats(arr):
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }

points_stats = summary_stats(points_samples)
gf_stats = summary_stats(gf_samples)
ga_stats = summary_stats(ga_samples)


THRESHOLD_80 = 80
prob_ge_80 = float((points_samples >= THRESHOLD_80).mean())


champions = hist[hist["Rank"] == 1]
if champions.empty:
    champion_threshold = float(hist["FinalPoints"].quantile(0.75))
else:
    champion_threshold = float(champions["FinalPoints"].median())

prob_ge_champion_thr = float((points_samples >= champion_threshold).mean())


pred_points_mean_mc = points_stats["mean"]
pred_gf_mean_mc = gf_stats["mean"]
pred_ga_mean_mc = ga_stats["mean"]

pred_GFpm = pred_gf_mean_mc / MAX_GAMES
pred_GApm = pred_ga_mean_mc / MAX_GAMES
pred_Ppm = pred_points_mean_mc / MAX_GAMES

current_games = int(current["Games"].values[0])


output = pd.DataFrame({
    "Season": ["25/26"],
    "CurrentGames": [current_games],

    
    "RF_PredFinalPoints": [round(pred_points_mean, 2)],
    "RF_PredFinalGoalsFor": [round(pred_gf_mean, 2)],
    "RF_PredFinalGoalsAgainst": [round(pred_ga_mean, 2)],

    
    "MC_Points_Mean": [round(points_stats["mean"], 2)],
    "MC_Points_Std": [round(points_stats["std"], 2)],
    "MC_Points_P10": [round(points_stats["p10"], 2)],
    "MC_Points_P50": [round(points_stats["p50"], 2)],
    "MC_Points_P90": [round(points_stats["p90"], 2)],

    
    "MC_GF_Mean": [round(gf_stats["mean"], 2)],
    "MC_GA_Mean": [round(ga_stats["mean"], 2)],

    
    "MC_Pred_GF_per_match": [round(pred_GFpm, 3)],
    "MC_Pred_GA_per_match": [round(pred_GApm, 3)],
    "MC_Pred_Points_per_match": [round(pred_Ppm, 3)],

    
    "Threshold_80_points": [THRESHOLD_80],
    "Prob_Points_ge_80": [round(prob_ge_80, 3)],

    "ChampionPointsThreshold": [round(champion_threshold, 2)],
    "Prob_Points_ge_ChampionThreshold": [round(prob_ge_champion_thr, 3)],
})

output.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"Forecast saved: {OUTPUT_FILE}")
