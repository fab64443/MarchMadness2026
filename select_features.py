# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

import warnings

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_DIR       = "/home/fabrice/notebooks/kaggle/marchmadness/input/"
WORK_DIR       = "/home/fabrice/notebooks/kaggle/marchmadness/work/"

# ─────────────────────────────────────────────
# LOAD PRECOMPUTED TEAMS DATA 
# ─────────────────────────────────────────────
def load_team_elo():
    # brier 2025 : 0.104718
    elo = pd.read_csv(WORK_DIR + "elo_ratings_2003_2026.csv")
    elo = elo[["Season","TeamID","elo_last","elo_vs_peak"]]
    print(f"elo shape : {elo.shape}, {list(elo.columns)}")
    return elo

def load_team_stats():
    # brier 2025 : 0.116735
    stats = pd.read_csv(WORK_DIR + "team_stats_2003-2026.csv")
    stats = stats[["Season","TeamID","Win","Margin","NetRate","OffRate","DefRate","Poss","eFGPct","TSPct",
                   "FGA3Rate","FTRate","TOVRate","ORPct","DRPct","Ast","Stl","Blk","PF","WinPct" ]]
    print(f"stats shape : {stats.shape}, {list(stats.columns)}")
    return stats

def load_team_advelo():
    # brier 2025 : 0.038587
    elo = pd.read_csv(WORK_DIR + "elo_ratings_2003_2026.csv")
    elo = elo[["Season","TeamID","elo_last","elo_vs_peak","elo_late_trend",
               "strength_of_schedule","quality_of_wins"]]
    print(f"elo shape : {elo.shape}, {list(elo.columns)}")
    return elo

# Mix advelo + stats : brier 2025 : 0.024594 

def load_road_warrior_index():
    # new brier 2025 : 0.021484
    rwi = pd.read_csv(WORK_DIR + "road_warrior_2003_2026.csv")
    rwi = rwi[["Season","TeamID","road_warrior"]]
    print(f"rwi shape : {rwi.shape}, {list(rwi.columns)}")
    return rwi

# ─────────────────────────────────────────────
# LOAD TOURNEY MATCHES 
# ─────────────────────────────────────────────
def load_tourney():
    m_tourney  = pd.read_csv(DATA_DIR+"MNCAATourneyDetailedResults.csv")
    w_tourney  = pd.read_csv(DATA_DIR+"WNCAATourneyDetailedResults.csv")
    tourney = pd.concat([m_tourney, w_tourney], ignore_index=True)
    print(f"tourney shape : {tourney.shape}, {list(tourney.columns)}")
    return tourney

# ─────────────────────────────────────────────
# BUILD TRAINING DATASET 
# ─────────────────────────────────────────────
def build_training_set(tourney, team_stats):

    df = tourney[["Season", "WTeamID", "LTeamID"]].copy()

    # TeamLow / TeamHigh
    df["TeamLow"]  = df[["WTeamID", "LTeamID"]].min(axis=1)
    df["TeamHigh"] = df[["WTeamID", "LTeamID"]].max(axis=1)
    df["target"] = (df["WTeamID"] == df["TeamLow"]).astype(int)

    dfr = df.copy()
    dfr["TeamLow"]  = df[["WTeamID", "LTeamID"]].max(axis=1)
    dfr["TeamHigh"] = df[["WTeamID", "LTeamID"]].min(axis=1)
    dfr["target"] = (df["WTeamID"] != df["TeamLow"]).astype(int)

    df = pd.concat([df, dfr], ignore_index=True).drop(columns=["WTeamID","LTeamID"])

    # Index stats
    stats = team_stats.set_index(["Season","TeamID"])

    stat_cols = [c for c in team_stats.columns if c not in ["Season","TeamID"]]

    # Join TeamLow
    low_stats = stats.rename(columns={c: f"{c}_low" for c in stat_cols})

    df = df.join(
        low_stats,
        on=["Season","TeamLow"]
    )

    # Join TeamHigh
    high_stats = stats.rename(columns={c: f"{c}_high" for c in stat_cols})

    df = df.join(
        high_stats,
        on=["Season","TeamHigh"]
    )

    # Diff features
    for col in stat_cols:
        df[f"{col}_diff"] = df[f"{col}_low"] - df[f"{col}_high"]
        
    features = ["Season", "TeamLow", "TeamHigh", "target"] + \
               [c for c in df.columns if c.endswith("_diff")]    

    df = df[features]
    return df

# ─────────────────────────────────────────────
# PERMUTATION IMPORTANCE (custom, rapide)
# ─────────────────────────────────────────────
def permutation_importance_fold(model, X_valid, y_valid, features, n_repeats=5):

    baseline_preds = model.predict(X_valid)
    baseline_score = log_loss(y_valid, baseline_preds)

    importances = np.zeros(len(features))

    for i, col in enumerate(features):

        scores = []

        for _ in range(n_repeats):
            X_perm = X_valid.copy()
            X_perm[col] = np.random.permutation(X_perm[col].values)

            preds = model.predict(X_perm)
            score = log_loss(y_valid, preds)

            scores.append(score)

        importances[i] = np.mean(scores) - baseline_score

    return importances

# ─────────────────────────────────────────────
# TRAIN MODEL + PERMUTATION IMPORTANCE
# ─────────────────────────────────────────────
def train_model_cv_with_perm(train_df, n_splits=5):

    features = [c for c in train_df.columns if c.endswith("_diff")]

    X = train_df[features]
    y = train_df["target"]

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X))
    models = []
    fold_metrics = []

    perm_importance_total = np.zeros(len(features))

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):

        print(f"\n{'='*40}")
        print(f"FOLD {fold + 1} / {n_splits}")
        print(f"{'='*40}")

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        params = {
            "objective": "binary",
            "metric": ["binary_logloss"],
            "learning_rate": 0.02,
            "num_leaves": 64,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbosity": -1
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[valid_data],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(100)
            ]
        )

        # ── OOF preds
        fold_preds = model.predict(X_valid)
        oof_preds[valid_idx] = fold_preds

        # ── metrics
        metrics = {
            "logloss": log_loss(y_valid, fold_preds),
            "auc": roc_auc_score(y_valid, fold_preds),
            "accuracy": accuracy_score(y_valid, (fold_preds > 0.5).astype(int)),
            "brier": brier_score_loss(y_valid, fold_preds)
        }
        fold_metrics.append(metrics)

        print(f"\nFold {fold + 1} metrics")
        for k, v in metrics.items():
            print(f"  {k}: {v:.5f}")

        # ── permutation importance (sur fold)
        print("  -> computing permutation importance...")
        perm_imp = permutation_importance_fold(
            model, X_valid, y_valid, features, n_repeats=5
        )

        perm_importance_total += perm_imp

        models.append(model)

    # OOF METRICS
    print(f"\n{'='*40}")
    print("OOF METRICS")
    print(f"{'='*40}")

    oof_metrics = {
        "logloss": log_loss(y, oof_preds),
        "auc": roc_auc_score(y, oof_preds),
        "accuracy": accuracy_score(y, (oof_preds > 0.5).astype(int)),
        "brier": brier_score_loss(y, oof_preds)
    }

    for k, v in oof_metrics.items():
        print(f"{k}: {v:.5f}")

    # PERMUTATION IMPORTANCE (moyenne)
    perm_importance_mean = perm_importance_total / n_splits

    perm_df = pd.DataFrame({
        "feature": features,
        "importance": perm_importance_mean
    }).sort_values("importance", ascending=False)

    print("\nPermutation importance (CV)")
    print(perm_df)

    # FEATURE SELECTION
    # seuil simple
    selected_features = perm_df[
        perm_df["importance"] > 0
    ]["feature"].tolist()

    print(f"\nSelected features ({len(selected_features)}/{len(features)})")
    print(selected_features)

    return models, oof_preds, perm_df, selected_features
    
# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

print("="*60)
print("March Mania 2026 — Models")
print("="*60)

print("\n[01.2] Loading teams stats")
team_stats = load_team_stats()

print("\n[01.3] Loading teams advelo")
advelo = load_team_advelo()
team_stats = team_stats.merge(advelo, on=["Season", "TeamID"], how="left")

print("\n[01.4] Loading road warrior index")
rwi = load_road_warrior_index()
team_stats = team_stats.merge(rwi, on=["Season", "TeamID"], how="left")

print("\n[02] Loading tourneys")
tourney = load_tourney()
    
print("\n[03] Building dataset train")
train = build_training_set(tourney, team_stats)
print(f"train shape : {train.shape}, {list(train.columns)}")
    
print("\n[04.1] Training model")
models, oof_preds, perm_df, selected_features = train_model_cv_with_perm(train)

