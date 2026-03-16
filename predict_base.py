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
# TRAIN MODEL 
# ─────────────────────────────────────────────
def train_model(train_df):
    
    # features
    features = [c for c in train_df.columns if c.endswith("_diff")]
    
    X = train_df[features]
    y = train_df["target"]

    # split validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
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
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )    

    # prédictions
    y_pred = model.predict(X_valid)

    # métriques
    metrics = {
        "logloss": log_loss(y_valid, y_pred),
        "auc": roc_auc_score(y_valid, y_pred),
        "accuracy": accuracy_score(y_valid, (y_pred > 0.5).astype(int)),
        "brier": brier_score_loss(y_valid, y_pred)
    }

    print("\nValidation metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.5f}")

    # importance
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importance()
    }).sort_values("importance", ascending=False)

    print("\nFeature importance")
    print(importance)

    return model

# ─────────────────────────────────────────────
# STAGES SUBMISSION DATA 
# ─────────────────────────────────────────────
def load_submission_template():
    stage1 = pd.read_csv(DATA_DIR + "SampleSubmissionStage1.csv")
    stage2 = pd.read_csv(DATA_DIR + "SampleSubmissionStage2.csv")
    return stage1, stage2

# ─────────────────────────────────────────────
# BUILD TESTING DATASET 
# ─────────────────────────────────────────────
def build_testing_set(sample, team_stats):
   
    df = sample.copy()

    # 1. Split ID proprement (vectorisé)
    id_split = df["ID"].str.split("_", expand=True)

    df["Season"] = id_split[0].astype(int)
    df["TeamLow"]  = id_split[1].astype(int)  # déjà low ID
    df["TeamHigh"]  = id_split[2].astype(int)  # déjà high ID

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
        
    features = ["Season", "TeamLow", "TeamHigh" ] + \
               [c for c in df.columns if c.endswith("_diff")]    

    df = df[features]

    return df

# ─────────────────────────────────────────────
# SUBMISSION
# ─────────────────────────────────────────────

def generate_submission(test, model, submission_path, clip):
    features = [c for c in test.columns if c.endswith("_diff")]

    X = test[features]

    # Prédiction batch
    preds = model.predict(X)

    # Remplacer NaN (équipes manquantes) par 0.5
    preds = np.where(np.isnan(preds), 0.5, preds)
    preds = np.clip(preds, clip, 1 - clip)

    test["Pred"] = preds

    # Export
    test["ID"] = (
        test["Season"].astype(str) + "_" +
        test["TeamLow"].astype(str) + "_" +
        test["TeamHigh"].astype(str)
    )    
    submission = test[["ID", "Pred"]]
    submission.to_csv(submission_path, index=False)

    print(f"Soumission sauvegardée : {submission_path}")

    return submission

# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

print("="*60)
print("March Mania 2026 — Models")
print("="*60)

# print("\n[01.1] Loading teams elo")
# team_stats = load_team_elo()
# print("\n[01.2] Loading teams stats")
# team_stats = load_team_stats()
print("\n[01.3] Loading teams advelo")
team_stats = load_team_advelo()

print("\n[02] Loading tourneys")
tourney = load_tourney()
    
print("\n[03] Building dataset train")
train = build_training_set(tourney, team_stats)
print(f"train shape : {train.shape}, {list(train.columns)}")
    
print("\n[04] Training model")
model = train_model(train)

print("\n[05] Loading submission templates")
stage1, stage2 = load_submission_template()

print("\n[06] Building dataset test (stage1 and stage2")
test_stage1 = build_testing_set(stage1, team_stats)
test_stage2 = build_testing_set(stage2, team_stats)
print(f"test stage1 shape: {test_stage1.shape}, {list(test_stage1.columns)}")
print(f"test stage2 shape: {test_stage2.shape}, {list(test_stage2.columns)}")


print("\n[07] Generating submissions (stage1 and stage2")
sub_stage1 = generate_submission(test_stage1, model, 'stage1_sub.csv', 0.0250)
print(f"Stage 1 saved: stage1_sub.csv")
sub_stage2 = generate_submission(test_stage2, model, 'stage2_sub.csv', 0.0250)
print(f"Stage 2 saved: stage2_sub.csv")


