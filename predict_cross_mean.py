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

def load_team_stats_opti():
    # brier 2025 : 0.116735
    stats = pd.read_csv(WORK_DIR + "team_stats_2003-2026.csv")
    stats = stats[["Season","TeamID","Win","NetRate","DefRate",
                   "ORPct","DRPct","Ast","Blk","WinPct" ]]
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
# TRAIN MODEL WITH CROSS-VALIDATION
# ─────────────────────────────────────────────
def train_model_cv(train_df, n_splits=5):

    # features
    features = [c for c in train_df.columns if c.endswith("_diff")]

    X = train_df[features]
    y = train_df["target"]

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X))
    models = []
    fold_metrics = []

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

        # prédictions OOF
        fold_preds = model.predict(X_valid)
        oof_preds[valid_idx] = fold_preds

        # métriques par fold
        metrics = {
            "logloss": log_loss(y_valid, fold_preds),
            "auc":     roc_auc_score(y_valid, fold_preds),
            "accuracy": accuracy_score(y_valid, (fold_preds > 0.5).astype(int)),
            "brier":   brier_score_loss(y_valid, fold_preds)
        }
        fold_metrics.append(metrics)

        print(f"\nFold {fold + 1} metrics")
        for k, v in metrics.items():
            print(f"  {k}: {v:.5f}")

        models.append(model)

    # ── métriques OOF globales ──────────────────
    print(f"\n{'='*40}")
    print("OOF METRICS (all folds)")
    print(f"{'='*40}")

    oof_metrics = {
        "logloss":  log_loss(y, oof_preds),
        "auc":      roc_auc_score(y, oof_preds),
        "accuracy": accuracy_score(y, (oof_preds > 0.5).astype(int)),
        "brier":    brier_score_loss(y, oof_preds)
    }

    fold_df = pd.DataFrame(fold_metrics)
    for k in oof_metrics:
        mean_v = fold_df[k].mean()
        std_v  = fold_df[k].std()
        oof_v  = oof_metrics[k]
        print(f"  {k}: OOF={oof_v:.5f}  |  mean±std={mean_v:.5f}±{std_v:.5f}")

    # ── importance moyenne ──────────────────────
    mean_importance = np.mean(
        [m.feature_importance() for m in models], axis=0
    )
    importance = pd.DataFrame({
        "feature":    features,
        "importance": mean_importance
    }).sort_values("importance", ascending=False)

    print("\nFeature importance (moyenne des folds)")
    print(importance)

    return models, oof_preds

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
# GENERATE SUBMISSION 
# ─────────────────────────────────────────────
def generate_submission(test, models, submission_path, clip, extra_features=None):

    features = [c for c in test.columns if c.endswith("_diff")]
    X = test[features]

    # ── Niveau 1 : moyenne des N modèles (input du méta-modèle) ──
    preds = np.mean([model.predict(X) for model in models], axis=0)

    # ── Post-processing ───────────────────────────────────────────
    preds = np.where(np.isnan(preds), 0.5, preds)
    preds = np.clip(preds, clip, 1 - clip)

    # ── Build submission ──────────────────────────────────────────
    test = test.copy()
    test["Pred"] = preds
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

print("\n[01.2] Loading teams stats")
team_stats = load_team_stats_opti()

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
models, oof_preds = train_model_cv(train)

print("\n[05] Loading submission templates")
stage1, stage2 = load_submission_template()

print("\n[06] Building dataset test (stage1 and stage2)")
test_stage1 = build_testing_set(stage1, team_stats)
test_stage2 = build_testing_set(stage2, team_stats)
print(f"test stage1 shape: {test_stage1.shape}, {list(test_stage1.columns)}")
print(f"test stage2 shape: {test_stage2.shape}, {list(test_stage2.columns)}")


print("\n[07] Generating submissions (stage1 and stage2)")
sub_stage1 = generate_submission(test_stage1, models, 'stage1_sub.csv', 0.0250)
print(f"Stage 1 saved: stage1_sub.csv")
sub_stage2 = generate_submission(test_stage2, models, 'stage2_sub.csv', 0.0250)
print(f"Stage 2 saved: stage2_sub.csv")


