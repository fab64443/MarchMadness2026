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
warnings.filterwarnings("ignore")

# +
# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR       = "/home/fabrice/notebooks/kaggle/marchmadness/input"
WORK_DIR       = "/home/fabrice/notebooks/kaggle/marchmadness/work/"

# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

def load_data(data_dir):
    m_reg      = pd.read_csv(os.path.join(data_dir, "MRegularSeasonDetailedResults.csv"))
    w_reg      = pd.read_csv(os.path.join(data_dir, "WRegularSeasonDetailedResults.csv"))
    m_tourney  = pd.read_csv(os.path.join(data_dir, "MNCAATourneyDetailedResults.csv"))
    w_tourney  = pd.read_csv(os.path.join(data_dir, "WNCAATourneyDetailedResults.csv"))
    sample_all = pd.read_csv(os.path.join(data_dir, "SampleSubmissionStage1.csv"))

    print(f"Hommes  — saison régulière : {len(m_reg):,}")
    print(f"Femmes  — saison régulière : {len(w_reg):,}")
    print(f"Hommes  — tournoi          : {len(m_tourney):,}")
    print(f"Femmes  — tournoi          : {len(w_tourney):,}")
    print(f"Soumission sample          : {len(sample_all):,}")

    return m_reg, w_reg, m_tourney, w_tourney, sample_all

# ─────────────────────────────────────────────
# 2. INFO ÉQUIPE SAISON
# ─────────────────────────────────────────────

def compute_adv_statistics(teams):
    teams["Margin"] = teams["ScoreOff"] - teams["ScoreDef"]
    teams["Poss"] = teams["FGA"] - teams["ORd"] + teams["TOr"] + 0.44 * teams["FTA"]
    teams["OffRate"] = teams["ScoreOff"] / teams["Poss"]
    teams["DefRate"] = teams["ScoreDef"] / teams["Poss"]
    teams["NetRate"] = teams["OffRate"] - teams["DefRate"]
    teams["eFGPct"] = (teams["FGM"] + 0.5 * teams["FGM3"]) / teams["FGA"]
    teams["TSPct"] = teams["ScoreOff"] /  (2 * (teams["FGM"] + 0.44 * teams["FTA"]))
    teams["FGA3Rate"] = teams["FGA3"] / teams["FGA"]
    teams["FTRate"] = teams["FTA"] / teams["FGA"]
    teams["TOVRate"] = teams["TOr"] / teams["Poss"]
    teams["ORPct"] = teams["ORd"] / (teams["ORd"] + teams["DRd"])
    teams["DRPct"] = teams["DRd"] / (teams["ORd"] + teams["DRd"])

    return teams

    
def build_team_infos(reg_df, start_season):
    reg_df = reg_df[reg_df.Season >= start_season].copy()

    # Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT, WFGM,WFGA,WFGM3,WFGA3,WFTM,WFTA,WOR,WDR,WAst,WTO,WStl,WBlk,WPF,
    #                                                         LFGM,LFGA,LFGM3,LFGA3,LFTM,LFTA,LOR,LDR,LAst,LTO,LStl,LBlk,LPF

    winners = pd.DataFrame({
        "Season": reg_df["Season"],
        "TeamID": reg_df["WTeamID"],
        "DayNum": reg_df["DayNum"],
        "ScoreOff": reg_df["WScore"],
        "ScoreDef" : reg_df["LScore"],
        "Loc" : reg_df["WLoc"],
        "NumOT" : reg_df["NumOT"],
        "FGM" : reg_df["WFGM"],
        "FGA" : reg_df["WFGA"],
        "FGM3" : reg_df["WFGM3"],
        "FGA3" : reg_df["WFGA3"],
        "FTM" : reg_df["WFTM"],
        "FTA" : reg_df["WFTA"],
        "ORd" : reg_df["WOR"],
        "DRd" : reg_df["WDR"],
        "Ast" : reg_df["WAst"],
        "TOr" : reg_df["WTO"],
        "Stl" : reg_df["WStl"],
        "Blk" : reg_df["WBlk"],
        "PF" : reg_df["WPF"],
        "Win": 1
    })    

    losers = pd.DataFrame({
        "Season": reg_df["Season"],
        "TeamID": reg_df["LTeamID"],
        "DayNum": reg_df["DayNum"],
        "ScoreOff": reg_df["LScore"],
        "ScoreDef": reg_df["WScore"],
        "Loc" : reg_df["WLoc"].map({"A": "H","H": "A"}).fillna("N"),
        "NumOT" : reg_df["NumOT"],
        "FGM" : reg_df["LFGM"],
        "FGA" : reg_df["LFGA"],
        "FGM3" : reg_df["LFGM3"],
        "FGA3" : reg_df["LFGA3"],
        "FTM" : reg_df["LFTM"],
        "FTA" : reg_df["LFTA"],
        "ORd" : reg_df["LOR"],
        "DRd" : reg_df["LDR"],
        "Ast" : reg_df["LAst"],
        "TOr" : reg_df["LTO"],
        "Stl" : reg_df["LStl"],
        "Blk" : reg_df["LBlk"],
        "PF" : reg_df["LPF"],
        "Win": 0
    })

    teams = pd.concat([winners, losers], ignore_index=True)
    # advanced statistics
    teams = compute_adv_statistics(teams)

    print(f"teams - lignes: {teams.shape[0]:>7,} colonnes: {list(teams.columns)}")   
    return teams


# ─────────────────────────────────────────────
# 3a. STATS ÉQUIPE SAISON
# ─────────────────────────────────────────────

def build_team_stats(teams):

    stats = teams.groupby(["Season", "TeamID"], as_index=False).mean(numeric_only=True)
    
    df = teams.groupby(["Season", "TeamID"], as_index=False).agg(
            Games=("Win", "count"),
            Wins=("Win", "sum") 
        )

    stats = stats.merge(df, on=["Season", "TeamID"], how="left")
    stats["WinPct"] = stats["Wins"] / stats["Games"]
    
    print(f"stats - lignes: {stats.shape[0]:>5,} colonnes: {list(stats.columns)}")   
    
    return stats


# ─────────────────────────────────────────────
# 3b. ROAD WARRIOR INDEX
# ─────────────────────────────────────────────

def compute_road_warrior_index(reg_results):

    # Matches où le winner joue à domicile
    home_games = reg_results[reg_results["WLoc"] == "H"]

    home_rows = pd.DataFrame({
        "Season": pd.concat([home_games["Season"], home_games["Season"]]),
        "TeamID": pd.concat([home_games["WTeamID"], home_games["LTeamID"]]),
        "loc": ["home"] * len(home_games) + ["away"] * len(home_games),
        "win": [1] * len(home_games) + [0] * len(home_games)
    })

    # Matches où le winner joue à l'extérieur
    away_games = reg_results[reg_results["WLoc"] == "A"]

    away_rows = pd.DataFrame({
        "Season": pd.concat([away_games["Season"], away_games["Season"]]),
        "TeamID": pd.concat([away_games["WTeamID"], away_games["LTeamID"]]),
        "loc": ["away"] * len(away_games) + ["home"] * len(away_games),
        "win": [1] * len(away_games) + [0] * len(away_games)
    })

    df = pd.concat([home_rows, away_rows], ignore_index=True)

    pivot = (
        df.groupby(["Season", "TeamID", "loc"])["win"]
        .mean()
        .unstack(fill_value=0.5)
    )

    pivot["road_warrior"] = pivot.get("away", 0.5) - pivot.get("home", 0.5)

    return pivot["road_warrior"].reset_index()	    

# ─────────────────────────────────────────────
# 4. TRAINING DATASET
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
        
    df = df.drop(columns=[c for c in df.columns if c.endswith("_low") or c.endswith("_high")])

    features = ["Season", "TeamLow", "TeamHigh", "target", "Win_diff", "Margin_diff", "NetRate_diff", 
                "OffRate_diff", "DefRate_diff", "Poss_diff", "eFGPct_diff", "TSPct_diff", "FGA3Rate_diff", "FTRate_diff",
                "TOVRate_diff", "ORPct_diff", "DRPct_diff", "Ast_diff", "Stl_diff", "Blk_diff", "PF_diff", "WinPct_diff"]

    df = df[features]
    return df

# ─────────────────────────────────────────────
# 5. ENTRAÎNEMENT
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

    # # importance
    # importance = pd.DataFrame({
    #     "feature": features,
    #     "importance": model.feature_importance()
    # }).sort_values("importance", ascending=False)

    # print("\nFeature importance")
    # print(importance)

    return model
    

# ─────────────────────────────────────────────
# 6. TESTING DATASET
# ─────────────────────────────────────────────

def build_testing_set(sample, team_stats):
   
    df = sample.copy()
    team_stats = team_stats.drop(columns=["DayNum"])

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

    df = df.drop(columns=[c for c in df.columns if c.endswith("_low") or c.endswith("_high")])
    
    features = ["Season", "TeamLow", "TeamHigh", "Win_diff", "Margin_diff", "NetRate_diff", 
                "OffRate_diff", "DefRate_diff", "Poss_diff", "eFGPct_diff", "TSPct_diff", "FGA3Rate_diff", "FTRate_diff",
                "TOVRate_diff", "ORPct_diff", "DRPct_diff", "Ast_diff", "Stl_diff", "Blk_diff", "PF_diff", "WinPct_diff"]

    df = df[features]

    return df

# ─────────────────────────────────────────────
# 7. GÉNÉRATION SOUMISSION
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

def evaluate(args):

    print("="*60)
    print("March Mania 2026 — Stats Regular Season")
    print("="*60)

    print("\n[1/7] Chargement")
    m_reg, w_reg, m_tourney, w_tourney, sample_all = load_data(args.data_dir)
    reg = pd.concat([m_reg, w_reg], ignore_index=True)
    tourney = pd.concat([m_tourney, w_tourney], ignore_index=True)
    del m_reg, w_reg, m_tourney, w_tourney

    print("\nRoad warrior index")
    roadwarrior = compute_road_warrior_index(reg)
    roadwarrior.to_csv(WORK_DIR + 'road_warrior_2003_2026.csv', index=False)
    return {},{}
    
    print("\n[2/7] Construction des infos équipes")
    teams = build_team_infos(reg, args.start_season)
    print(f"teams - lignes: {teams.shape[0]:>7,} colonnes: {list(teams.columns)}")   
    
    print("\n[3/7] Construction des stats équipes")
    reg_stats = build_team_stats(teams)
    print(f"stats - lignes: {reg_stats.shape[0]:>7,}")
    print(f"Sauvegarde des stats par équipe")
    reg_stats.to_csv(WORK_DIR+'team_stats_2003-2026.csv', index=False)

    print("\n[4/7] Construction dataset train")
    train = build_training_set(tourney, reg_stats)
    print(f"train - lignes: {train.shape[0]:>7,} colonnes: {list(train.columns)}")   
    print("Taille train:", len(train))
    
    print("\n[5/7] Entrainement du modèle")
    model = train_model(train)

    print("\n[6/7] Construction dataset test")
    test = build_testing_set(sample_all, reg_stats)
    print(f"test - lignes: {test.shape[0]:>7,} colonnes: {list(test.columns)}")   
    print("Taille test:", len(test))
    
    print("\n[7/7] Génération soumission")
    sub = generate_submission(test, model, args.submission, args.clip)
                                                     
    return model, sub
    
# ─────────────────────────────────────────────
# 8. UTILITAIRES
# ─────────────────────────────────────────────

def float_range(min_value: float, max_value: float):
    def validator(value):
        try:
            value = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{value} n'est pas un float valide"
            )

        if not (min_value <= value <= max_value):
            raise argparse.ArgumentTypeError(
                f"{value} doit être compris entre {min_value} et {max_value}"
            )

        return value

    return validator

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="March Mania 2026 - Regular Stats Model")

    parser.add_argument("--submission", type=str, required=True)
    parser.add_argument("--start_season", type=int, default=2003)
    parser.add_argument("--clip", type=float_range(0.0001, 0.0500), default=0.0250)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)

    args = parser.parse_args()

    evaluate(args)
