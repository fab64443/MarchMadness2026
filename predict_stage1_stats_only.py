import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings("ignore")

# +
# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR       = "/home/fabrice/notebooks/kaggle/marchmadness/input"

# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

def load_data(data_dir):
    m_reg      = pd.read_csv(os.path.join(data_dir, "MRegularSeasonDetailedResults.csv"))
    w_reg      = pd.read_csv(os.path.join(data_dir, "WRegularSeasonDetailedResults.csv"))
    m_tourney  = pd.read_csv(os.path.join(data_dir, "MNCAATourneyCompactResults.csv"))
    w_tourney  = pd.read_csv(os.path.join(data_dir, "WNCAATourneyCompactResults.csv"))
    sample_all = pd.read_csv(os.path.join(data_dir, "SampleSubmissionStage1.csv"))

    print(f"Hommes  — saison régulière : {len(m_reg):,}")
    print(f"Femmes  — saison régulière : {len(w_reg):,}")
    print(f"Hommes  — tournoi          : {len(m_tourney):,}")
    print(f"Femmes  — tournoi          : {len(w_tourney):,}")
    print(f"Soumission sample          : {len(sample_all):,}")

    return m_reg, w_reg, m_tourney, w_tourney, sample_all

# ─────────────────────────────────────────────
# 2. FEATURES ÉQUIPE SAISON (REGULAR ONLY)
# ─────────────────────────────────────────────

def build_team_stats(reg_df, start_season):
    reg_df = reg_df[reg_df.Season >= start_season].copy()

    winners = pd.DataFrame({
        "Season": reg_df["Season"],
        "TeamID": reg_df["WTeamID"],
        "points_for": reg_df["WScore"],
        "points_against": reg_df["LScore"],
        "win": 1
    })

    losers = pd.DataFrame({
        "Season": reg_df["Season"],
        "TeamID": reg_df["LTeamID"],
        "points_for": reg_df["LScore"],
        "points_against": reg_df["WScore"],
        "win": 0
    })

    games = pd.concat([winners, losers], ignore_index=True)

    stats = (
        games
        .groupby(["Season", "TeamID"])
        .agg(
            games=("win", "count"),
            wins=("win", "sum"),
            points_for=("points_for", "mean"),
            points_against=("points_against", "mean")
        )
        .reset_index()
    )

    stats["win_pct"] = stats["wins"] / stats["games"]
    stats["point_diff"] = stats["points_for"] - stats["points_against"]

    print(f"stats - lignes: {stats.shape[0]:>5,} colonnes: {list(stats.columns)}")   
    
    return stats[[
        "Season",
        "TeamID",
        "win_pct",
        "point_diff",
        "points_for",
        "points_against"
    ]]


def build_training_set(tourney_df, team_stats):

    # ─────────────────────────────────────
    # 1. Construire TeamLow / TeamHigh
    # ─────────────────────────────────────

    df = tourney_df.copy()

    df["TeamLow"]  = df[["WTeamID", "LTeamID"]].min(axis=1)
    df["TeamHigh"] = df[["WTeamID", "LTeamID"]].max(axis=1)

    df["target"] = (df["WTeamID"] == df["TeamLow"]).astype(int)

    # ─────────────────────────────────────
    # 2. Merge stats TeamLow
    # ─────────────────────────────────────

    df = df.merge(
        team_stats,
        left_on=["Season", "TeamLow"],
        right_on=["Season", "TeamID"],
        how="left"
    )

    df = df.rename(columns={
        "win_pct": "win_pct_low",
        "point_diff": "point_diff_low",
        "points_for": "points_for_low",
        "points_against": "points_against_low"
    })

    df = df.drop(columns=["TeamID"])

    # ─────────────────────────────────────
    # 3. Merge stats TeamHigh
    # ─────────────────────────────────────

    df = df.merge(
        team_stats,
        left_on=["Season", "TeamHigh"],
        right_on=["Season", "TeamID"],
        how="left"
    )

    df = df.rename(columns={
        "win_pct": "win_pct_high",
        "point_diff": "point_diff_high",
        "points_for": "points_for_high",
        "points_against": "points_against_high"
    })

    df = df.drop(columns=["TeamID"])

    # ─────────────────────────────────────
    # 4. Calcul différentiel vectorisé
    # ─────────────────────────────────────

    df["win_pct_diff"]        = df["win_pct_low"]        - df["win_pct_high"]
    df["point_diff_diff"]     = df["point_diff_low"]     - df["point_diff_high"]
    df["points_for_diff"]     = df["points_for_low"]     - df["points_for_high"]
    df["points_against_diff"] = df["points_against_low"] - df["points_against_high"]

    # ─────────────────────────────────────
    # 5. Sélection finale + drop NA
    # ─────────────────────────────────────

    final_cols = [
        "win_pct_diff",
        "point_diff_diff",
        "points_for_diff",
        "points_against_diff",
        "target"
    ]

    df = df[final_cols].dropna()

    print(f"train_set - lignes: {df.shape[0]:>5,} colonnes: {list(df.columns)}")

    return df
    

# ─────────────────────────────────────────────
# 4. ENTRAÎNEMENT
# ─────────────────────────────────────────────

def train_model(train_df):

    X = train_df.drop(columns=["target"])
    y = train_df["target"]

    model = LogisticRegression(max_iter=500)
    model.fit(X, y)

    preds = model.predict_proba(X)[:,1]
    print("LogLoss train:", log_loss(y, preds))

    return model

    
# ─────────────────────────────────────────────
# 5. GÉNÉRATION SOUMISSION
# ─────────────────────────────────────────────

def generate_submission_OLD(sample_all, team_stats, model, submission_path, clip):

    rows = []

    for _, row in tqdm(sample_all.iterrows(), total=len(sample_all), mininterval=2, dynamic_ncols=True):        

        parts = row.ID.split("_")
        season = int(parts[0])
        team1  = int(parts[1])   # garanti t1 < t2 par Kaggle
        team2  = int(parts[2])        
        
        stats1 = team_stats[
            (team_stats.Season == season) &
            (team_stats.TeamID == team1)
        ]

        stats2 = team_stats[
            (team_stats.Season == season) &
            (team_stats.TeamID == team2)
        ]

        if len(stats1)==0 or len(stats2)==0:
            pred = 0.5
        else:
            diff = stats1.iloc[0][["win_pct","point_diff","points_for","points_against"]] \
                 - stats2.iloc[0][["win_pct","point_diff","points_for","points_against"]]

            pred = model.predict_proba([diff.values])[0][1]

        pred = np.clip(pred, clip, 1-clip)
        rows.append(pred)

    # sub = sample_all.copy()
    sample_all["Pred"] = rows
    sample_all.to_csv(submission_path, index=False)

    print(f"Soumission sauvegardée : {submission_path}")

    return sample_all


def generate_submission(sample_all, team_stats, model, submission_path, clip):

    df = sample_all.copy()

    # ─────────────────────────────────────
    # 1. Split ID proprement (vectorisé)
    # ─────────────────────────────────────
    id_split = df["ID"].str.split("_", expand=True)

    df["Season"] = id_split[0].astype(int)
    df["Team1"]  = id_split[1].astype(int)  # déjà low ID
    df["Team2"]  = id_split[2].astype(int)  # déjà high ID

    # ─────────────────────────────────────
    # 2. Merge Team1 stats
    # ─────────────────────────────────────
    df = df.merge(
        team_stats,
        left_on=["Season", "Team1"],
        right_on=["Season", "TeamID"],
        how="left"
    )

    df = df.rename(columns={
        "win_pct": "win_pct_1",
        "point_diff": "point_diff_1",
        "points_for": "points_for_1",
        "points_against": "points_against_1"
    }).drop(columns=["TeamID"])

    # ─────────────────────────────────────
    # 3. Merge Team2 stats
    # ─────────────────────────────────────
    df = df.merge(
        team_stats,
        left_on=["Season", "Team2"],
        right_on=["Season", "TeamID"],
        how="left"
    )

    df = df.rename(columns={
        "win_pct": "win_pct_2",
        "point_diff": "point_diff_2",
        "points_for": "points_for_2",
        "points_against": "points_against_2"
    }).drop(columns=["TeamID"])

    # ─────────────────────────────────────
    # 4. Calcul différentiel vectorisé
    # ─────────────────────────────────────
    df["win_pct_diff"]        = df["win_pct_1"]        - df["win_pct_2"]
    df["point_diff_diff"]     = df["point_diff_1"]     - df["point_diff_2"]
    df["points_for_diff"]     = df["points_for_1"]     - df["points_for_2"]
    df["points_against_diff"] = df["points_against_1"] - df["points_against_2"]

    feature_cols = [
        "win_pct_diff",
        "point_diff_diff",
        "points_for_diff",
        "points_against_diff"
    ]

    X = df[feature_cols]

    # ─────────────────────────────────────
    # 5. Prédiction batch
    # ─────────────────────────────────────
    preds = model.predict_proba(X)[:, 1]

    # Remplacer NaN (équipes manquantes) par 0.5
    preds = np.where(np.isnan(preds), 0.5, preds)

    preds = np.clip(preds, clip, 1 - clip)

    df["Pred"] = preds

    # ─────────────────────────────────────
    # 6. Export
    # ─────────────────────────────────────
    submission = df[["ID", "Pred"]]
    submission.to_csv(submission_path, index=False)

    print(f"Soumission sauvegardée : {submission_path}")

    return submission

    
# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def evaluate(submission_path, start_season, clip, data_dir):

    print("="*60)
    print("March Mania 2026 — Stats Regular Season Only")
    print("="*60)

    print("\n[1/5] Chargement")
    m_reg, w_reg, m_tourney, w_tourney, sample_all = load_data(data_dir)

    print("\n[2/5] Construction des stats équipes")
    m_stats = build_team_stats(m_reg, start_season)
    w_stats = build_team_stats(w_reg, start_season)
    team_stats = pd.concat([m_stats, w_stats], ignore_index=True)

    print("\n[3/5] Construction dataset train")
    train_m = build_training_set(m_tourney, team_stats)
    train_w = build_training_set(w_tourney, team_stats)
    train_df = pd.concat([train_m, train_w], ignore_index=True)

    print("Taille train:", len(train_df))
    # print(train_df[:4])

    print("\n[4/5] Entrainement du modèle")
    model = train_model(train_df)

    print("\n[5/5] Génération soumission")
    sub = generate_submission(sample_all, team_stats, model, submission_path, clip)

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

    evaluate(
        submission_path = args.submission,
        start_season    = args.start_season,
        clip            = args.clip,
        data_dir        = args.data_dir,
    )

    