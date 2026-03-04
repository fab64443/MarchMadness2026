import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
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

    print(f"teams - lignes: {teams.shape[0]:>7,} colonnes: {list(teams.columns)}")   
    return teams


# ─────────────────────────────────────────────
# 3a. STATS ÉQUIPE SAISON
# ─────────────────────────────────────────────

def build_team_stats_global(teams):

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
# 3b. STATS ÉQUIPE SAISON
# ─────────────────────────────────────────────

def build_team_stats_cumul(teams):

    # add 5 virtual matches
    start_year = teams.groupby("TeamID", as_index=False)["Season"].min()
    virt = start_year.loc[start_year.index.repeat(5)].copy()

    virt_stats = { "DayNum": 0, "ScoreOff": 70, "ScoreDef": 70, "NumOT": 0, "FGM": 25, "FGA": 58, "FGM3": 7, "FGA3": 22, 
                   "FTM": 13, "FTA": 18, "ORd": 10, "DRd": 25, "Ast": 14, "TOr": 13, "Stl": 6, "Blk": 4, "PF": 17 }
    for k, v in virt_stats.items():
        virt[k] = v

    virt["Margin"] = virt["ScoreOff"] - virt["ScoreDef"]
    virt["Poss"] = virt["FGA"] - virt["ORd"] + virt["TOr"] + 0.44 * virt["FTA"]
    virt["OffRate"] = virt["ScoreOff"] / virt["Poss"]
    virt["DefRate"] = virt["ScoreDef"] / virt["Poss"]
    virt["NetRate"] = virt["OffRate"] - virt["DefRate"]
    virt["eFGPct"] = (virt["FGM"] + 0.5 * virt["FGM3"]) / virt["FGA"]
    virt["TSPct"] = virt["ScoreOff"] /  (2 * (virt["FGM"] + 0.44 * virt["FTA"]))
    virt["FGA3Rate"] = virt["FGA3"] / virt["FGA"]
    virt["FTRate"] = virt["FTA"] / virt["FGA"]
    virt["TOVRate"] = virt["TOr"] / virt["Poss"]
    virt["ORPct"] = virt["ORd"] / (virt["ORd"] + virt["DRd"])
    virt["DRPct"] = virt["DRd"] / (virt["ORd"] + virt["DRd"])

    # add
    games = pd.concat([teams, virt], ignore_index=True)

    games = games.sort_values(["Season", "TeamID", "DayNum"])
    group = games.groupby(["TeamID"])

    stats_cols = ['ScoreOff', 'ScoreDef', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'ORd', 'DRd', 'Ast', 'TOr', 
                  'Stl', 'Blk', 'PF', 'Win', 'Margin', 'Poss', 'OffRate', 'DefRate', 'NetRate', 'eFGPct', 'TSPct', 
                  'FGA3Rate', 'FTRate', 'TOVRate', 'ORPct', 'DRPct']
    
    games["Games"] = group.cumcount()
    for col in stats_cols:
        games[f"{col}_cum"] = group[col].cumsum().shift()
        games[f"{col}"] = games[f"{col}_cum"] / games["Games"]
        games = games.drop(columns=[f"{col}_cum"])

    # remove virtual matches
    games = games[games.DayNum>0]
    
    return games

    
# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def evaluate(args):

    print("="*60)
    print("March Mania 2026 — Stats Regular Season")
    print("="*60)

    print("\n[1/5] Chargement")
    m_reg, w_reg, m_tourney, w_tourney, sample_all = load_data(args.data_dir)

    print("\n[2/5] Construction des infos équipes")
    m_info = build_team_infos(m_reg, args.start_season)
    w_info = build_team_infos(w_reg, args.start_season)
    teams = pd.concat([m_info, w_info], ignore_index=True)
    print(f"teams - lignes: {teams.shape[0]:>7,} colonnes: {list(teams.columns)}")   
    
    print("\n[3/5] Construction des stats équipes")
    team_stats_global = build_team_stats_global(teams)
    team_stats_cumul = build_team_stats_cumul(teams)
    print(f"{team_stats_cumul.head(5)}")
    print(f"{team_stats_cumul.tail(5)}")
    
    # print("\n[3/5] Construction dataset train")
    # train_w = build_training_set(w_tourney, team_stats)
    # train_df = pd.concat([train_m, train_w], ignore_index=True)

    # print("Taille train:", len(train_df))
    # # print(train_df[:4])

    # print("\n[4/5] Entrainement du modèle")
    # model = train_model(train_df)

    # print("\n[5/5] Génération soumission")
    # sub = generate_submission(sample_all, team_stats, model, submission_path, clip)

    # return model, sub

    
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
    # evaluate(
    #     submission_path = args.submission,
    #     start_season    = args.start_season,
    #     clip            = args.clip,
    #     data_dir        = args.data_dir,
    # )
