import pandas as pd

DATA_DIR       = "/home/fabrice/notebooks/kaggle/marchmadness/input"

# Load this dataset
elo_m = pd.read_csv(DATA_DIR + "/ncaa_elo_mens_2003_2026.csv")
elo_w = pd.read_csv(DATA_DIR + "/ncaa_elo_womens_2003_2026.csv")

# Load competition sample submission
sample = pd.read_csv(DATA_DIR + "/SampleSubmissionStage1.csv")

# Parse matchup IDs
sample[["Season", "Team1", "Team2"]] = (
    sample["ID"].str.split("_", expand=True).astype(int)
)

# Join Elo features for Team 1
sample = sample.merge(
    elo_m[["Season","TeamID","FinalElo","EloVsPeakDiff"]].rename(
        columns={"TeamID":"Team1","FinalElo":"Elo1","EloVsPeakDiff":"PeakDiff1"}
    ), on=["Season","Team1"], how="left"
)

# Join Elo features for Team 2
sample = sample.merge(
    elo_m[["Season","TeamID","FinalElo","EloVsPeakDiff"]].rename(
        columns={"TeamID":"Team2","FinalElo":"Elo2","EloVsPeakDiff":"PeakDiff2"}
    ), on=["Season","Team2"], how="left"
)

# Simple Elo win probability
def elo_prob(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

sample["Pred"] = elo_prob(sample["Elo1"], sample["Elo2"])
sample=sample[['ID','Pred']]

submission_path = "stage1_elo_peak.csv"
sample.to_csv(submission_path, index=False)

print(f"Soumission sauvegardée : {submission_path}")
