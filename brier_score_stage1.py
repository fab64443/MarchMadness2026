"""
=============================================================
  March Machine Learning Mania — Calculateur Brier Score
=============================================================
Calcule le Brier Score d'une soumission stage1 (2022-2025)
comparant aux vrais résultats du tournoi (hommes + femmes).

Brier Score = moyenne de (p_prédit - résultat_réel)²
  → Plus c'est bas, mieux c'est
  → Score parfait = 0.0
  → Prédiction naïve (toujours 0.5) ≈ 0.25

Usage :
  python brier_score_stage1.py \
    --submission  submission_stage1.csv \
    --data_dir    /kaggle/input/march-machine-learning-mania-2026/
=============================================================
"""

import os
import argparse
import numpy as np
import pandas as pd


DATA_DIR = "/home/fabrice/notebooks/kaggle/marchmadness/input"

# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

def load_results(data_dir):
    """
    Charge les vrais résultats du tournoi (hommes + femmes)
    pour la saison demandée.
    """
    m_path = os.path.join(data_dir, "MNCAATourneyCompactResults.csv")
    w_path = os.path.join(data_dir, "WNCAATourneyCompactResults.csv")

    m_results = pd.read_csv(m_path)
    w_results = pd.read_csv(w_path)

    # Filtrer sur la saison cible
    seasons = [2022, 2023, 2024, 2025]
    
    m_season = m_results[m_results["Season"].isin(seasons)].copy()    
    w_season = w_results[w_results["Season"].isin(seasons)].copy()

    print(f"  Matchs hommes trouvés : {len(m_season)}")
    print(f"  Matchs femmes trouvés : {len(w_season)}")

    if len(m_season) == 0 and len(w_season) == 0:
        raise ValueError(
            "Aucun résultat trouvé pour les saisons 2022-2025. "
            f"Vérifiez que la saison existe dans les fichiers."
        )

    return pd.concat([m_season, w_season], ignore_index=True)


def load_submission(submission_path):
    """Charge le fichier de soumission."""
    sub = pd.read_csv(submission_path)

    # Vérification du format
    if "ID" not in sub.columns or "Pred" not in sub.columns:
        raise ValueError("Le fichier de soumission doit contenir les colonnes 'ID' et 'Pred'.")

    # Parsing des IDs → Season, Team1, Team2
    split = sub["ID"].str.split("_", expand=True)
    sub["Season"] = split[0].astype(int)
    sub["Team1"]  = split[1].astype(int)   # toujours le plus petit TeamID
    sub["Team2"]  = split[2].astype(int)

    print(f"  Matchups dans la soumission : {len(sub):,}")
    print(f"  Saisons présentes           : {sorted(sub['Season'].unique().tolist())}")

    seasons = [2022, 2023, 2024, 2025]
    sub = sub[sub["Season"].isin(seasons)]
    print(f"  Saisons retenues            : {sorted(sub['Season'].unique().tolist())}")

    return sub


# ─────────────────────────────────────────────
# 2. CONSTRUCTION DES VRAIS RÉSULTATS
# ─────────────────────────────────────────────

def build_ground_truth(results_df):
    """
    Convertit les résultats bruts en dict :
    où label = 1 si Team1 (plus petit ID) a gagné, 0 sinon.
    Convention : Team1 < Team2 (identique à la soumission Kaggle).
    """
    ground_truth = {}

    for _, row in results_df.iterrows():
        s  = row["Season"]
        w  = row["WTeamID"]
        l  = row["LTeamID"]
        t1 = min(w, l)
        t2 = max(w, l)
        label = 1 if w == t1 else 0
        id = '_'.join([str(s), str(t1), str(t2)])
        ground_truth[id] = label

    print(f"  Matchs réels indexés        : {len(ground_truth)}")
    return ground_truth


# ─────────────────────────────────────────────
# 3. CALCUL DU BRIER SCORE
# ─────────────────────────────────────────────

def compute_brier_score(submission, ground_truth):
    """
    Brier Score = (1/N) * Σ (p_i - y_i)²

    Ne calcule le score QUE sur les matchs qui ont vraiment eu lieu
    (les matchups hypothétiques non joués sont ignorés).
    """
    
    ids = set(ground_truth)   # équivalent à ground_truth.keys()
    sub = submission[submission["ID"].isin(ids)].copy()

    n_submission = submission.shape[0]
    matched = sub.shape[0]
    unmatched = n_submission - matched
    
    scores   = []
    # matched  = 0
    # unmatched = 0

    rows_matched = []

    for _, row in sub.iterrows():
        key = row["ID"]

        if key not in ground_truth:
            continue

        y_true = ground_truth[key]
        y_pred = float(row["Pred"])

        brier  = (y_pred - y_true) ** 2
        scores.append(brier)

        rows_matched.append({
            "Season"  : row["Season"],
            "ID"      : row["ID"],
            "Team1"   : row["Team1"],
            "Team2"   : row["Team2"],
            "y_pred"  : y_pred,
            "y_true"  : y_true,
            "brier"   : brier,
        })

    if matched == 0:
        raise ValueError(
            "Aucun match réel trouvé dans la soumission pour les saisons {season}.\n"
            "Vérifiez que la soumission contient bien les saisons 2022-2025."
        )

    print(f"\n  Matchs joués retrouvés       : {matched}")
    print(f"  Matchups non joués (ignorés) : {unmatched:,}")

    details_df  = pd.DataFrame(rows_matched)

    stats = details_df.groupby("Season").agg(
        brier = ("brier","mean"),
        n     = ("brier","count")
    ).reset_index()

    # stats
    print(f"\nStats:")
    stats.columns = ["Season", "Brier Score", "Rencontres"]
    print(stats.to_string(index=False))

    brier_score = np.mean(scores)
    details_df  = pd.DataFrame(rows_matched)

    return brier_score, details_df

# ─────────────────────────────────────────────
# 4. ANALYSE DÉTAILLÉE
# ─────────────────────────────────────────────

def analyze_results(details_df):
    """Affiche une analyse complète des prédictions."""

    brier_score = np.mean(details_df['brier'])
    
    n       = len(details_df)
    correct = (
        ((details_df["y_pred"] > 0.5) & (details_df["y_true"] == 1)) |
        ((details_df["y_pred"] < 0.5) & (details_df["y_true"] == 0))
    ).sum()
    accuracy = correct / n

    # Baseline naïve : toujours prédire 0.5
    naive_brier = np.mean((0.5 - details_df["y_true"]) ** 2)

    # Amélioration vs baseline
    improvement = (naive_brier - brier_score) / naive_brier * 100
    
    print("\n" + "=" * 55)
    print(f"  RÉSULTATS {sorted(details_df['Season'].unique().tolist())}")
    print("=" * 55)
    print(f"\n  Brier Score          : {brier_score:.6f}")
    print(f"  Baseline naïve (0.5) : {naive_brier:.6f}")
    print(f"  Amélioration         : {improvement:+.2f}%")
    print(f"  Accuracy (>0.5)      : {accuracy:.1%}  ({correct}/{n})")

    # Distribution des erreurs
    print(f"\n  Distribution des erreurs quadratiques :")
    print(f"    Erreur moyenne     : {details_df['brier'].mean():.4f}")
    print(f"    Erreur médiane     : {details_df['brier'].median():.4f}")
    print(f"    Erreur max         : {details_df['brier'].max():.4f}")
    print(f"    Erreur min         : {details_df['brier'].min():.4f}")

    # # Pires prédictions
    # print(f"\n  🔴 Les 5 pires prédictions :")
    # worst = details_df.nlargest(5, "brier")[["Season","ID","y_pred","y_true","brier"]]
    # worst["y_true"] = worst["y_true"].map({1: "Team1 a gagné", 0: "Team2 a gagné"})
    # worst.columns   = ["Season","ID", "Proba prédite", "Résultat réel", "Erreur²"]
    # print(worst.to_string(index=False))

    # # Meilleures prédictions
    # print(f"\n  🟢 Les 5 meilleures prédictions :")
    # best = details_df.nsmallest(5, "brier")[["Season","ID","y_pred","y_true","brier"]]
    # best["y_true"] = best["y_true"].map({1: "Team1 a gagné", 0: "Team2 a gagné"})
    # best.columns   = ["Season","ID", "Proba prédite", "Résultat réel", "Erreur²"]
    # print(best.to_string(index=False))

    # Calibration : le modèle est-il bien calibré ?
    print(f"\n  📐 Calibration (proba prédite vs taux de victoire réel) :")
    bins = [0.0, 0.2, 0.35, 0.45, 0.55, 0.65, 0.80, 1.0]
    details_df["bin"] = pd.cut(details_df["y_pred"], bins=bins)
    calib = details_df.groupby("bin", observed=True).agg(
        n          = ("y_true", "count"),
        mean_pred  = ("y_pred", "mean"),
        actual_win = ("y_true", "mean"),
    ).reset_index()
    calib["gap"] = (calib["mean_pred"] - calib["actual_win"]).abs()
    print(calib[["bin","n","mean_pred","actual_win","gap"]].to_string(index=False))

    return {
        "brier_score" : brier_score,
        "naive_brier" : naive_brier,
        "improvement" : improvement,
        "accuracy"    : accuracy,
        "n_games"     : n,
    }


# ─────────────────────────────────────────────
# 5. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def evaluate(submission_path, details, data_dir):
    print("=" * 55)
    print("  Évaluation Brier Score stage 1 — Saisons 2022-2025")
    print("=" * 55)

    print("\n[1/3] Chargement des données...")
    results_df = load_results(data_dir)    
    submission = load_submission(submission_path)

    print("\n[2/3] Construction de la vérité terrain...")
    ground_truth = build_ground_truth(results_df)

    print("\n[3/3] Calcul du Brier Score...")
    brier_score, details_df = compute_brier_score(submission, ground_truth)

    for season in [2022, 2023, 2024, 2025]:
        df = details_df[details_df["Season"] == season].copy()   
        _ = analyze_results(df)
    
    metrics = analyze_results(details_df)

    # Sauvegarde optionnelle du détail match par match
    if details:
        print("Création du fichier détaillé...")
        detail_path = "brier_details_stage1.csv"
        details_df.to_csv(detail_path, index=False)
        print(f"\n  💾 Détail match par match sauvegardé : {detail_path}")

    return metrics, details_df


# ─────────────────────────────────────────────
# 6. POINT D'ENTRÉE — LIGNE DE COMMANDE
# ─────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calcule le Brier Score d'une soumission March Mania."
    )
    parser.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Chemin vers le fichier de soumission CSV (ex: submission_2025.csv)"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Crée un fichier de résultats détaillés"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR,
        help=f"Répertoire des données Kaggle (défaut: {DATA_DIR})"
    )

    args = parser.parse_args()

    metrics, details = evaluate(
        submission_path = args.submission,
        details         = args.details,
        data_dir        = args.data_dir
    )
