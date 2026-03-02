"""
=============================================================
  March Machine Learning Mania 2026 — Modèle ELO Only
=============================================================
Feature unique : ELO rating calculé sur les CompactResults
                 (hommes + femmes)

Format soumission 2026 :
  - Hommes + Femmes dans un seul fichier
  - Tous les matchups possibles (pas seulement les qualifiés)
  - P(team avec le plus petit TeamID gagne)

Sources de données utilisées :
  - MRegularSeasonCompactResults.csv
  - WRegularSeasonCompactResults.csv
  - MNCAATourneyCompactResults.csv
  - WNCAATourneyCompactResults.csv
  - MSampleSubmission.csv
=============================================================
"""

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

# Hyperparamètres ELO
ELO_INITIAL    = 1500   # Rating de départ pour toute équipe
ELO_K          = 20     # Sensibilité aux résultats récents

# +
# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────

def load_data(data_dir=DATA_DIR):
    m_reg      = pd.read_csv(os.path.join(data_dir, "MRegularSeasonCompactResults.csv"))
    w_reg      = pd.read_csv(os.path.join(data_dir, "WRegularSeasonCompactResults.csv"))
    m_tourney  = pd.read_csv(os.path.join(data_dir, "MNCAATourneyCompactResults.csv"))
    w_tourney  = pd.read_csv(os.path.join(data_dir, "WNCAATourneyCompactResults.csv"))
    sample_all = pd.read_csv(os.path.join(data_dir, "SampleSubmissionStage1.csv"))
    
    print(f"  Hommes  — saison régulière : {len(m_reg):,} matchs")
    print(f"  Femmes  — saison régulière : {len(w_reg):,} matchs")
    print(f"  Hommes  — tournoi          : {len(m_tourney):,} matchs")
    print(f"  Femmes  — tournoi          : {len(w_tourney):,} matchs")
    print(f"  Soumission sample          : {len(sample_all):,} matchups")

    return m_reg, w_reg, m_tourney, w_tourney, sample_all


# +
# ─────────────────────────────────────────────
# 2. CALCUL DU RATING ELO
# ─────────────────────────────────────────────

def expected_score(ra, rb):
    """Probabilité que l'équipe A batte l'équipe B selon leurs ratings."""
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def k_factor(margin, k_base=ELO_K):
    """
    Facteur K ajusté selon la marge de victoire.
    Une victoire de 30 pts doit peser plus qu'une victoire de 1 pt.
    Formule inspirée de FiveThirtyEight NBA Elo.
    """
    # Racine carrée de la marge, plafonnée pour éviter les valeurs extrêmes
    mov_multiplier = np.sqrt(min(margin, 40)) / np.sqrt(10)
    return k_base * mov_multiplier


def k_factor_advanced(margin, elo_diff, k_base=ELO_K):
    """
    K-factor qui tient compte :
    - de la marge de victoire (déjà fait)
    - de l'écart ELO avant le match
    
    L'idée : battre un adversaire bien classé avec 20 pts d'écart
    est plus significatif que battre un faible avec le même écart.
    Formule FiveThirtyEight NFL adaptée au basket.
    """
    mov_mult  = np.log(max(margin, 1) + 1)
    # Correction : une grosse victoire contre un faible
    # est moins méritoire qu'une grosse victoire contre un fort
    elo_corr  = 2.2 / (abs(elo_diff) * 0.001 + 2.2)
    return k_base * mov_mult * elo_corr
    
    
def compute_elo_OLD(results_df, inter_season, initial=ELO_INITIAL):
    """
    Calcule les ratings ELO pour chaque équipe et chaque saison.

    Stratégie de réinitialisation inter-saisons :
      - On ne repart pas de zéro chaque saison
      - On régresse vers la moyenne (1500) de 30%
        pour tenir compte du renouvellement des effectifs

    Retourne un dict {(Season, TeamID): elo_final}
    """
    elo_current = {}   # ELO courant (mis à jour match par match)
    elo_history = {}   # ELO final enregistré à la fin de chaque saison

    def get_elo(team):
        return elo_current.get(team, initial)

    current_season = None

    # Tri chronologique strict
    df = results_df.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    for _, row in df.iterrows():
        s = row["Season"]
        w = row["WTeamID"]
        l = row["LTeamID"]

        # Détection du changement de saison → régression vers la moyenne
        if s != current_season:
            if current_season is not None:
                # Sauvegarde des ELO de fin de saison précédente
                for team, rating in elo_current.items():
                    elo_history[(current_season, team)] = rating

                # Régression vers 1500 (30%) pour la nouvelle saison
                regressed = {}
                for team, rating in elo_current.items():
                    regressed[team] = rating * (1-inter_season) + initial * inter_season
                    # regressed[team] = rating * 0.70 + initial * 0.30
                elo_current = regressed

            current_season = s

        # Ratings avant le match
        elo_w = get_elo(w)
        elo_l = get_elo(l)

        # Probabilité attendue
        exp_w = expected_score(elo_w, elo_l)

        # Facteur K ajusté sur la marge
        margin = row["WScore"] - row["LScore"]
        # k = k_factor(margin)
        k = k_factor_advanced(margin, elo_w-elo_l)

        # Mise à jour
        elo_current[w] = elo_w + k * (1.0 - exp_w)
        elo_current[l] = elo_l + k * (0.0 - (1.0 - exp_w))

    # Sauvegarde de la dernière saison
    if current_season is not None:
        for team, rating in elo_current.items():
            elo_history[(current_season, team)] = rating

    return elo_history


def compute_elo(results_df, inter_season, initial=1500):

    elo_current = {}
    elo_history = {}

    current_season = None

    df = results_df.sort_values(["Season", "DayNum"])

    for row in df.itertuples(index=False):

        s = row.Season
        w = row.WTeamID
        l = row.LTeamID

        # ─────────────────────────────
        # Changement de saison
        # ─────────────────────────────
        if s != current_season:

            if current_season is not None:

                # Sauvegarde rapide
                elo_history.update({
                    (current_season, team): rating
                    for team, rating in elo_current.items()
                })

                # Régression vectorisée
                elo_current = {
                    team: rating * (1 - inter_season) + initial * inter_season
                    for team, rating in elo_current.items()
                }

            current_season = s

        # ─────────────────────────────
        # Récupération ELO
        # ─────────────────────────────
        elo_w = elo_current.get(w, initial)
        elo_l = elo_current.get(l, initial)

        # ─────────────────────────────
        # Update
        # ─────────────────────────────
        exp_w = 1.0 / (1.0 + 10 ** ((elo_l - elo_w) / 400))

        margin = row.WScore - row.LScore
        k = k_factor_advanced(margin, elo_w - elo_l)

        elo_current[w] = elo_w + k * (1.0 - exp_w)
        elo_current[l] = elo_l - k * (1.0 - exp_w)

    # ─────────────────────────────
    # Dernière saison
    # ─────────────────────────────
    if current_season is not None:
        elo_history.update({
            (current_season, team): rating
            for team, rating in elo_current.items()
        })

    return elo_history

    
def build_combined_elo(m_reg, w_reg, inter_season):
    """
    Calcule l'ELO séparément pour les hommes et les femmes,
    puis fusionne dans un seul dictionnaire.
    Les TeamIDs hommes et femmes ne se chevauchent pas (garanti par Kaggle).
    """
    print("  Calcul ELO hommes...")
    elo_men   = compute_elo(m_reg, inter_season)
    print(f"  → {len(elo_men):,} entrées (Season, TeamID)")

    print("  Calcul ELO femmes...")
    elo_women = compute_elo(w_reg, inter_season)
    print(f"  → {len(elo_women):,} entrées (Season, TeamID)")

    return {**elo_men, **elo_women}

# +
# ─────────────────────────────────────────────
# 3. CONSTRUCTION DU DATASET D'ENTRAÎNEMENT
# ─────────────────────────────────────────────

def build_training_data(tourney_res, elo_dict, label="", start_season=2003):
    """
    Chaque ligne = un match historique du tournoi.
    Feature : elo_diff = ELO(team1) - ELO(team2)
              calculé avec les ELO de FIN de saison régulière

    Convention Kaggle : team1 = équipe avec le plus petit TeamID
    """
    records = []
    skipped = 0

    for _, row in tourney_res[tourney_res["Season"] >= start_season].iterrows():
        s  = row["Season"]
        t1 = min(row["WTeamID"], row["LTeamID"])
        t2 = max(row["WTeamID"], row["LTeamID"])

        elo1 = elo_dict.get((s, t1))
        elo2 = elo_dict.get((s, t2))

        # On ignore les matchs sans ELO disponible
        if elo1 is None or elo2 is None:
            skipped += 1
            continue

        label_val = 1 if row["WTeamID"] == t1 else 0
        records.append({
            "Season"   : s,
            "Team1"    : t1,
            "Team2"    : t2,
            "elo1"     : elo1,
            "elo2"     : elo2,
            "elo_diff" : elo1 - elo2,
            "label"    : label_val
        })

    if skipped:
        print(f"  ⚠️  {skipped} matchs ignorés (ELO manquant) {label}")

    return pd.DataFrame(records)



# +
# ─────────────────────────────────────────────
# 4. ENTRAÎNEMENT DU MODÈLE
# ─────────────────────────────────────────────

def train_model(train_df):
    """
    Régression logistique sur elo_diff.
    Analytiquement, c'est la forme fonctionnelle correcte :
    la formule ELO elle-même est une sigmoïde sur la différence de rating.
    Le modèle apprend simplement le meilleur scaling.
    """
    X = train_df[["elo_diff"]].values
    y = train_df["label"].values

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X, y)

    # Validation croisée 5-fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_log_loss")
    print(f"  CV Log-Loss : {-scores.mean():.4f} ± {scores.std():.4f}")

    # Coefficient appris vs théorie ELO pure
    coef = model.coef_[0][0]
    # La formule ELO implique un coefficient de ln(10)/400 ≈ 0.00575
    elo_theoretical = np.log(10) / 400
    print(f"  Coefficient appris    : {coef:.6f}")
    print(f"  Coefficient ELO pur   : {elo_theoretical:.6f}")
    print(f"  Ratio (calibration)   : {coef / elo_theoretical:.3f}")

    return model



# +
# ─────────────────────────────────────────────
# 5. ANALYSE DES PRÉDICTIONS
# ─────────────────────────────────────────────

def analyze_predictions(model, train_df):
    """Montre les probabilités prédites pour différents écarts ELO."""
    print("\n  Probabilités selon l'écart ELO :")
    print(f"  {'ELO diff':<12} {'P(favori)':<15} {'Interprétation'}")
    print("  " + "-" * 50)

    scenarios = [
        ( 600, "Favori écrasant (ex: seed 1 vs 16)"),
        ( 300, "Nette supériorité (ex: seed 2 vs 10)"),
        ( 150, "Léger avantage   (ex: seed 3 vs 6) "),
        (  50, "Match serré      (ex: seed 4 vs 5) "),
        (   0, "Équilibre parfait                  "),
        ( -50, "Léger désavantage                  "),
        (-150, "Nette infériorité                  "),
    ]

    for diff, label in scenarios:
        p = model.predict_proba([[diff]])[0][1]
        print(f"  {diff:>+8}      {p:.3f}          {label}")


# +
# ─────────────────────────────────────────────
# 6. GÉNÉRATION DE LA SOUMISSION
# ─────────────────────────────────────────────

def generate_submission(sample_all, elo_dict, model, output_path, clip):
    """
    Version vectorisée.
    Prédit P(t1 bat t2) pour tous les matchups.
    Fallback = 0.5 si ELO manquant.
    """

    df = sample_all.copy()

    # ─────────────────────────────────────
    # 1. Split ID vectorisé
    # ─────────────────────────────────────
    id_split = df["ID"].str.split("_", expand=True)

    df["Season"] = id_split[0].astype(int)
    df["Team1"]  = id_split[1].astype(int)  # t1 < t2 garanti
    df["Team2"]  = id_split[2].astype(int)

    # ─────────────────────────────────────
    # 2. Transformer elo_dict → DataFrame
    # ─────────────────────────────────────
    elo_df = (
        pd.DataFrame(
            [(s, t, e) for (s, t), e in elo_dict.items()],
            columns=["Season", "TeamID", "elo"]
        )
    )

    # ─────────────────────────────────────
    # 3. Merge ELO Team1
    # ─────────────────────────────────────
    df = df.merge(
        elo_df,
        left_on=["Season", "Team1"],
        right_on=["Season", "TeamID"],
        how="left"
    ).rename(columns={"elo": "elo1"}).drop(columns=["TeamID"])

    # ─────────────────────────────────────
    # 4. Merge ELO Team2
    # ─────────────────────────────────────
    df = df.merge(
        elo_df,
        left_on=["Season", "Team2"],
        right_on=["Season", "TeamID"],
        how="left"
    ).rename(columns={"elo": "elo2"}).drop(columns=["TeamID"])

    # ─────────────────────────────────────
    # 5. Calcul diff ELO
    # ─────────────────────────────────────
    df["elo_diff"] = df["elo1"] - df["elo2"]

    # Matchups valides (ELO dispo pour les deux équipes)
    mask_valid = df["elo_diff"].notna()

    n_elo     = mask_valid.sum()
    n_neutral = len(df) - n_elo

    # ─────────────────────────────────────
    # 6. Prédiction batch
    # ─────────────────────────────────────
    preds = np.full(len(df), 0.5)

    if n_elo > 0:
        X_valid = df.loc[mask_valid, ["elo_diff"]].values
        p_valid = model.predict_proba(X_valid)[:, 1]
        p_valid = np.clip(p_valid, clip, 1 - clip)
        preds[mask_valid] = p_valid

    df["Pred"] = preds

    # ─────────────────────────────────────
    # 7. Export
    # ─────────────────────────────────────
    result = df[["ID", "Pred"]]
    result.to_csv(output_path, index=False)

    print(f"  ✅ Fichier sauvegardé     : {output_path}")
    print(f"     Total matchups         : {len(result):,}")
    print(f"     → Prédits via ELO      : {n_elo:,}")
    print(f"     → Neutres (ELO manq.)  : {n_neutral:,}")

    return result

    

# ─────────────────────────────────────────────
# 7. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def evaluate(submission_path, start_season, inter_season, clip, data_dir):
    print("=" * 55)
    print("  March Mania 2026 — Modèle ELO Only")
    print("=" * 55)
    
    # ── Chargement ──────────────────────────────
    print("\n[1/4] Chargement des données...")
    m_reg, w_reg, m_tourney, w_tourney, sample_all = load_data(data_dir)

    # ── Calcul ELO ──────────────────────────────
    print("\n[2/4] Calcul des ratings ELO...")
    # On calcule l'ELO sur la saison régulière uniquement
    # (le tournoi ne doit PAS alimenter l'ELO pour éviter le data leakage)
    elo_dict = build_combined_elo(m_reg, w_reg, inter_season)

    # ── Entraînement ────────────────────────────
    print("\n[3/4] Entraînement du modèle...")
    # On utilise les matchs de tournoi historiques comme labels
    # (hommes + femmes combinés pour plus de robustesse)
    m_train  = build_training_data(m_tourney, elo_dict, label="(hommes)", start_season=start_season)
    w_train  = build_training_data(w_tourney, elo_dict, label="(femmes)", start_season=start_season)
    train_df = pd.concat([m_train, w_train], ignore_index=True)

    print(f"  Dataset : {len(train_df):,} matchs")
    print(f"  Saisons : {train_df['Season'].min()}–{train_df['Season'].max()}")
    print(f"  ELO diff — min: {train_df['elo_diff'].min():.0f} "
          f"/ max: {train_df['elo_diff'].max():.0f} "
          f"/ moy: {train_df['elo_diff'].mean():.0f}")

    model = train_model(train_df)
    analyze_predictions(model, train_df)

    # ── Soumission ──────────────────────────────
    print("\n[4/4] Génération de la soumission...")
    sub = generate_submission(sample_all, elo_dict, model, submission_path, clip)

    metrics = {
        "mean"    : sub['Pred'].mean(),
        "median"  : sub['Pred'].median(),
        "std"     : sub['Pred'].std(),
        "min"     : sub['Pred'].min(),
        "max"     : sub['Pred'].max()
    }

    print("\n📊 Statistiques de la soumission :")
    print(f"   Proba moyenne  : {metrics['mean']:.4f}  (attendu ≈ 0.500)")
    print(f"   Proba médiane  : {metrics['median']:.4f}")
    print(f"   Écart-type     : {metrics['std']:.4f}")
    print(f"   Min / Max      : {metrics['min']:.4f} / {metrics['max']:.4f}")

    return model, sub, metrics


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
# 9. POINT D'ENTRÉE — LIGNE DE COMMANDE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Génère les soumissions 2022-2025 pour March Mania."
    )
    parser.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Chemin vers le fichier de soumission CSV (ex: submission_2025.csv)"
    )
    parser.add_argument(
        "--start_season",
        type=int,
        default=2003,
        help="Année de départ pour calculer les statistiques (défaut: 2003)"
    )
    parser.add_argument(
        "--inter_season",
        type=float_range(0.0, 1.0),
        default=0.3,
        help="Regression inter-saison (défaut: 0.3)"
    )
    parser.add_argument(
        "--clip",
        type=float_range(0.0001, 0.05),
        default=0.025,
        help="Clipping des prédictions (défaut: 0.025)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR,
        help=f"Répertoire des données Kaggle (défaut: {DATA_DIR})"
    )

    args = parser.parse_args()

    model, subs, metrics = evaluate(
        submission_path = args.submission,
        start_season    = args.start_season,
        inter_season    = args.inter_season,
        clip            = args.clip,
        data_dir        = args.data_dir,
    )


