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
ELO_MOV        = True   # Utiliser la marge de victoire (Margin of Victory)


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


def k_factor(margin, k_base=ELO_K, use_mov=ELO_MOV):
    """
    Facteur K ajusté selon la marge de victoire.
    Une victoire de 30 pts doit peser plus qu'une victoire de 1 pt.
    Formule inspirée de FiveThirtyEight NBA Elo.
    """
    if not use_mov:
        return k_base
    # Racine carrée de la marge, plafonnée pour éviter les valeurs extrêmes
    mov_multiplier = np.sqrt(min(margin, 40)) / np.sqrt(10)
    return k_base * mov_multiplier


def compute_elo(results_df, initial=ELO_INITIAL):
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
                    regressed[team] = rating * 0.70 + initial * 0.30
                elo_current = regressed

            current_season = s

        # Ratings avant le match
        elo_w = get_elo(w)
        elo_l = get_elo(l)

        # Probabilité attendue
        exp_w = expected_score(elo_w, elo_l)

        # Facteur K ajusté sur la marge
        margin = row["WScore"] - row["LScore"]
        k = k_factor(margin)

        # Mise à jour
        elo_current[w] = elo_w + k * (1.0 - exp_w)
        elo_current[l] = elo_l + k * (0.0 - (1.0 - exp_w))

    # Sauvegarde de la dernière saison
    if current_season is not None:
        for team, rating in elo_current.items():
            elo_history[(current_season, team)] = rating

    return elo_history


def build_combined_elo(m_reg, w_reg):
    """
    Calcule l'ELO séparément pour les hommes et les femmes,
    puis fusionne dans un seul dictionnaire.
    Les TeamIDs hommes et femmes ne se chevauchent pas (garanti par Kaggle).
    """
    print("  Calcul ELO hommes...")
    elo_men   = compute_elo(m_reg)
    print(f"  → {len(elo_men):,} entrées (Season, TeamID)")

    print("  Calcul ELO femmes...")
    elo_women = compute_elo(w_reg)
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

def generate_submission(sample_all, elo_dict, model, output_path):
    """
    Prédit P(t1 bat t2) pour tous les matchups du fichier sample.

    Fallback si ELO manquant : 0.5 (neutre)
    Les seeds ne sont pas encore connus → on ne peut pas les utiliser.
    """
    new_ids   = []
    preds     = []
    n_elo     = 0
    n_neutral = 0

    # sample_sub = sample_all[sample_all["ID"].str.startswith(f"{current_season}_")]
    # Prédictions pour les saisons 2022-2025
    sample_sub = sample_all
    
    for i, row in sample_sub.iterrows():
        parts = row["ID"].split("_")
        s  = int(parts[0])
        t1 = int(parts[1])   # garanti t1 < t2 par Kaggle
        t2 = int(parts[2])

        elo1 = elo_dict.get((s, t1))
        elo2 = elo_dict.get((s, t2))

        if elo1 is not None and elo2 is not None:
            diff = np.array([[elo1 - elo2]])
            p    = model.predict_proba(diff)[0][1]
            p    = float(np.clip(p, 0.025, 0.975))
            n_elo += 1
        else:
            # ELO de la saison en cours pas encore disponible
            # → probabilité neutre
            p = 0.5
            n_neutral += 1

        preds.append(p)
        if (i % 50000):
            print("")

    result = sample_sub.copy()
    result["Pred"] = preds
    result.to_csv(output_path, index=False)

    print(f"  ✅ Fichier sauvegardé     : {output_path}")
    print(f"     Total matchups         : {len(result):,}")
    print(f"     → Prédits via ELO      : {n_elo:,}")
    print(f"     → Neutres (ELO manq.)  : {n_neutral:,}")

    return result



# ─────────────────────────────────────────────
# 7. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def evaluate(submission_path, start_season, data_dir):
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
    elo_dict = build_combined_elo(m_reg, w_reg)

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
    sub = generate_submission(sample_all, elo_dict, model, submission_path)

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

    print("\n💡 Notes importantes :")
    print("   - ELO calculé sur saison régulière uniquement (pas de leakage)")
    print("   - Régression inter-saisons : 70% ELO précédent + 30% moyenne")
    print("   - Hommes et femmes traités séparément puis fusionnés")
    print("   - Sélectionner manuellement vos 2 meilleures soumissions !")

    
    return model, sub, metrics

 
# ─────────────────────────────────────────────
# 6. POINT D'ENTRÉE — LIGNE DE COMMANDE
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
        "--data_dir",
        type=str,
        default=DATA_DIR,
        help=f"Répertoire des données Kaggle (défaut: {DATA_DIR})"
    )

    args = parser.parse_args()

    model, subs, metrics = evaluate(
        submission_path = args.submission,
        start_season    = args.start_season,
        data_dir        = args.data_dir,
    )






