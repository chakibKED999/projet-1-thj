"""
Module : Tests Statistiques et Métriques d'Évaluation
======================================================
Fournit l'ensemble des outils statistiques rigoureux pour la Phase 4 :
  1. Test de Wilcoxon des rangs signés apparié BILATÉRAL (two-sided).
  2. Procédure d'ajustement séquentiel de Holm (Holm-Bonferroni) pour tests multiples.
  3. Mesure de taille d'effet A12 de Vargha-Delaney (non paramétrique).
  4. Calcul de l'AUC de la courbe d'erreur (Area Under Curve).
  5. Calcul du nombre d'évaluations pour atteindre la cible (AES).
  6. Détermination du verdict Gagne / Égalité / Perd (Win / Tie / Loss).
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy import stats


def wilcoxon_signed_rank_test(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """
    Réalise le test des rangs signés de Wilcoxon apparié et BILATÉRAL.

    Paramètres
    ----------
    x : np.ndarray
        Résultats de l'algorithme 1 (ex. GA-Nash), de taille N.
    y : np.ndarray
        Résultats de l'algorithme 2 (ex. GA standard), de taille N (apparié par graine).

    Retourne
    --------
    statistic : float
        Statistique de Wilcoxon W.
    p_value : float
        Valeur-p bilatérale (two-sided).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Si tous les scores sont strictement identiques, la différence est nulle
    diff = x - y
    if np.all(diff == 0.0) or np.all(np.abs(diff) < 1e-12):
        return 0.0, 1.0

    try:
        res = stats.wilcoxon(x, y, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        # Repli si le nombre d'éléments non nuls est insuffisant
        return 0.0, 1.0


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Applique la correction de Holm-Bonferroni à une liste de valeurs-p.

    Paramètres
    ----------
    p_values : List[float]
        Liste des valeurs-p brutes.

    Retourne
    --------
    List[float]
        Liste des valeurs-p ajustées ordonnées dans l'ordre originel.
    """
    m = len(p_values)
    if m == 0:
        return []

    sorted_indices = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=float)

    # Calcul des p-valeurs ajustées
    for rank, idx in enumerate(sorted_indices):
        mult = m - rank
        raw_adj = p_values[idx] * mult
        adjusted[idx] = raw_adj

    # Assurer la monotonie croissante
    running_max = 0.0
    for idx in sorted_indices:
        running_max = max(running_max, adjusted[idx])
        adjusted[idx] = min(1.0, running_max)

    return adjusted.tolist()


def vargha_delaney_a12(
    x: np.ndarray,
    y: np.ndarray,
    minimize: bool = True,
) -> float:
    """
    Calcule la taille d'effet non paramétrique A12 de Vargha et Delaney (2000).

    Dans le contexte d'une minimisation :
      A12 = ( nombre_de_cas(x < y) + 0.5 * nombre_de_cas(x == y) ) / N

    Interprétation :
      - A12 = 0.5 : performance équivalente entre x et y.
      - A12 > 0.5 : l'algorithme x est supérieur à y (plus petites valeurs).
      - A12 < 0.5 : l'algorithme y est supérieur à x.
      Seuils conventionnels : faible (>=0.56), moyen (>=0.64), fort (>=0.71).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n == 0:
        return 0.5

    if minimize:
        wins = np.sum(x < y)
        ties = np.sum(np.isclose(x, y, atol=1e-12))
    else:
        wins = np.sum(x > y)
        ties = np.sum(np.isclose(x, y, atol=1e-12))

    return float((wins + 0.5 * ties) / n)


def win_tie_loss(
    x: np.ndarray,
    y: np.ndarray,
    p_adj: float,
    alpha: float = 0.05,
    minimize: bool = True,
) -> str:
    """
    Détermine le statut comparatif (Gagne / Égalité / Perd) à partir de la p-valeur ajustée
    et de la taille d'effet A12.
    """
    a12 = vargha_delaney_a12(x, y, minimize=minimize)
    if p_adj < alpha:
        if a12 > 0.5:
            return "Win"
        elif a12 < 0.5:
            return "Loss"
    return "Tie"


def compute_auc_error(
    best_history: List[float],
    optimum: float,
) -> float:
    """
    Calcule l'aire sous la courbe d'erreur (AUC) au fil des générations
    par intégration trapézoïdale : AUC = int_0^G |f_best(g) - optimum| dg.
    """
    errors = np.abs(np.asarray(best_history, dtype=float) - optimum)
    if len(errors) <= 1:
        return float(errors[0]) if len(errors) == 1 else 0.0
    # Règle des trapèzes sur l'axe des générations (dx = 1)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(errors))
    return float(np.trapz(errors))


def compute_evaluations_to_target(
    best_history: List[float],
    target_threshold: float,
    pop_size: int,
    max_evals: int,
) -> int:
    """
    Calcule le nombre d'évaluations pour atteindre pour la première fois le seuil cible.
    Convention : si non atteint, retourne max_evals.
    """
    for g, val in enumerate(best_history):
        if val <= target_threshold:
            # Évaluations consommées jusqu'à la fin de la génération g
            return (g + 1) * pop_size
    return max_evals
