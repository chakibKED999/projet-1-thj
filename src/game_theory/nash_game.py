"""
Module : Composant Théorie des Jeux (Game Theory Component)
============================================================
Modélisation d'un jeu stratégique à deux joueurs entre opérateurs génétiques :
  - Joueur 1 (J1) : Opérateur de Croisement (sélectionne le taux cr ∈ S₁)
  - Joueur 2 (J2) : Opérateur de Mutation (sélectionne le taux mr ∈ S₂)

Nature du jeu :
  JEU À INTÉRÊTS COMMUNS (Coordination pure) : u₁ = u₂ = U.
  Les deux opérateurs coopèrent vers l'objectif partagé de maximiser
  la qualité des enfants produits dans la métaheuristique.

Formulation théorique :
  - Récompense d'un enfant :
      r = clip((f_moy_parents - f_enfant) / (f_max_pop - f_min_pop + ε), -1, 1)
      u = (r + 1) / 2 ∈ [0, 1]
  - Matrice des gains U : mise à jour EMA (paramètre β) pour chaque case (i, j)
    ayant produit au moins un enfant au cours de la génération.
  - Normalisation min-max de U avant le calcul des gains espérés :
      U_norm = (U - min(U)) / (max(U) - min(U) + ε)
  - Dynamique de réplication discrète (Replicator Dynamics) :
      v₁ = U_norm · σ₂
      v₂ = U_norm^T · σ₁
      σ_k ← σ_k · exp(η · (v_k - σ_k · v_k)), normalisé
      Plancher d'exploration ε_min : σ_k ← ε_min + (1 - K·ε_min) · σ_k
  - Diagnostics :
      * Équilibres de Nash purs (meilleures réponses mutuelles, Chapitre 3)
      * Nash Gap : (max v₁ - σ₁·v₁) + (max v₂ - σ₂·v₂) (écart à l'équilibre)
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np


# ===========================================================================
# Diagnostics théoriques : Équilibres de Nash purs
# ===========================================================================

def pure_nash_equilibria(
    U1: np.ndarray,
    U2: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    """
    Identifie tous les équilibres de Nash en stratégies pures (Chapitre 3).

    Dans un jeu sous forme normale, (i*, j*) est un équilibre de Nash pur si
    et seulement si chaque joueur joue une meilleure réponse à l'autre :
      - u₁(i*, j*) = max_{i} u₁(i, j*)  (meilleure réponse de J1 à j*)
      - u₂(i*, j*) = max_{j} u₂(i*, j)  (meilleure réponse de J2 à i*)

    Pour un jeu à intérêts communs (U1 = U2 = U), (i*, j*) est un équilibre
    de Nash pur ssi U[i*, j*] est simultanément le maximum de sa colonne
    et le maximum de sa ligne.

    Paramètres
    ----------
    U1 : np.ndarray de forme (k1, k2)
        Matrice de gains du joueur 1.
    U2 : Optional[np.ndarray] de forme (k1, k2)
        Matrice de gains du joueur 2 (par défaut égale à U1).

    Retourne
    --------
    List[Tuple[int, int]]
        Liste des profils de stratégies pures (i*, j*) constituant un équilibre de Nash.
    """
    if U2 is None:
        U2 = U1

    k1, k2 = U1.shape
    col_maxes = np.max(U1, axis=0)  # max par colonne (pour J1)
    row_maxes = np.max(U2, axis=1)  # max par ligne (pour J2)

    equilibria: List[Tuple[int, int]] = []
    eps_tol = 1e-9

    for i in range(k1):
        for j in range(k2):
            is_best_j1 = abs(U1[i, j] - col_maxes[j]) < eps_tol
            is_best_j2 = abs(U2[i, j] - row_maxes[i]) < eps_tol
            if is_best_j1 and is_best_j2:
                equilibria.append((i, j))

    return equilibria


# ===========================================================================
# Dynamique de Réplication avec Plancher d'Exploration
# ===========================================================================

def replicator_step(
    sigma: np.ndarray,
    v: np.ndarray,
    eta: float = 0.5,
    eps_min: float = 0.05,
) -> np.ndarray:
    """
    Calcule un pas de la dynamique de réplication exponentielle discrète.

    Formule :
      1. Gain moyen : avg_v = σ · v
      2. Croissance exponentielle : σ_i' = σ_i · exp(η · (v_i - avg_v))
      3. Normalisation : σ_i'' = σ_i' / sum(σ')
      4. Plancher d'exploration ε_min : σ_final = ε_min + (1 - K · ε_min) · σ''

    Paramètres
    ----------
    sigma : np.ndarray
        Distribution de probabilité courante (stratégie mixte).
    v : np.ndarray
        Vecteur des gains espérés de chaque stratégie pure.
    eta : float
        Taux d'apprentissage de la dynamique.
    eps_min : float
        Plancher d'exploration garantissant σ_i >= eps_min > 0.

    Retourne
    --------
    np.ndarray
        Nouvelle distribution de probabilité normalisée.
    """
    k = len(sigma)
    avg_v = float(np.dot(sigma, v))
    diff = eta * (v - avg_v)

    # Stabilité numérique : soustraction du maximum avant l'exponentielle
    diff -= np.max(diff)
    new_sigma = sigma * np.exp(diff)
    sum_new = np.sum(new_sigma)
    if sum_new > 1e-15:
        new_sigma /= sum_new
    else:
        new_sigma = np.ones(k) / k

    # Application du plancher d'exploration pour éviter l'extinction
    if eps_min > 0.0 and k * eps_min < 1.0:
        new_sigma = eps_min + (1.0 - k * eps_min) * new_sigma
        new_sigma /= np.sum(new_sigma)

    return new_sigma


# ===========================================================================
# Composant Jeu : OperatorGame
# ===========================================================================

class OperatorGame:
    """
    Jeu stratégique à deux joueurs à intérêts communs (u1 = u2 = U)
    coordonnant le croisement et la mutation dans un algorithme génétique.

    Joueurs :
      - Joueur 1 (Croisement) : sélectionne le taux de croisement cr
      - Joueur 2 (Mutation)   : sélectionne le taux de mutation mr

    Paramètres
    ----------
    eta : float
        Taux d'apprentissage de la dynamique de réplication (défaut 0.5).
    eps_min : float
        Plancher d'exploration par stratégie (défaut 0.05).
    beta : float
        Facteur de lissage EMA pour la matrice de gains U (défaut 0.1).
    rng : Optional[np.random.Generator]
        Générateur aléatoire explicite pour la reproductibilité stricte.
    """

    # Grilles de stratégies discrètes
    # Note : Le couple par défaut du GA standard (cr=0.80, mr=0.05) est explicitement présent à (indice 2, indice 1)
    STRATEGIES_CROSSOVER: List[float] = [0.50, 0.65, 0.80, 0.95]
    STRATEGIES_MUTATION: List[float] = [0.01, 0.05, 0.10, 0.20]

    # Alias pour compatibilité ascendante
    STRATEGIES_EXPLORATION = STRATEGIES_CROSSOVER
    STRATEGIES_EXPLOITATION = STRATEGIES_MUTATION

    def __init__(
        self,
        eta: float = 0.5,
        eps_min: float = 0.05,
        beta: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ):
        self.k1: int = len(self.STRATEGIES_CROSSOVER)
        self.k2: int = len(self.STRATEGIES_MUTATION)
        self.eta: float = float(eta)
        self.eps_min: float = float(eps_min)
        self.beta: float = float(beta)
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

        # Matrice de gains à intérêts communs U (u1 = u2 = U)
        # Initialisée à 0.5 (valeur neutre dans [0, 1] évitant le biais d'initialisation à 0)
        self.U: np.ndarray = np.full((self.k1, self.k2), 0.5, dtype=float)

        # Stratégies mixtes initiales : uniformes
        self.sigma1: np.ndarray = np.ones(self.k1, dtype=float) / self.k1
        self.sigma2: np.ndarray = np.ones(self.k2, dtype=float) / self.k2

        # Historiques de diagnostic
        self.sigma1_history: List[np.ndarray] = []
        self.sigma2_history: List[np.ndarray] = []
        self.nash_gap_history: List[float] = []
        self.pure_nash_history: List[List[Tuple[int, int]]] = []
        self.U_history: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Tirage des stratégies par enfant
    # ------------------------------------------------------------------

    def sample_strategies(
        self,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, int, float, float]:
        """
        Tire une paire de stratégies (i, j) selon les distributions (σ₁, σ₂).

        Paramètres
        ----------
        rng : Optional[np.random.Generator]
            Générateur explicite.

        Retourne
        --------
        idx1 : int
            Indice de la stratégie de croisement choisie.
        idx2 : int
            Indice de la stratégie de mutation choisie.
        cr : float
            Valeur concrète du taux de croisement.
        mr : float
            Valeur concrète du taux de mutation.
        """
        gen = rng if rng is not None else self.rng
        idx1 = int(gen.choice(self.k1, p=self.sigma1))
        idx2 = int(gen.choice(self.k2, p=self.sigma2))
        cr = self.STRATEGIES_CROSSOVER[idx1]
        mr = self.STRATEGIES_MUTATION[idx2]
        return idx1, idx2, cr, mr

    # ------------------------------------------------------------------
    # Normalisation Min-Max de U
    # ------------------------------------------------------------------

    def get_normalized_payoff_matrix(self) -> np.ndarray:
        """
        Normalise la matrice U dans [0, 1] par min-max.
        Si tous les éléments sont égaux, retourne une matrice uniforme.
        """
        u_min = float(np.min(self.U))
        u_max = float(np.max(self.U))
        diff = u_max - u_min
        if diff > 1e-10:
            return (self.U - u_min) / diff
        return np.full_like(self.U, 0.5)

    # ------------------------------------------------------------------
    # Mise à jour par génération
    # ------------------------------------------------------------------

    def update_generation(
        self,
        children_evaluations: List[Tuple[int, int, float]],
    ) -> Dict[str, Any]:
        """
        Met à jour le jeu à partir de tous les enfants produits au cours de la génération.

        Étapes :
          1. Agrégation des payoffs u des enfants par case (i, j).
          2. Mise à jour EMA : U[i, j] ← (1 - β)·U[i, j] + β·moyenne(u)
             (uniquement pour les cases ayant produit au moins un enfant).
          3. Normalisation min-max de U → U_norm.
          4. Vecteurs de gains espérés : v₁ = U_norm · σ₂,  v₂ = U_norm^T · σ₁.
          5. Dynamique de réplication discrète sur σ₁ et σ₂ avec plancher ε_min.
          6. Calcul du Nash Gap : (max v₁ - σ₁·v₁) + (max v₂ - σ₂·v₂).
          7. Détection des équilibres de Nash purs de U_norm.

        Paramètres
        ----------
        children_evaluations : List[Tuple[int, int, float]]
            Liste des triplets (i, j, u) où u ∈ [0, 1] est le payoff de chaque enfant.

        Retourne
        --------
        dict
            Métriques de diagnostic de la génération courante.
        """
        if not children_evaluations:
            return {}

        # 1. Regrouper les payoffs par case (i, j)
        cell_payoffs: Dict[Tuple[int, int], List[float]] = {}
        for idx1, idx2, u in children_evaluations:
            cell_payoffs.setdefault((idx1, idx2), []).append(u)

        # 2. Mise à jour EMA uniquement pour les cases ayant au moins un enfant
        for (i, j), payoffs in cell_payoffs.items():
            mean_u = float(np.mean(payoffs))
            self.U[i, j] = (1.0 - self.beta) * self.U[i, j] + self.beta * mean_u

        # 3. Normalisation min-max
        U_norm = self.get_normalized_payoff_matrix()

        # 4. Gains espérés
        v1 = U_norm @ self.sigma2
        v2 = U_norm.T @ self.sigma1

        # 5. Dynamique de réplication avec plancher eps_min
        self.sigma1 = replicator_step(self.sigma1, v1, self.eta, self.eps_min)
        self.sigma2 = replicator_step(self.sigma2, v2, self.eta, self.eps_min)

        # 6. Diagnostics théoriques : Nash Gap
        # Définition : incitation maximale unilatérale à dévier pour les deux joueurs
        gap1 = float(np.max(v1) - np.dot(self.sigma1, v1))
        gap2 = float(np.max(v2) - np.dot(self.sigma2, v2))
        nash_gap = max(0.0, gap1 + gap2)

        # 7. Équilibres de Nash purs
        pure_eqs = pure_nash_equilibria(U_norm)

        # Enregistrement dans les historiques
        self.sigma1_history.append(self.sigma1.copy())
        self.sigma2_history.append(self.sigma2.copy())
        self.nash_gap_history.append(nash_gap)
        self.pure_nash_history.append(pure_eqs)
        self.U_history.append(self.U.copy())

        return {
            "nash_gap": nash_gap,
            "pure_nash_equilibria": pure_eqs,
            "sigma1": self.sigma1.copy(),
            "sigma2": self.sigma2.copy(),
            "U_normalized": U_norm.copy(),
        }

    # ------------------------------------------------------------------
    # Rétro-compatibilité et accesseurs
    # ------------------------------------------------------------------

    @property
    def A(self) -> np.ndarray:
        """Alias pour U (jeu à intérêts communs u1 = u2 = U)."""
        return self.U

    @property
    def B(self) -> np.ndarray:
        """Alias pour U (jeu à intérêts communs u1 = u2 = U)."""
        return self.U

    def update(self, idx1: int, idx2: int, gain1: float, gain2: float) -> None:
        """Interface compatible avec une mise à jour scalaire simple."""
        u = float(np.clip(0.5 * (gain1 + gain2 + 2.0) / 2.0, 0.0, 1.0))
        self.update_generation([(idx1, idx2, u)])

    def get_strategy_proportions(self) -> dict:
        """Retourne les distributions et métriques courantes."""
        last_gap = self.nash_gap_history[-1] if self.nash_gap_history else 0.0
        last_pure = self.pure_nash_history[-1] if self.pure_nash_history else []
        return {
            "crossover_probs": self.sigma1.tolist(),
            "mutation_probs": self.sigma2.tolist(),
            "crossover_strategies": self.STRATEGIES_CROSSOVER,
            "mutation_strategies": self.STRATEGIES_MUTATION,
            "exploration_probs": self.sigma1.tolist(),
            "exploitation_probs": self.sigma2.tolist(),
            "exploration_strategies": self.STRATEGIES_CROSSOVER,
            "exploitation_strategies": self.STRATEGIES_MUTATION,
            "nash_gap": last_gap,
            "pure_nash_equilibria": last_pure,
        }

    def nash_equilibrium_value(self) -> float:
        """Gain espéré global sous le profil mixte courant."""
        U_norm = self.get_normalized_payoff_matrix()
        return float(self.sigma1 @ U_norm @ self.sigma2)
