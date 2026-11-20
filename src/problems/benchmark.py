"""
Module : Problèmes de Benchmark
================================
Famille 1 — Fonctions continues :
  - Rastrigin-10, Rosenbrock-10, Ackley-10 (Sphere-10 comme test de cohérence)
Famille 2 — Sac à dos 0/1 binaire natif :
  - Knapsack-50, Knapsack-100, Knapsack-200
  - Instances à graine fixe déterministe
  - Résolution exacte par programmation dynamique (DP)
  - Réparateur heuristique des solutions infaisables (tri par ratio v_i / w_i)

Le module supporte la détection automatique de la taille directement depuis le nom
(ex. 'rastrigin-10', 'knapsack-50', 'knapsack_200').
"""

import re
from typing import Callable, Tuple, Any, Optional, Dict
import numpy as np


# ===========================================================================
# Résolution exacte du Sac à dos par Programmation Dynamique
# ===========================================================================

def solve_knapsack_dp(
    weights: np.ndarray,
    values: np.ndarray,
    capacity: int,
) -> Tuple[int, np.ndarray]:
    """
    Résout exactement le problème du sac à dos 0/1 binaire par programmation dynamique.

    Paramètres
    ----------
    weights : np.ndarray d'entiers
        Poids des objets.
    values : np.ndarray d'entiers
        Valeurs des objets.
    capacity : int
        Capacité maximale du sac.

    Retourne
    --------
    best_value : int
        Valeur optimale globale exacte.
    best_solution : np.ndarray
        Vecteur binaire {0, 1}^n correspondant à la sélection optimale.
    """
    n = len(weights)
    cap = int(capacity)
    w = np.asarray(weights, dtype=np.int64)
    v = np.asarray(values, dtype=np.int64)

    dp = np.zeros(cap + 1, dtype=np.int64)
    keep = np.zeros((n, cap + 1), dtype=bool)

    for i in range(n):
        wi, vi = int(w[i]), int(v[i])
        if wi > cap:
            continue
        cand = dp[:cap + 1 - wi] + vi
        better = cand > dp[wi:]
        idx = np.where(better)[0]
        keep[i, wi + idx] = True
        dp[wi + idx] = cand[idx]

    best_value = int(dp[cap])

    # Rétro-propagation (backtracking) pour extraire la solution binaire
    solution = np.zeros(n, dtype=int)
    c = cap
    for i in range(n - 1, -1, -1):
        if keep[i, c]:
            solution[i] = 1
            c -= int(w[i])

    return best_value, solution


# ===========================================================================
# Classe de base abstraite
# ===========================================================================

class BaseProblem:
    """
    Classe de base pour tous les problèmes d'optimisation (minimisation).

    Gère le comptage strict des évaluations de la fonction objectif
    pour garantir l'équité absolue du budget entre algorithmes comparés.
    """

    def __init__(self, name: str, dim: int, is_binary: bool = False):
        self.name: str = name
        self.dim: int = dim
        self.is_binary: bool = is_binary
        self.evaluations_count: int = 0
        self.lb: float = 0.0
        self.ub: float = 1.0
        self.optimum: Optional[float] = None
        self.optimum_solution: Optional[np.ndarray] = None

    def reset_evaluations_count(self) -> None:
        """Réinitialise le compteur d'évaluations à zéro."""
        self.evaluations_count = 0

    def evaluate(self, x: np.ndarray) -> float:
        """
        Évalue une solution candidate et incrémente le compteur d'évaluations.

        Paramètres
        ----------
        x : np.ndarray
            Vecteur solution candidat.

        Retourne
        --------
        float
            Valeur objectif à minimiser.
        """
        self.evaluations_count += 1
        return self._evaluate_impl(x)

    def _evaluate_impl(self, x: np.ndarray) -> float:
        """Implémentation spécifique de la fonction objectif."""
        raise NotImplementedError

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        """Génère une solution aléatoire valide à l'aide du Generator fourni."""
        raise NotImplementedError


# ===========================================================================
# Famille 1 : Fonctions continues de benchmark
# ===========================================================================

class ContinuousBenchmark(BaseProblem):
    """
    Fonctions continues standard pour l'optimisation (minimisation).

    Problèmes disponibles :
      - Rastrigin : multimodale hautement non-linéaire (optimum f(0)=0)
      - Rosenbrock : vallée parabolique étroite (optimum f(1)=0)
      - Sphere : convexe quadratique unimodale (test de cohérence)
      - Ackley : multimodale avec bassin central étroit (optimum f(0)=0)
    """

    def __init__(self, name: str, dim: int = 10):
        name_clean = name.lower()
        super().__init__(name=f"{name_clean}-{dim}", dim=dim, is_binary=False)
        self._setup(name_clean)

    def _setup(self, name: str) -> None:
        """Configure les bornes, l'optimum connu et la fonction objectif."""
        configs = {
            "rastrigin": {
                "bounds": (-5.12, 5.12),
                "optimum": 0.0,
                "optimum_solution": np.zeros(self.dim),
                "func": self._rastrigin,
            },
            "rosenbrock": {
                "bounds": (-2.048, 2.048),
                "optimum": 0.0,
                "optimum_solution": np.ones(self.dim),
                "func": self._rosenbrock,
            },
            "sphere": {
                "bounds": (-5.12, 5.12),
                "optimum": 0.0,
                "optimum_solution": np.zeros(self.dim),
                "func": self._sphere,
            },
            "ackley": {
                "bounds": (-32.768, 32.768),
                "optimum": 0.0,
                "optimum_solution": np.zeros(self.dim),
                "func": self._ackley,
            },
        }
        if name not in configs:
            raise ValueError(f"Fonction continue inconnue : {name}")

        cfg = configs[name]
        self.lb, self.ub = cfg["bounds"]
        self.optimum = cfg["optimum"]
        self.optimum_solution = cfg["optimum_solution"]
        self._func: Callable[[np.ndarray], float] = cfg["func"]

    def _evaluate_impl(self, x: np.ndarray) -> float:
        """Évalue la fonction objectif continue."""
        return float(self._func(x))

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        """Génère une solution continue aléatoire uniforme dans [lb, ub]."""
        return rng.uniform(self.lb, self.ub, self.dim)

    # --- Formulations mathématiques des fonctions continues ---

    def _rastrigin(self, x: np.ndarray) -> float:
        """
        f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
        Minimum global : f(0,...,0) = 0.
        """
        n = len(x)
        return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

    def _rosenbrock(self, x: np.ndarray) -> float:
        """
        f(x) = sum(100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2)
        Minimum global : f(1,...,1) = 0.
        """
        return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

    def _sphere(self, x: np.ndarray) -> float:
        """f(x) = sum(x_i^2). Minimum global : f(0,...,0) = 0."""
        return float(np.sum(x**2))

    def _ackley(self, x: np.ndarray) -> float:
        """
        f(x) = -20*exp(-0.2*sqrt(sum(x^2)/n)) - exp(sum(cos(2*pi*x))/n) + 20 + e
        Minimum global : f(0,...,0) = 0.
        """
        n = len(x)
        a, b, c = 20.0, 0.2, 2.0 * np.pi
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)
        return float(term1 + term2 + a + np.e)


# ===========================================================================
# Famille 2 : Problème du Sac à dos binaire natif (Knapsack 0/1)
# ===========================================================================

class KnapsackProblem(BaseProblem):
    """
    Problème du Sac à dos 0/1 binaire natif.

    Maximiser sum(v_i * x_i) sous contrainte sum(w_i * x_i) <= W, avec x_i in {0, 1}.
    Formulé en minimisation standard : f(x) = -sum(v_i * x_i).

    Caractéristiques :
      - Instances déterministes à graine fixe (50, 100, 200 objets)
      - Optimum exact résolu par programmation dynamique (solve_knapsack_dp)
      - Opérateur de réparation heuristique pour toute solution infaisable
        (retrait prioritaire des objets selon le ratio v_i / w_i croissant)
    """

    def __init__(self, n_items: int = 50, seed: int = 42):
        super().__init__(name=f"knapsack-{n_items}", dim=n_items, is_binary=True)
        self.n_items: int = n_items
        self.instance_seed: int = seed

        # Génération déterministe de l'instance
        rng = np.random.default_rng(seed)
        self.weights: np.ndarray = rng.integers(1, 31, size=n_items, dtype=np.int64)
        self.values: np.ndarray = rng.integers(1, 51, size=n_items, dtype=np.int64)
        self.capacity: int = int(0.5 * self.weights.sum())

        # Résolution exacte par programmation dynamique
        self.optimum_value, self.optimum_solution = solve_knapsack_dp(
            self.weights, self.values, self.capacity
        )
        # En minimisation : l'optimum correspond à -valeur_optimale
        self.optimum: float = -float(self.optimum_value)
        self.lb: float = 0.0
        self.ub: float = 1.0

    def repair(self, x: np.ndarray) -> np.ndarray:
        """
        Répare une solution binaire infaisable (dépassement de capacité).

        Heuristique : retire en priorité les objets sélectionnés ayant le plus
        faible ratio valeur / poids (v_i / w_i) jusqu'à respecter la capacité W.

        Paramètres
        ----------
        x : np.ndarray
            Vecteur solution candidat (binaire ou quasi-binaire).

        Retourne
        --------
        np.ndarray
            Vecteur solution binaire réparé et garanti faisable.
        """
        repaired = (x >= 0.5).astype(int)
        total_weight = int(np.dot(repaired, self.weights))

        if total_weight <= self.capacity:
            return repaired.astype(float)

        selected = np.where(repaired == 1)[0]
        # Tri par ratio valeur/poids croissant (les moins efficaces d'abord)
        ratios = self.values[selected] / self.weights[selected]
        order = selected[np.argsort(ratios)]

        for idx in order:
            repaired[idx] = 0
            total_weight -= int(self.weights[idx])
            if total_weight <= self.capacity:
                break

        return repaired.astype(float)

    def _evaluate_impl(self, x: np.ndarray) -> float:
        """
        Évalue une solution candidate.
        La solution est réparée si infaisable, puis la valeur est inversée pour minimisation.
        """
        repaired = self.repair(x)
        total_value = float(np.dot(repaired, self.values))
        return -total_value

    def random_solution(self, rng: np.random.Generator) -> np.ndarray:
        """Génère une solution aléatoire binaire garantie faisable."""
        raw = rng.integers(0, 2, size=self.n_items).astype(float)
        return self.repair(raw)

    def decode(self, x: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Décode une solution candidate en solution binaire réparée.

        Retourne
        --------
        bits : np.ndarray
            Vecteur binaire réparé.
        weight : int
            Poids total cumulé (<= capacité).
        value : int
            Valeur totale cumulée.
        """
        repaired = self.repair(x).astype(int)
        w = int(np.dot(repaired, self.weights))
        v = int(np.dot(repaired, self.values))
        return repaired, w, v


# ===========================================================================
# Analyseur de nom et Fabrique (Factory)
# ===========================================================================

def parse_problem_name(name: str) -> Tuple[str, Optional[int]]:
    """
    Extrait le nom de base du problème et sa dimension / taille si présente dans la chaîne.

    Exemples :
      - 'rastrigin-10' -> ('rastrigin', 10)
      - 'knapsack_50'   -> ('knapsack', 50)
      - 'ackley'        -> ('ackley', None)
      - 'knapsack-200'  -> ('knapsack', 200)
    """
    cleaned = name.strip().lower()
    match = re.match(r"^([a-z]+)[-_]?(\d+)?$", cleaned)
    if match:
        base = match.group(1)
        size = int(match.group(2)) if match.group(2) is not None else None
        return base, size
    return cleaned, None


def get_problem(name: str, **kwargs: Any) -> BaseProblem:
    """
    Instancie un problème de benchmark à partir de son nom.
    Lit automatiquement la dimension ou la taille d'objets depuis le nom si disponible.

    Paramètres
    ----------
    name : str
        Nom du problème ('rastrigin-10', 'rosenbrock-10', 'ackley-10', 'sphere-10',
                         'knapsack-50', 'knapsack-100', 'knapsack-200', etc.).
    **kwargs :
        Paramètres supplémentaires facultatifs :
          - dim : dimension pour fonctions continues (par défaut 10)
          - n_items : nombre d'objets pour knapsack (par défaut 50)
          - seed : graine pour l'instance knapsack (par défaut 42)

    Retourne
    --------
    BaseProblem
        Instance configurée héritant de BaseProblem.
    """
    base_name, size = parse_problem_name(name)
    continuous_names = ["rastrigin", "rosenbrock", "sphere", "ackley"]

    if base_name in continuous_names:
        dim = size if size is not None else kwargs.get("dim", 10)
        return ContinuousBenchmark(base_name, dim=dim)
    elif base_name == "knapsack":
        n_items = size if size is not None else kwargs.get("n_items", 50)
        seed = kwargs.get("seed", 42)
        return KnapsackProblem(n_items=n_items, seed=seed)
    else:
        raise ValueError(
            f"Problème inconnu : '{name}'. Problèmes valides : "
            f"{AVAILABLE_PROBLEMS['continuous']} ou {AVAILABLE_PROBLEMS['combinatorial']}"
        )


AVAILABLE_PROBLEMS: Dict[str, list] = {
    "continuous": ["rastrigin-10", "rosenbrock-10", "ackley-10", "sphere-10"],
    "combinatorial": ["knapsack-50", "knapsack-100", "knapsack-200"],
}
