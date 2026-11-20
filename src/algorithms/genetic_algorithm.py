"""
Module : Algorithme Génétique de base (GA standard)
======================================================
Algorithme de référence — sans théorie des jeux.
Paramètres fixes : crossover_rate (cr), mutation_rate (mr) constants.

Prise en charge native :
  - Problèmes continus : Croisement BLX-alpha, Mutation gaussienne
  - Problèmes combinatoires binaires : Croisement uniforme ou 1-point, Mutation bit-flip,
    et opérateur de réparation des solutions infaisables.
"""

from typing import Optional, List, Tuple
import numpy as np


class GeneticAlgorithm:
    """
    Algorithme génétique standard (baseline).

    Opérateurs selon la nature du problème :
      - Continu :
          * Sélection : tournoi binaire (k=2)
          * Croisement : BLX-alpha (alpha=0.5)
          * Mutation : gaussienne (sigma = 0.1 * (ub - lb))
          * Remplacement : élitisme
      - Binaire (ex. Sac à dos) :
          * Sélection : tournoi binaire (k=2)
          * Croisement : uniforme (par défaut) ou 1-point
          * Mutation : bit-flip indépendant par gène
          * Réparation : élimination des dépassements de capacité
          * Remplacement : élitisme
    """

    def __init__(
        self,
        problem,
        pop_size: int = 50,
        max_gen: int = 200,
        crossover_rate: float = 0.80,
        mutation_rate: float = 0.05,
        crossover_type: str = "uniform",
        seed: Optional[int] = None,
    ):
        self.problem = problem
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.cr = crossover_rate
        self.mr = mutation_rate
        self.crossover_type = crossover_type
        self.rng = np.random.default_rng(seed)
        self.dim = problem.dim
        self.lb = problem.lb
        self.ub = problem.ub
        self.is_binary: bool = getattr(problem, "is_binary", False)

        # Historique
        self.best_fitness_history: List[float] = []
        self.mean_fitness_history: List[float] = []
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = float("inf")

    def _init_population(self) -> np.ndarray:
        """Initialise la population aléatoirement selon le type d'encodage."""
        if self.is_binary:
            return np.array([self.problem.random_solution(self.rng)
                             for _ in range(self.pop_size)], dtype=float)
        return self.rng.uniform(self.lb, self.ub,
                                size=(self.pop_size, self.dim))

    def _evaluate_population(self, pop: np.ndarray) -> np.ndarray:
        """Évalue toute la population via le problème."""
        return np.array([self.problem.evaluate(ind) for ind in pop])

    def _tournament_selection(self, pop: np.ndarray,
                               fitness: np.ndarray,
                               k: int = 2) -> np.ndarray:
        """Sélection par tournoi déterministe de taille k."""
        indices = self.rng.choice(self.pop_size, size=k, replace=False)
        winner = indices[np.argmin(fitness[indices])]
        return pop[winner].copy()

    # --- Croisements ---

    def _crossover(self, p1: np.ndarray, p2: np.ndarray,
                   cr: float) -> Tuple[np.ndarray, np.ndarray]:
        """Aiguille vers l'opérateur de croisement adapté."""
        if self.is_binary:
            if self.crossover_type == "single_point":
                return self._crossover_binary_single_point(p1, p2, cr)
            return self._crossover_binary_uniform(p1, p2, cr)
        return self._crossover_continuous_blx(p1, p2, cr)

    def _crossover_continuous_blx(self, p1: np.ndarray, p2: np.ndarray,
                                  cr: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Croisement continu BLX-alpha (Blend Crossover).
        alpha = 0.5 produit deux enfants dans la région élargie.
        """
        if self.rng.random() > cr:
            return p1.copy(), p2.copy()

        alpha = 0.5
        child1 = np.empty(self.dim)
        child2 = np.empty(self.dim)
        for i in range(self.dim):
            lo = min(p1[i], p2[i]) - alpha * abs(p1[i] - p2[i])
            hi = max(p1[i], p2[i]) + alpha * abs(p1[i] - p2[i])
            lo = max(lo, self.lb)
            hi = min(hi, self.ub)
            child1[i] = self.rng.uniform(lo, hi)
            child2[i] = self.rng.uniform(lo, hi)
        return child1, child2

    def _crossover_binary_uniform(self, p1: np.ndarray, p2: np.ndarray,
                                  cr: float) -> Tuple[np.ndarray, np.ndarray]:
        """Croisement binaire uniforme."""
        if self.rng.random() > cr:
            return p1.copy(), p2.copy()
        mask = self.rng.random(self.dim) < 0.5
        child1 = np.where(mask, p1, p2).astype(float)
        child2 = np.where(mask, p2, p1).astype(float)
        return child1, child2

    def _crossover_binary_single_point(self, p1: np.ndarray, p2: np.ndarray,
                                       cr: float) -> Tuple[np.ndarray, np.ndarray]:
        """Croisement binaire en 1 point."""
        if self.rng.random() > cr or self.dim < 2:
            return p1.copy(), p2.copy()
        pt = int(self.rng.integers(1, self.dim))
        child1 = np.concatenate([p1[:pt], p2[pt:]]).astype(float)
        child2 = np.concatenate([p2[:pt], p1[pt:]]).astype(float)
        return child1, child2

    # --- Mutations ---

    def _mutate(self, ind: np.ndarray, mr: float) -> np.ndarray:
        """Aiguille vers l'opérateur de mutation adapté."""
        if self.is_binary:
            return self._mutate_binary_bitflip(ind, mr)
        return self._mutate_continuous_gaussian(ind, mr)

    def _mutate_continuous_gaussian(self, ind: np.ndarray, mr: float) -> np.ndarray:
        """
        Mutation gaussienne : chaque coordonnée est perturbée avec probabilité mr.
        Amplitude sigma = 0.1 * (ub - lb).
        """
        sigma = 0.1 * (self.ub - self.lb)
        mask = self.rng.random(self.dim) < mr
        noise = self.rng.normal(0.0, sigma, self.dim)
        mutated = ind + mask * noise
        return np.clip(mutated, self.lb, self.ub)

    def _mutate_binary_bitflip(self, ind: np.ndarray, mr: float) -> np.ndarray:
        """Mutation bit-flip : chaque bit est inversé avec probabilité mr."""
        mask = self.rng.random(self.dim) < mr
        mutated = ind.copy()
        mutated[mask] = 1.0 - mutated[mask]
        return mutated

    # --- Évolution d'une génération ---

    def _next_generation(self, pop: np.ndarray,
                          fitness: np.ndarray,
                          cr: float, mr: float) -> Tuple[np.ndarray, np.ndarray]:
        """Génère la génération suivante avec croisement, mutation, réparation et élitisme."""
        new_pop = []
        while len(new_pop) < self.pop_size:
            p1 = self._tournament_selection(pop, fitness)
            p2 = self._tournament_selection(pop, fitness)
            c1, c2 = self._crossover(p1, p2, cr)
            c1 = self._mutate(c1, mr)
            c2 = self._mutate(c2, mr)

            # Réparation des solutions infaisables (pour sac à dos)
            if hasattr(self.problem, "repair"):
                c1 = self.problem.repair(c1)
                c2 = self.problem.repair(c2)

            new_pop.extend([c1, c2])

        new_pop = np.array(new_pop[:self.pop_size])
        new_fitness = self._evaluate_population(new_pop)

        # Élitisme : préserve la meilleure solution de la population précédente
        worst_idx = np.argmax(new_fitness)
        best_old_idx = np.argmin(fitness)
        if fitness[best_old_idx] < new_fitness[worst_idx]:
            new_pop[worst_idx] = pop[best_old_idx].copy()
            new_fitness[worst_idx] = fitness[best_old_idx]

        return new_pop, new_fitness

    def run(self, initial_pop: Optional[np.ndarray] = None) -> dict:
        """
        Lance l'optimisation.

        Paramètres
        ----------
        initial_pop : Optional[np.ndarray]
            Population initiale optionnelle pour appariement strict entre algorithmes.

        Retourne
        --------
        dict avec :
            best_fitness, best_solution,
            best_history, mean_history, evaluations
        """
        self.problem.reset_evaluations_count()
        if initial_pop is not None:
            pop = np.array(initial_pop, copy=True)
        else:
            pop = self._init_population()
        fitness = self._evaluate_population(pop)

        self.best_fitness_history = []
        self.mean_fitness_history = []

        for gen in range(self.max_gen):
            pop, fitness = self._next_generation(pop, fitness, self.cr, self.mr)
            best_idx = np.argmin(fitness)
            best_f = float(fitness[best_idx])
            mean_f = float(fitness.mean())

            self.best_fitness_history.append(best_f)
            self.mean_fitness_history.append(mean_f)

            if best_f < self.best_fitness:
                self.best_fitness = best_f
                self.best_solution = pop[best_idx].copy()

        return {
            "best_fitness": self.best_fitness,
            "best_solution": self.best_solution,
            "best_history": self.best_fitness_history,
            "mean_history": self.mean_fitness_history,
            "evaluations": self.problem.evaluations_count,
        }
