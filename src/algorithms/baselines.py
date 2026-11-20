"""
Module : Algorithmes de Référence (Baselines pour l'évaluation comparative)
==========================================================================
Ce module implémente les deux métaheuristiques de référence additionnelles :
  1. GAUniformRandom : GA avec taux tirés uniformément sans apprentissage.
  2. GAProbabilityMatching : GA à sélection adaptative d'opérateurs (AOS)
     sans théorie des jeux (Probability Matching classique à récompenses indépendantes).

Ces baselines permettent d'isoler scientifiquement :
  - L'effet de la stochasticité dynamique vs le jeu (GAUniformRandom vs GA-Nash).
  - L'effet de la modélisation en théorie des jeux (coordination matricielle U[i, j]
    et dynamique de réplication) vs une adaptation heuristique indépendante (GAProbabilityMatching vs GA-Nash).
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.game_theory.nash_game import OperatorGame


# ===========================================================================
# 1. GA à Taux Aléatoires Uniformes sans Apprentissage
# ===========================================================================

class GAUniformRandom(GeneticAlgorithm):
    """
    Algorithme Génétique avec tirage uniforme et indépendant des taux (cr, mr) par enfant.
    Aucune mise à jour de probabilités, aucune mémoire, aucun apprentissage.
    """

    def __init__(
        self,
        problem,
        pop_size: int = 50,
        max_gen: int = 200,
        crossover_type: str = "uniform",
        seed: Optional[int] = None,
    ):
        super().__init__(
            problem=problem,
            pop_size=pop_size,
            max_gen=max_gen,
            crossover_type=crossover_type,
            seed=seed,
        )
        self.strategies_cr = OperatorGame.STRATEGIES_CROSSOVER
        self.strategies_mr = OperatorGame.STRATEGIES_MUTATION
        self.k1 = len(self.strategies_cr)
        self.k2 = len(self.strategies_mr)

    def run(self, initial_pop: Optional[np.ndarray] = None) -> dict:
        """Lance l'optimisation avec taux aléatoires uniformes par enfant."""
        self.problem.reset_evaluations_count()
        if initial_pop is not None:
            pop = np.array(initial_pop, copy=True)
        else:
            pop = self._init_population()
        fitness = self._evaluate_population(pop)

        self.best_fitness_history = []
        self.mean_fitness_history = []

        for gen in range(self.max_gen):
            children = []
            children_fitness = []

            while len(children) < self.pop_size:
                p1 = self._tournament_selection(pop, fitness)
                p2 = self._tournament_selection(pop, fitness)

                # Tirage uniforme sans apprentissage pour enfant 1
                cr1 = float(self.rng.choice(self.strategies_cr))
                mr1 = float(self.rng.choice(self.strategies_mr))
                c1, _ = self._crossover(p1, p2, cr1)
                c1 = self._mutate(c1, mr1)
                if hasattr(self.problem, "repair"):
                    c1 = self.problem.repair(c1)
                f_c1 = float(self.problem.evaluate(c1))
                children.append(c1)
                children_fitness.append(f_c1)

                if len(children) < self.pop_size:
                    # Tirage uniforme sans apprentissage pour enfant 2
                    cr2 = float(self.rng.choice(self.strategies_cr))
                    mr2 = float(self.rng.choice(self.strategies_mr))
                    c2, _ = self._crossover(p2, p1, cr2)
                    c2 = self._mutate(c2, mr2)
                    if hasattr(self.problem, "repair"):
                        c2 = self.problem.repair(c2)
                    f_c2 = float(self.problem.evaluate(c2))
                    children.append(c2)
                    children_fitness.append(f_c2)

            new_pop = np.array(children[:self.pop_size])
            new_fitness = np.array(children_fitness[:self.pop_size])

            # Élitisme
            worst_idx = np.argmax(new_fitness)
            best_old_idx = np.argmin(fitness)
            if fitness[best_old_idx] < new_fitness[worst_idx]:
                new_pop[worst_idx] = pop[best_old_idx].copy()
                new_fitness[worst_idx] = fitness[best_old_idx]

            pop = new_pop
            fitness = new_fitness

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


# ===========================================================================
# 2. GA à Sélection Adaptative sans Jeu (Probability Matching)
# ===========================================================================

class GAProbabilityMatching(GeneticAlgorithm):
    """
    Algorithme Génétique avec sélection adaptative d'opérateurs (Probability Matching).
    Chaque opérateur (croisement et mutation) met à jour indépendamment son vecteur
    d'estimation de qualité q par EMA, sans coordination ni matrice de jeu U[i, j].

    Règle de Probability Matching :
      p_i = eps_min + (1 - K * eps_min) * (q_i / sum(q))
    """

    def __init__(
        self,
        problem,
        pop_size: int = 50,
        max_gen: int = 200,
        beta: float = 0.1,
        eps_min: float = 0.05,
        crossover_type: str = "uniform",
        seed: Optional[int] = None,
    ):
        super().__init__(
            problem=problem,
            pop_size=pop_size,
            max_gen=max_gen,
            crossover_type=crossover_type,
            seed=seed,
        )
        self.beta = beta
        self.eps_min = eps_min
        self.strategies_cr = OperatorGame.STRATEGIES_CROSSOVER
        self.strategies_mr = OperatorGame.STRATEGIES_MUTATION
        self.k1 = len(self.strategies_cr)
        self.k2 = len(self.strategies_mr)

        # Estimations de qualité indépendantes (sans matrice de jeu)
        self.q1 = np.full(self.k1, 0.5, dtype=float)
        self.q2 = np.full(self.k2, 0.5, dtype=float)

        # Probabilités initiales uniformes
        self.p1 = np.ones(self.k1, dtype=float) / self.k1
        self.p2 = np.ones(self.k2, dtype=float) / self.k2

        self.strategy_history: List[dict] = []

    def _update_probabilities(self) -> None:
        """Met à jour p1 et p2 par la règle de Probability Matching standard."""
        sum_q1 = np.sum(self.q1)
        if sum_q1 > 1e-15:
            ratio1 = self.q1 / sum_q1
            self.p1 = self.eps_min + (1.0 - self.k1 * self.eps_min) * ratio1
            self.p1 /= np.sum(self.p1)

        sum_q2 = np.sum(self.q2)
        if sum_q2 > 1e-15:
            ratio2 = self.q2 / sum_q2
            self.p2 = self.eps_min + (1.0 - self.k2 * self.eps_min) * ratio2
            self.p2 /= np.sum(self.p2)

    def run(self, initial_pop: Optional[np.ndarray] = None) -> dict:
        """Lance l'optimisation avec sélection adaptative par Probability Matching."""
        self.problem.reset_evaluations_count()
        if initial_pop is not None:
            pop = np.array(initial_pop, copy=True)
        else:
            pop = self._init_population()
        fitness = self._evaluate_population(pop)

        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.strategy_history = []

        eps_denom = 1e-10

        for gen in range(self.max_gen):
            f_min_pop = float(np.min(fitness))
            f_max_pop = float(np.max(fitness))
            f_range = f_max_pop - f_min_pop

            children = []
            children_fitness = []
            cr_payoffs: Dict[int, List[float]] = {}
            mr_payoffs: Dict[int, List[float]] = {}

            while len(children) < self.pop_size:
                # Sélection des parents
                idx1 = self.rng.choice(self.pop_size, size=2, replace=False)
                w1 = idx1[np.argmin(fitness[idx1])]
                p1, f_p1 = pop[w1].copy(), float(fitness[w1])

                idx2 = self.rng.choice(self.pop_size, size=2, replace=False)
                w2 = idx2[np.argmin(fitness[idx2])]
                p2, f_p2 = pop[w2].copy(), float(fitness[w2])

                f_moy_parents = (f_p1 + f_p2) / 2.0

                # Enfant 1
                i1 = int(self.rng.choice(self.k1, p=self.p1))
                j1 = int(self.rng.choice(self.k2, p=self.p2))
                cr1 = self.strategies_cr[i1]
                mr1 = self.strategies_mr[j1]
                c1, _ = self._crossover(p1, p2, cr1)
                c1 = self._mutate(c1, mr1)
                if hasattr(self.problem, "repair"):
                    c1 = self.problem.repair(c1)
                f_c1 = float(self.problem.evaluate(c1))

                r1 = float(np.clip((f_moy_parents - f_c1) / (f_range + eps_denom), -1.0, 1.0))
                u1 = float((r1 + 1.0) / 2.0)
                children.append(c1)
                children_fitness.append(f_c1)
                cr_payoffs.setdefault(i1, []).append(u1)
                mr_payoffs.setdefault(j1, []).append(u1)

                if len(children) < self.pop_size:
                    # Enfant 2
                    i2 = int(self.rng.choice(self.k1, p=self.p1))
                    j2 = int(self.rng.choice(self.k2, p=self.p2))
                    cr2 = self.strategies_cr[i2]
                    mr2 = self.strategies_mr[j2]
                    c2, _ = self._crossover(p2, p1, cr2)
                    c2 = self._mutate(c2, mr2)
                    if hasattr(self.problem, "repair"):
                        c2 = self.problem.repair(c2)
                    f_c2 = float(self.problem.evaluate(c2))

                    r2 = float(np.clip((f_moy_parents - f_c2) / (f_range + eps_denom), -1.0, 1.0))
                    u2 = float((r2 + 1.0) / 2.0)
                    children.append(c2)
                    children_fitness.append(f_c2)
                    cr_payoffs.setdefault(i2, []).append(u2)
                    mr_payoffs.setdefault(j2, []).append(u2)

            new_pop = np.array(children[:self.pop_size])
            new_fitness = np.array(children_fitness[:self.pop_size])

            # Élitisme
            worst_idx = np.argmax(new_fitness)
            best_old_idx = np.argmin(fitness)
            if fitness[best_old_idx] < new_fitness[worst_idx]:
                new_pop[worst_idx] = pop[best_old_idx].copy()
                new_fitness[worst_idx] = fitness[best_old_idx]

            # Mise à jour des estimations de qualité marginales q1 et q2
            for i, vals in cr_payoffs.items():
                self.q1[i] = (1.0 - self.beta) * self.q1[i] + self.beta * float(np.mean(vals))
            for j, vals in mr_payoffs.items():
                self.q2[j] = (1.0 - self.beta) * self.q2[j] + self.beta * float(np.mean(vals))

            # Recalcul des probabilités par Probability Matching
            self._update_probabilities()

            pop = new_pop
            fitness = new_fitness

            best_idx = np.argmin(fitness)
            best_f = float(fitness[best_idx])
            mean_f = float(fitness.mean())

            self.best_fitness_history.append(best_f)
            self.mean_fitness_history.append(mean_f)
            self.strategy_history.append({
                "crossover_probs": self.p1.tolist(),
                "mutation_probs": self.p2.tolist(),
            })

            if best_f < self.best_fitness:
                self.best_fitness = best_f
                self.best_solution = pop[best_idx].copy()

        return {
            "best_fitness": self.best_fitness,
            "best_solution": self.best_solution,
            "best_history": self.best_fitness_history,
            "mean_history": self.mean_fitness_history,
            "strategy_history": self.strategy_history,
            "evaluations": self.problem.evaluations_count,
        }
