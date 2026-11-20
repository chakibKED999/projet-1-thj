"""
Module : Algorithme Génétique hybride avec Théorie des Jeux (GA-Nash)
=======================================================================
Intégration rigoureuse d'un jeu à intérêts communs entre le croisement et la mutation.

Modélisation formelle du jeu (Partie B du projet) :
---------------------------------------------------
1. JOUEURS :
   - Joueur 1 (J1) : Opérateur de Croisement (sélectionne cr ∈ S₁)
   - Joueur 2 (J2) : Opérateur de Mutation (sélectionne mr ∈ S₂)

2. STRATÉGIES :
   - S₁ = {0.50, 0.65, 0.80, 0.95} (taux de croisement)
   - S₂ = {0.01, 0.05, 0.10, 0.20} (taux de mutation)
   (Le profil par défaut du GA standard cr=0.80, mr=0.05 est présent à l'indice (2, 1)).

3. GAINS (Jeu à intérêts communs : u₁ = u₂ = U) :
   - Chaque ENFANT k tire un couple (i, j) selon (σ₁, σ₂).
   - Récompense individuelle de l'enfant :
       r = clip((f_moy_parents - f_enfant) / (f_max_pop - f_min_pop + ε), -1, 1)
       payoff u = (r + 1) / 2 ∈ [0, 1]
   - Matrice U[i, j] mise à jour par EMA (facteur β) sur la moyenne des u des enfants
     ayant joué la case (i, j) lors de la génération.
   - Normalisation min-max de U avant le calcul des gains espérés.

4. DYNAMIQUE D'APPRENTISSAGE :
   - Dynamique de réplication discrète (Replicator Dynamics) sur σ₁ et σ₂ avec taux η.
   - Plancher d'exploration ε_min garantissant σ_k >= ε_min > 0 pour éviter l'extinction.

5. CE QUE LE JEU MODIFIE CONCRÈTEMENT :
   Les paramètres (cr, mr) ne sont plus statiques ni uniformes à toute la population :
   chaque enfant hérite d'une combinaison stratégique tirée selon l'état du jeu,
   qui s'adapte dynamiquement au paysage de recherche au fil des générations.

6. DIAGNOSTICS DE THÉORIE DES JEUX :
   - Équilibres de Nash purs (meilleures réponses mutuelles selon Définition Chapitre 3).
   - Nash Gap = (max v₁ - σ₁·v₁) + (max v₂ - σ₂·v₂) (mesure quantitative de convergence).
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.game_theory.nash_game import OperatorGame


class GANash(GeneticAlgorithm):
    """
    GA hybride avec composant de Théorie des Jeux (Jeu à intérêts communs).

    Hérite de GeneticAlgorithm et remplace les paramètres fixes cr, mr
    par un jeu stratégique entre les opérateurs de croisement et de mutation.
    """

    def __init__(
        self,
        problem,
        pop_size: int = 50,
        max_gen: int = 200,
        eta: float = 0.5,
        eps_min: float = 0.05,
        beta: float = 0.1,
        crossover_type: str = "uniform",
        seed: Optional[int] = None,
    ):
        super().__init__(
            problem=problem,
            pop_size=pop_size,
            max_gen=max_gen,
            crossover_rate=0.80,
            mutation_rate=0.05,
            crossover_type=crossover_type,
            seed=seed,
        )
        self.eta: float = eta
        self.eps_min: float = eps_min
        self.beta: float = beta

        # Composant de théorie des jeux
        self.game = OperatorGame(
            eta=eta,
            eps_min=eps_min,
            beta=beta,
            rng=self.rng,
        )

        # Suivi historique
        self.strategy_history: List[dict] = []
        self.nash_gap_history: List[float] = []
        self.pure_nash_history: List[List[Tuple[int, int]]] = []

    def _tournament_selection_with_fitness(
        self,
        pop: np.ndarray,
        fitness: np.ndarray,
        k: int = 2,
    ) -> Tuple[np.ndarray, float]:
        """Sélection par tournoi retournant à la fois l'individu et son fitness sans réévaluation."""
        indices = self.rng.choice(self.pop_size, size=k, replace=False)
        winner = indices[np.argmin(fitness[indices])]
        return pop[winner].copy(), float(fitness[winner])

    def run(self, initial_pop: Optional[np.ndarray] = None) -> dict:
        """
        Lance l'optimisation GA-Nash.

        Paramètres
        ----------
        initial_pop : Optional[np.ndarray]
            Population initiale optionnelle pour appariement strict entre algorithmes.

        Chaque enfant tire sa propre stratégie (cr, mr) selon (σ₁, σ₂),
        est évalué, et contribue à la mise à jour de la matrice de gains U
        et de la dynamique de réplication.
        """
        self.problem.reset_evaluations_count()
        if initial_pop is not None:
            pop = np.array(initial_pop, copy=True)
        else:
            pop = self._init_population()
        fitness = self._evaluate_population(pop)

        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.strategy_history = []
        self.nash_gap_history = []
        self.pure_nash_history = []

        eps_denom = 1e-10

        for gen in range(self.max_gen):
            f_min_pop = float(np.min(fitness))
            f_max_pop = float(np.max(fitness))
            f_range = f_max_pop - f_min_pop

            children: List[np.ndarray] = []
            children_fitness: List[float] = []
            children_payoffs: List[Tuple[int, int, float]] = []

            # Production des enfants par sélection, croisement et mutation
            while len(children) < self.pop_size:
                p1, f_p1 = self._tournament_selection_with_fitness(pop, fitness)
                p2, f_p2 = self._tournament_selection_with_fitness(pop, fitness)
                f_moy_parents = (f_p1 + f_p2) / 2.0

                # --- Enfant 1 ---
                idx1_1, idx2_1, cr1, mr1 = self.game.sample_strategies(self.rng)
                c1, _ = self._crossover(p1, p2, cr1)
                c1 = self._mutate(c1, mr1)
                if hasattr(self.problem, "repair"):
                    c1 = self.problem.repair(c1)
                f_c1 = float(self.problem.evaluate(c1))

                # Récompense de l'enfant 1 (minimisation)
                r1 = float(np.clip((f_moy_parents - f_c1) / (f_range + eps_denom), -1.0, 1.0))
                u1 = float((r1 + 1.0) / 2.0)

                children.append(c1)
                children_fitness.append(f_c1)
                children_payoffs.append((idx1_1, idx2_1, u1))

                # --- Enfant 2 (si budget restant dans la génération) ---
                if len(children) < self.pop_size:
                    idx1_2, idx2_2, cr2, mr2 = self.game.sample_strategies(self.rng)
                    c2, _ = self._crossover(p2, p1, cr2)
                    c2 = self._mutate(c2, mr2)
                    if hasattr(self.problem, "repair"):
                        c2 = self.problem.repair(c2)
                    f_c2 = float(self.problem.evaluate(c2))

                    # Récompense de l'enfant 2 (minimisation)
                    r2 = float(np.clip((f_moy_parents - f_c2) / (f_range + eps_denom), -1.0, 1.0))
                    u2 = float((r2 + 1.0) / 2.0)

                    children.append(c2)
                    children_fitness.append(f_c2)
                    children_payoffs.append((idx1_2, idx2_2, u2))

            new_pop = np.array(children[:self.pop_size])
            new_fitness = np.array(children_fitness[:self.pop_size])

            # Élitisme : préserve le meilleur individu précédent sans évaluation supplémentaire
            worst_idx = np.argmax(new_fitness)
            best_old_idx = np.argmin(fitness)
            if fitness[best_old_idx] < new_fitness[worst_idx]:
                new_pop[worst_idx] = pop[best_old_idx].copy()
                new_fitness[worst_idx] = fitness[best_old_idx]

            # Mise à jour du composant Théorie des Jeux (U, U_norm, σ₁, σ₂, Nash gap)
            game_diag = self.game.update_generation(children_payoffs)

            # Remplacement de la population
            pop = new_pop
            fitness = new_fitness

            # Enregistrement des statistiques
            best_idx = np.argmin(fitness)
            best_f = float(fitness[best_idx])
            mean_f = float(fitness.mean())

            self.best_fitness_history.append(best_f)
            self.mean_fitness_history.append(mean_f)
            self.strategy_history.append(self.game.get_strategy_proportions())
            self.nash_gap_history.append(game_diag.get("nash_gap", 0.0))
            self.pure_nash_history.append(game_diag.get("pure_nash_equilibria", []))

            if best_f < self.best_fitness:
                self.best_fitness = best_f
                self.best_solution = pop[best_idx].copy()

        return {
            "best_fitness": self.best_fitness,
            "best_solution": self.best_solution,
            "best_history": self.best_fitness_history,
            "mean_history": self.mean_fitness_history,
            "strategy_history": self.strategy_history,
            "nash_gap_history": self.nash_gap_history,
            "pure_nash_history": self.pure_nash_history,
            "final_payoff_matrix": self.game.U.tolist(),
            "final_payoff_matrices": {
                "A": self.game.U.tolist(),
                "B": self.game.U.tolist(),
            },
            "final_normalized_payoff_matrix": self.game.get_normalized_payoff_matrix().tolist(),
            "final_strategies": self.game.get_strategy_proportions(),
            "evaluations": self.problem.evaluations_count,
        }
