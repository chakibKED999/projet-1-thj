"""
Tests unitaires pour la Phase 1 : Hygiène et reproductibilité
============================================================
Vérifie :
  1. Comptage rigoureux des évaluations (BaseProblem.evaluations_count)
  2. Égalité stricte du budget d'évaluations entre GA et GA-Nash
  3. Reproductibilité exacte (même graine => résultats strictement identiques)
  4. Isolation du générateur aléatoire (aucun recours à np.random.* global)
"""

import unittest
import numpy as np

from src.problems.benchmark import get_problem, ContinuousBenchmark, KnapsackProblem
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.ga_nash import GANash
from src.game_theory.nash_game import OperatorGame


class TestEvaluationsBudget(unittest.TestCase):
    """Vérification du compteur d'évaluations et de l'équité des budgets."""

    def test_problem_evaluations_counter_increments(self):
        """Le compteur s'incrémente à chaque évaluation et se réinitialise correctement."""
        prob = get_problem("rastrigin", dim=5)
        self.assertEqual(prob.evaluations_count, 0)

        rng = np.random.default_rng(123)
        for i in range(1, 15):
            x = prob.random_solution(rng)
            prob.evaluate(x)
            self.assertEqual(prob.evaluations_count, i)

        prob.reset_evaluations_count()
        self.assertEqual(prob.evaluations_count, 0)

    def test_identical_evaluation_budget_between_algorithms(self):
        """GA et GA-Nash doivent consommer exactement le même nombre d'évaluations."""
        pop_size = 25
        max_gen = 30
        expected_evals = pop_size * (max_gen + 1)  # population initiale + max_gen enfants

        prob_ga = get_problem("rastrigin", dim=5)
        ga = GeneticAlgorithm(
            problem=prob_ga,
            pop_size=pop_size,
            max_gen=max_gen,
            seed=42,
        )
        res_ga = ga.run()

        prob_nash = get_problem("rastrigin", dim=5)
        ga_nash = GANash(
            problem=prob_nash,
            pop_size=pop_size,
            max_gen=max_gen,
            seed=42,
        )
        res_nash = ga_nash.run()

        # Vérification sur l'attribut du problème
        self.assertEqual(prob_ga.evaluations_count, expected_evals)
        self.assertEqual(prob_nash.evaluations_count, expected_evals)

        # Vérification sur le dictionnaire retourné
        self.assertEqual(res_ga["evaluations"], expected_evals)
        self.assertEqual(res_nash["evaluations"], expected_evals)
        self.assertEqual(res_ga["evaluations"], res_nash["evaluations"])


class TestReproducibility(unittest.TestCase):
    """Vérification de la reproductibilité stricte avec np.random.Generator."""

    def test_ga_reproducibility(self):
        """Une même graine doit produire exactement les mêmes résultats pour le GA standard."""
        prob1 = get_problem("sphere", dim=5)
        ga1 = GeneticAlgorithm(problem=prob1, pop_size=20, max_gen=25, seed=12345)
        res1 = ga1.run()

        prob2 = get_problem("sphere", dim=5)
        ga2 = GeneticAlgorithm(problem=prob2, pop_size=20, max_gen=25, seed=12345)
        res2 = ga2.run()

        self.assertEqual(res1["best_fitness"], res2["best_fitness"])
        np.testing.assert_array_equal(res1["best_solution"], res2["best_solution"])
        self.assertEqual(res1["best_history"], res2["best_history"])
        self.assertEqual(res1["mean_history"], res2["mean_history"])

    def test_ga_nash_reproducibility(self):
        """Une même graine doit produire exactement les mêmes résultats pour GA-Nash."""
        prob1 = get_problem("ackley", dim=5)
        nash1 = GANash(problem=prob1, pop_size=20, max_gen=25, seed=777)
        res1 = nash1.run()

        prob2 = get_problem("ackley", dim=5)
        nash2 = GANash(problem=prob2, pop_size=20, max_gen=25, seed=777)
        res2 = nash2.run()

        self.assertEqual(res1["best_fitness"], res2["best_fitness"])
        np.testing.assert_array_equal(res1["best_solution"], res2["best_solution"])
        self.assertEqual(res1["best_history"], res2["best_history"])
        self.assertEqual(res1["strategy_history"], res2["strategy_history"])
        self.assertEqual(res1["final_payoff_matrices"], res2["final_payoff_matrices"])

    def test_different_seeds_produce_different_trajectories(self):
        """Deux graines différentes doivent conduire à des trajectoires différentes."""
        prob1 = get_problem("rastrigin", dim=5)
        ga1 = GeneticAlgorithm(problem=prob1, pop_size=20, max_gen=25, seed=1)
        res1 = ga1.run()

        prob2 = get_problem("rastrigin", dim=5)
        ga2 = GeneticAlgorithm(problem=prob2, pop_size=20, max_gen=25, seed=2)
        res2 = ga2.run()

        self.assertNotEqual(res1["best_fitness"], res2["best_fitness"])
        self.assertFalse(np.array_equal(res1["best_solution"], res2["best_solution"]))

    def test_operator_game_generator_isolation(self):
        """OperatorGame utilise le Generator passé et ne dépend pas du random global."""
        # Deux jeux avec deux générateurs indépendants initialisés à la même graine
        rng1 = np.random.default_rng(999)
        rng2 = np.random.default_rng(999)

        game1 = OperatorGame(rng=rng1)
        game2 = OperatorGame(rng=rng2)

        # Polluer volontairement l'état global de numpy
        np.random.seed(0)
        _ = np.random.random(100)

        samples1 = [game1.sample_strategies(rng1) for _ in range(10)]
        samples2 = [game2.sample_strategies(rng2) for _ in range(10)]

        self.assertEqual(samples1, samples2)


if __name__ == "__main__":
    unittest.main()
