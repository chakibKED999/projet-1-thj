"""
Tests unitaires pour la Phase 2 : Problèmes
===========================================
Vérifie :
  1. Lecture automatique de la taille depuis le nom du problème ('rastrigin-10', 'knapsack-50', etc.)
  2. Résolution exacte du Sac à dos par programmation dynamique (DP)
  3. Opérateur de réparation heuristique des solutions infaisables
  4. Génération déterministe à graine fixe pour Knapsack-50, 100, 200
  5. Intégration des opérateurs binaires natifs (croisement uniforme/1-point, bit-flip)
  6. Équité absolue du budget d'évaluations sur les problèmes de Phase 2
"""

import unittest
import numpy as np

from src.problems.benchmark import (
    get_problem,
    parse_problem_name,
    solve_knapsack_dp,
    ContinuousBenchmark,
    KnapsackProblem,
)
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.ga_nash import GANash


class TestProblemParsing(unittest.TestCase):
    """Test de l'analyseur de nom de problème get_problem()."""

    def test_parse_problem_name_variants(self):
        """Vérifie le parsing pour différents séparateurs et formats."""
        self.assertEqual(parse_problem_name("rastrigin-10"), ("rastrigin", 10))
        self.assertEqual(parse_problem_name("rosenbrock_10"), ("rosenbrock", 10))
        self.assertEqual(parse_problem_name("ackley-10"), ("ackley", 10))
        self.assertEqual(parse_problem_name("sphere-10"), ("sphere", 10))
        self.assertEqual(parse_problem_name("knapsack-50"), ("knapsack", 50))
        self.assertEqual(parse_problem_name("knapsack_100"), ("knapsack", 100))
        self.assertEqual(parse_problem_name("knapsack-200"), ("knapsack", 200))
        self.assertEqual(parse_problem_name("rastrigin"), ("rastrigin", None))

    def test_get_problem_continuous_instances(self):
        """Vérifie l'instanciation des fonctions continues en dimension 10."""
        for name in ["rastrigin-10", "rosenbrock-10", "ackley-10", "sphere-10"]:
            prob = get_problem(name)
            self.assertIsInstance(prob, ContinuousBenchmark)
            self.assertEqual(prob.dim, 10)
            self.assertFalse(prob.is_binary)
            self.assertEqual(prob.optimum, 0.0)

    def test_get_problem_knapsack_instances(self):
        """Vérifie l'instanciation des trois tailles de sac à dos (50, 100, 200)."""
        for n in [50, 100, 200]:
            prob = get_problem(f"knapsack-{n}")
            self.assertIsInstance(prob, KnapsackProblem)
            self.assertEqual(prob.dim, n)
            self.assertEqual(prob.n_items, n)
            self.assertTrue(prob.is_binary)
            self.assertIsNotNone(prob.optimum)
            self.assertLess(prob.optimum, 0.0)  # f(x) = -valeur en minimisation


class TestKnapsackDPAndRepair(unittest.TestCase):
    """Vérification de la programmation dynamique et du réparateur pour Knapsack."""

    def test_knapsack_dp_toy_instance(self):
        """Vérifie la DP sur un problème jouet à optimum analytique connu."""
        # 4 objets : poids [2, 3, 4, 5], valeurs [3, 4, 5, 6], capacité 5
        # Meilleure combinaison : objets 0 et 1 (poids 2+3=5, valeur 3+4=7)
        weights = np.array([2, 3, 4, 5])
        values = np.array([3, 4, 5, 6])
        cap = 5
        best_val, sol = solve_knapsack_dp(weights, values, cap)

        self.assertEqual(best_val, 7)
        self.assertLessEqual(int(sol @ weights), cap)
        self.assertEqual(int(sol @ values), 7)
        np.testing.assert_array_equal(sol, [1, 1, 0, 0])

    def test_knapsack_dp_exactness_50_100_200(self):
        """Vérifie la validité mathématique de la solution optimale pour 50, 100 et 200 objets."""
        for n in [50, 100, 200]:
            prob = get_problem(f"knapsack-{n}", seed=42)
            self.assertLessEqual(
                prob.optimum_solution @ prob.weights,
                prob.capacity,
                f"La solution DP de knapsack-{n} dépasse la capacité !",
            )
            self.assertEqual(
                prob.optimum_solution @ prob.values,
                prob.optimum_value,
                f"La valeur de la solution DP de knapsack-{n} ne correspond pas à optimum_value",
            )
            self.assertEqual(prob.optimum, -float(prob.optimum_value))

    def test_knapsack_fixed_seed_determinism(self):
        """Garantit que deux instances avec la même graine sont strictement identiques."""
        prob1 = KnapsackProblem(n_items=50, seed=42)
        prob2 = KnapsackProblem(n_items=50, seed=42)

        np.testing.assert_array_equal(prob1.weights, prob2.weights)
        np.testing.assert_array_equal(prob1.values, prob2.values)
        self.assertEqual(prob1.capacity, prob2.capacity)
        self.assertEqual(prob1.optimum_value, prob2.optimum_value)
        np.testing.assert_array_equal(prob1.optimum_solution, prob2.optimum_solution)

    def test_repair_operator_makes_infeasible_feasible(self):
        """Le réparateur doit rendre faisable tout vecteur infaisable."""
        prob = get_problem("knapsack-50", seed=42)

        # Vecteur infaisable : tous les objets à 1
        all_ones = np.ones(prob.n_items)
        self.assertGreater(all_ones @ prob.weights, prob.capacity)

        repaired = prob.repair(all_ones)
        self.assertLessEqual(repaired @ prob.weights, prob.capacity)
        # Vérifie que les valeurs sont binaires (0 ou 1)
        self.assertTrue(set(np.unique(repaired)).issubset({0.0, 1.0}))

    def test_repair_operator_preserves_feasible(self):
        """Le réparateur ne doit pas altérer une solution déjà faisable."""
        prob = get_problem("knapsack-50", seed=42)
        feasible = np.zeros(prob.n_items)
        feasible[0] = 1.0  # un seul objet respecte toujours la capacité

        repaired = prob.repair(feasible)
        np.testing.assert_array_equal(repaired, feasible)


class TestBinaryGAOperators(unittest.TestCase):
    """Vérification des opérateurs génétiques binaires natifs."""

    def test_ga_binary_uniform_crossover(self):
        """Le croisement uniforme binaire produit des enfants binaires."""
        prob = get_problem("knapsack-50", seed=42)
        ga = GeneticAlgorithm(problem=prob, crossover_type="uniform", seed=10)

        p1 = np.ones(prob.dim)
        p2 = np.zeros(prob.dim)
        c1, c2 = ga._crossover(p1, p2, cr=1.0)

        self.assertTrue(set(np.unique(c1)).issubset({0.0, 1.0}))
        self.assertTrue(set(np.unique(c2)).issubset({0.0, 1.0}))

    def test_ga_binary_single_point_crossover(self):
        """Le croisement 1-point binaire produit des enfants binaires."""
        prob = get_problem("knapsack-50", seed=42)
        ga = GeneticAlgorithm(problem=prob, crossover_type="single_point", seed=10)

        p1 = np.ones(prob.dim)
        p2 = np.zeros(prob.dim)
        c1, c2 = ga._crossover(p1, p2, cr=1.0)

        self.assertTrue(set(np.unique(c1)).issubset({0.0, 1.0}))
        self.assertTrue(set(np.unique(c2)).issubset({0.0, 1.0}))

    def test_ga_binary_mutation_bitflip(self):
        """La mutation bit-flip inverse les bits avec la probabilité voulue."""
        prob = get_problem("knapsack-50", seed=42)
        ga = GeneticAlgorithm(problem=prob, seed=10)

        ind = np.zeros(prob.dim)
        # mr=1.0 doit inverser tous les bits
        mutated = ga._mutate(ind, mr=1.0)
        np.testing.assert_array_equal(mutated, np.ones(prob.dim))

    def test_ga_on_knapsack_produces_feasible_best_solution(self):
        """Une exécution complète du GA sur knapsack-50 produit une solution réalisable."""
        prob = get_problem("knapsack-50", seed=42)
        ga = GeneticAlgorithm(problem=prob, pop_size=20, max_gen=30, seed=123)
        res = ga.run()

        sol = res["best_solution"]
        w = int(sol @ prob.weights)
        v = int(sol @ prob.values)

        self.assertLessEqual(w, prob.capacity)
        self.assertEqual(res["best_fitness"], -float(v))
        self.assertGreater(v, 0)
        # Vérifie qu'on est à moins de 25% de l'optimum en seulement 30 générations
        self.assertGreater(v, 0.75 * prob.optimum_value)

    def test_identical_evaluations_knapsack(self):
        """GA et GA-Nash consomment exactement le même budget sur le sac à dos."""
        pop_size = 20
        max_gen = 25
        expected_evals = pop_size * (max_gen + 1)

        prob_ga = get_problem("knapsack-50", seed=42)
        ga = GeneticAlgorithm(prob_ga, pop_size=pop_size, max_gen=max_gen, seed=1)
        res_ga = ga.run()

        prob_nash = get_problem("knapsack-50", seed=42)
        nash = GANash(prob_nash, pop_size=pop_size, max_gen=max_gen, seed=1)
        res_nash = nash.run()

        self.assertEqual(res_ga["evaluations"], expected_evals)
        self.assertEqual(res_nash["evaluations"], expected_evals)
        self.assertEqual(prob_ga.evaluations_count, expected_evals)
        self.assertEqual(prob_nash.evaluations_count, expected_evals)


if __name__ == "__main__":
    unittest.main()
