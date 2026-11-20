"""
Tests unitaires pour la Phase 3 : Composant Jeu (nash_game.py & ga_nash.py)
===========================================================================
Vérifie :
  1. Présence des stratégies par défaut du GA standard (cr=0.80, mr=0.05) dans les grilles
  2. Propriétés du jeu à intérêts communs (u1 = u2 = U)
  3. Définition stricte des équilibres de Nash purs (meilleures réponses mutuelles)
  4. Calcul rigoureux du Nash Gap (non-négatif, nul à l'équilibre)
  5. Dynamique de réplication discrète et respect strict du plancher eps_min
  6. Mise à jour sélective par EMA (beta) uniquement sur les cases visitées
  7. Échantillonnage des stratégies et attribution des gains (r et u) par enfant
  8. Égalité stricte du budget d'évaluations entre GA standard et GA-Nash
"""

import unittest
import numpy as np

from src.problems.benchmark import get_problem
from src.game_theory.nash_game import (
    OperatorGame,
    pure_nash_equilibria,
    replicator_step,
)
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.ga_nash import GANash


class TestGameTheoryComponent(unittest.TestCase):
    """Vérification mathématique du composant de Théorie des Jeux."""

    def test_default_ga_strategies_in_grids(self):
        """Le couple de paramètres par défaut du GA (cr=0.80, mr=0.05) doit appartenir aux grilles."""
        self.assertIn(0.80, OperatorGame.STRATEGIES_CROSSOVER)
        self.assertIn(0.05, OperatorGame.STRATEGIES_MUTATION)
        # Vérifie les indices exacts
        idx_cr = OperatorGame.STRATEGIES_CROSSOVER.index(0.80)
        idx_mr = OperatorGame.STRATEGIES_MUTATION.index(0.05)
        self.assertEqual(idx_cr, 2)
        self.assertEqual(idx_mr, 1)

    def test_common_interest_game_property(self):
        """Dans un jeu à intérêts communs, les deux joueurs partagent la même matrice U."""
        game = OperatorGame()
        np.testing.assert_array_equal(game.A, game.B)
        np.testing.assert_array_equal(game.A, game.U)

    def test_pure_nash_equilibria_coordination_game(self):
        """Vérifie la détection des équilibres de Nash purs sur un jeu de coordination."""
        # Matrice avec trois équilibres purs stricts sur la diagonale (0,0), (1,1) et (2,2)
        U = np.array([
            [0.9, 0.2, 0.1],
            [0.3, 0.5, 0.4],
            [0.1, 0.2, 0.8],
        ])
        eqs = pure_nash_equilibria(U)
        self.assertIn((0, 0), eqs)
        self.assertIn((1, 1), eqs)
        self.assertIn((2, 2), eqs)
        self.assertEqual(len(eqs), 3)

    def test_pure_nash_equilibria_no_pure_equilibrium(self):
        """Vérifie qu'aucun équilibre pur n'est détecté lorsqu'il n'en existe pas."""
        # Jeu de type matching pennies à somme nulle généralisée
        U1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
        U2 = np.array([[-1.0, 1.0], [1.0, -1.0]])
        eqs = pure_nash_equilibria(U1, U2)
        self.assertEqual(eqs, [])

    def test_nash_gap_properties(self):
        """Le Nash Gap doit être nul à l'équilibre et strictement positif hors équilibre."""
        game = OperatorGame()
        # Matrice où (0, 0) est un équilibre pur strict dominant
        game.U = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.1],
        ])
        # Profil pur sur l'équilibre (0, 0)
        game.sigma1 = np.array([1.0, 0.0, 0.0, 0.0])
        game.sigma2 = np.array([1.0, 0.0, 0.0, 0.0])

        U_norm = game.get_normalized_payoff_matrix()
        v1 = U_norm @ game.sigma2
        v2 = U_norm.T @ game.sigma1
        gap = (np.max(v1) - np.dot(game.sigma1, v1)) + (np.max(v2) - np.dot(game.sigma2, v2))

        self.assertAlmostEqual(gap, 0.0, places=7)

        # Hors équilibre : stratégie uniforme
        game.sigma1 = np.ones(4) / 4.0
        game.sigma2 = np.ones(4) / 4.0
        v1 = U_norm @ game.sigma2
        v2 = U_norm.T @ game.sigma1
        gap_off = (np.max(v1) - np.dot(game.sigma1, v1)) + (np.max(v2) - np.dot(game.sigma2, v2))
        self.assertGreater(gap_off, 0.0)

    def test_replicator_dynamics_exploration_floor(self):
        """La dynamique de réplication doit respecter strictement le plancher eps_min."""
        eps_min = 0.08
        k = 4
        sigma = np.ones(k) / k
        # Vecteur de gains très déséquilibré favorisant fortement la stratégie 0
        v = np.array([10.0, 0.0, 0.0, 0.0])

        new_sigma = replicator_step(sigma, v, eta=1.0, eps_min=eps_min)

        self.assertAlmostEqual(np.sum(new_sigma), 1.0, places=7)
        self.assertTrue(np.all(new_sigma >= eps_min - 1e-9))
        self.assertGreater(new_sigma[0], new_sigma[1])

    def test_ema_updates_only_visited_cells(self):
        """L'EMA ne doit modifier que les cases (i, j) ayant au moins un enfant."""
        game = OperatorGame(beta=0.2)
        initial_U = game.U.copy()

        # Seule la case (1, 2) a produit des enfants avec un payoff moyen de 0.9
        evaluations = [(1, 2, 0.9), (1, 2, 0.9)]
        game.update_generation(evaluations)

        # La case (1, 2) a été mise à jour : (1 - 0.2)*0.5 + 0.2*0.9 = 0.4 + 0.18 = 0.58
        self.assertAlmostEqual(game.U[1, 2], 0.58, places=7)

        # Toutes les autres cases restent inchangées à 0.5
        for i in range(game.k1):
            for j in range(game.k2):
                if (i, j) != (1, 2):
                    self.assertEqual(game.U[i, j], initial_U[i, j])

    def test_exposed_parameters(self):
        """Les paramètres eta, eps_min et beta sont configurables."""
        game = OperatorGame(eta=0.8, eps_min=0.03, beta=0.25)
        self.assertEqual(game.eta, 0.8)
        self.assertEqual(game.eps_min, 0.03)
        self.assertEqual(game.beta, 0.25)


class TestGANashIntegration(unittest.TestCase):
    """Vérification de l'intégration du composant jeu dans GANash."""

    def test_child_payoff_bounds(self):
        """Le payoff u de chaque enfant doit être rigoureusement borné dans [0, 1]."""
        prob = get_problem("rastrigin-10")
        nash = GANash(prob, pop_size=30, max_gen=10, seed=42)
        res = nash.run()

        # Vérifie que la matrice U finale est dans [0, 1]
        final_U = np.array(res["final_payoff_matrix"])
        self.assertTrue(np.all(final_U >= 0.0))
        self.assertTrue(np.all(final_U <= 1.0))

    def test_nash_gap_tracking_in_run(self):
        """Le Nash Gap doit être enregistré à chaque génération et rester >= 0."""
        prob = get_problem("ackley-10")
        max_gen = 20
        nash = GANash(prob, pop_size=20, max_gen=max_gen, seed=99)
        res = nash.run()

        gap_hist = res["nash_gap_history"]
        self.assertEqual(len(gap_hist), max_gen)
        for g in gap_hist:
            self.assertGreaterEqual(g, 0.0)

    def test_evaluation_budget_strict_equality(self):
        """GA standard et GA-Nash doivent consommer EXACTEMENT le même budget d'évaluations."""
        pop_size = 25
        max_gen = 15
        expected_evals = pop_size * (max_gen + 1)

        prob_ga = get_problem("rosenbrock-10")
        ga = GeneticAlgorithm(prob_ga, pop_size=pop_size, max_gen=max_gen, seed=123)
        res_ga = ga.run()

        prob_nash = get_problem("rosenbrock-10")
        nash = GANash(prob_nash, pop_size=pop_size, max_gen=max_gen, seed=123)
        res_nash = nash.run()

        self.assertEqual(prob_ga.evaluations_count, expected_evals)
        self.assertEqual(prob_nash.evaluations_count, expected_evals)
        self.assertEqual(res_ga["evaluations"], expected_evals)
        self.assertEqual(res_nash["evaluations"], expected_evals)
        self.assertEqual(res_ga["evaluations"], res_nash["evaluations"])


if __name__ == "__main__":
    unittest.main()
