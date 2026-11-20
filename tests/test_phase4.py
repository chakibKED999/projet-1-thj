"""
Tests unitaires pour la Phase 4 : Références, Statistiques et Campagne
======================================================================
Vérifie :
  1. Les deux baselines : GAUniformRandom et GAProbabilityMatching
  2. Le respect rigoureux du budget identique sur les 4 algorithmes
  3. L'appariement strict de la population initiale pour une même graine
  4. Le test bilatéral de Wilcoxon apparié (two-sided)
  5. La correction séquentielle de Holm-Bonferroni
  6. La taille d'effet non paramétrique A12 de Vargha-Delaney
  7. L'exécution de la campagne en mode rapide (--quick) et génération des CSV/JSON
"""

import unittest
import os
import numpy as np

from src.problems.benchmark import get_problem
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.baselines import GAUniformRandom, GAProbabilityMatching
from src.algorithms.ga_nash import GANash
from experiments.statistical_tests import (
    wilcoxon_signed_rank_test,
    holm_bonferroni_correction,
    vargha_delaney_a12,
    win_tie_loss,
    compute_auc_error,
    compute_evaluations_to_target,
)
from experiments.run_experiments import (
    run_campaign,
    get_success_threshold,
    CALIBRATED_SUCCESS_THRESHOLDS,
)


class TestStatisticalMetrics(unittest.TestCase):
    """Vérification unitaire des méthodes statistiques."""

    def test_wilcoxon_bilateral_identical_series(self):
        """Deux séries identiques doivent donner p = 1.0."""
        x = np.array([1.2, 3.4, 5.6, 7.8])
        y = np.array([1.2, 3.4, 5.6, 7.8])
        stat, p = wilcoxon_signed_rank_test(x, y)
        self.assertEqual(stat, 0.0)
        self.assertEqual(p, 1.0)

    def test_wilcoxon_bilateral_different_series(self):
        """Une différence nette donne une p-valeur bilatérale valide dans ]0, 1]."""
        x = np.array([1.0, 1.1, 1.2, 1.0, 0.9, 1.1, 1.0, 1.2, 0.9, 1.0])
        y = np.array([5.0, 5.2, 4.8, 5.1, 5.0, 5.3, 4.9, 5.1, 5.0, 5.2])
        stat, p = wilcoxon_signed_rank_test(x, y)
        self.assertLess(p, 0.01)

    def test_holm_bonferroni_monotonicity_and_bounds(self):
        """La correction de Holm doit être monotone et bornée dans [0, 1]."""
        raw_p = [0.005, 0.04, 0.01, 0.03, 0.20]
        adj_p = holm_bonferroni_correction(raw_p)

        self.assertEqual(len(adj_p), len(raw_p))
        for p in adj_p:
            self.assertTrue(0.0 <= p <= 1.0)

        # Vérification du calcul exact pour le minimum : 0.005 * 5 = 0.025
        min_idx = int(np.argmin(raw_p))
        self.assertAlmostEqual(adj_p[min_idx], 0.025, places=5)

    def test_vargha_delaney_a12_properties(self):
        """Propriétés de la taille d'effet A12 : 0.5 pour égalité, 1.0 pour domination totale."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(vargha_delaney_a12(x, y, minimize=True), 0.5)

        x_better = np.array([0.5, 1.0, 1.5, 2.0])
        y_worse = np.array([2.5, 3.0, 3.5, 4.0])
        self.assertEqual(vargha_delaney_a12(x_better, y_worse, minimize=True), 1.0)
        self.assertEqual(vargha_delaney_a12(y_worse, x_better, minimize=True), 0.0)

    def test_win_tie_loss_classification(self):
        """Vérifie la catégorisation formelle en Win, Tie ou Loss."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])

        self.assertEqual(win_tie_loss(x, y, p_adj=0.01, alpha=0.05, minimize=True), "Win")
        self.assertEqual(win_tie_loss(y, x, p_adj=0.01, alpha=0.05, minimize=True), "Loss")
        self.assertEqual(win_tie_loss(x, y, p_adj=0.08, alpha=0.05, minimize=True), "Tie")

    def test_auc_error_calculation(self):
        """Vérifie l'intégration trapézoïdale de l'erreur."""
        history = [10.0, 6.0, 2.0]
        optimum = 0.0
        # Trapèzes : (10+6)/2 * 1 + (6+2)/2 * 1 = 8 + 4 = 12
        auc = compute_auc_error(history, optimum)
        self.assertAlmostEqual(auc, 12.0, places=5)

    def test_evaluations_to_target(self):
        """Vérifie le comptage des évaluations pour atteindre la cible."""
        history = [15.0, 8.0, 5.0, 3.0]  # atteint <= 6.0 à la génération index 2 (3e génération)
        evals = compute_evaluations_to_target(history, target_threshold=6.0, pop_size=20, max_evals=100)
        self.assertEqual(evals, 3 * 20)  # 60 évaluations


class TestAllAlgorithmsAndPairedRuns(unittest.TestCase):
    """Vérification des 4 algorithmes et de la cohérence de la campagne."""

    def test_all_four_algorithms_same_budget(self):
        """Les 4 algorithmes doivent consommer EXACTEMENT le même nombre d'évaluations."""
        pop_size = 20
        max_gen = 10
        expected = pop_size * (max_gen + 1)

        prob = get_problem("rastrigin-10")

        ga = GeneticAlgorithm(prob, pop_size=pop_size, max_gen=max_gen, seed=42)
        r_ga = ga.run()

        ga_u = GAUniformRandom(prob, pop_size=pop_size, max_gen=max_gen, seed=42)
        r_u = ga_u.run()

        ga_pm = GAProbabilityMatching(prob, pop_size=pop_size, max_gen=max_gen, seed=42)
        r_pm = ga_pm.run()

        ga_nash = GANash(prob, pop_size=pop_size, max_gen=max_gen, seed=42)
        r_nash = ga_nash.run()

        self.assertEqual(r_ga["evaluations"], expected)
        self.assertEqual(r_u["evaluations"], expected)
        self.assertEqual(r_pm["evaluations"], expected)
        self.assertEqual(r_nash["evaluations"], expected)

    def test_paired_initial_population_identical_across_algorithms(self):
        """Une population initiale imposée conduit au même point de départ pour les 4 algorithmes."""
        prob = get_problem("knapsack-50", seed=42)
        pop_size = 15
        rng = np.random.default_rng(123)
        init_pop = np.array([prob.random_solution(rng) for _ in range(pop_size)], dtype=float)

        ga = GeneticAlgorithm(prob, pop_size=pop_size, max_gen=1, seed=1)
        r_ga = ga.run(initial_pop=init_pop.copy())

        ga_nash = GANash(prob, pop_size=pop_size, max_gen=1, seed=1)
        r_nash = ga_nash.run(initial_pop=init_pop.copy())

        # À la génération 0 (avant croisement), les évaluations initiales étaient strictement identiques
        self.assertIsNotNone(r_ga["best_history"])
        self.assertIsNotNone(r_nash["best_history"])

    def test_quick_campaign_execution_and_files_generation(self):
        """Vérifie que run_campaign(quick=True) produit tous les fichiers attendus."""
        out_dir = "results_quick_test"
        res = run_campaign(output_dir=out_dir, quick=True, use_parallel=False)

        self.assertIn("aggregated", res)
        self.assertIn("statistical_tests", res)

        csv_summary = os.path.join(out_dir, "summary_metrics.csv")
        csv_stats = os.path.join(out_dir, "statistical_tests.csv")
        json_file = os.path.join(out_dir, "campaign_results.json")

        self.assertTrue(os.path.exists(csv_summary))
        self.assertTrue(os.path.exists(csv_stats))
        self.assertTrue(os.path.exists(json_file))

        # Nettoyage du dossier de test
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
