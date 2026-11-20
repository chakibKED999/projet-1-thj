"""
Module : Expériences et Évaluation Expérimentale Complète
==========================================================
Partie E et F du projet :
  - 4 Algorithmes comparés :
      * GA standard (Baseline à paramètres fixes cr=0.80, mr=0.05)
      * GA uniforme sans apprentissage (cr et mr tirés uniformément par enfant)
      * GA Probability Matching (AOS adaptatif sans théorie des jeux)
      * GA-Nash (Jeu à intérêts communs avec dynamique de réplication)
  - 7 Instances de test :
      * Rastrigin-10, Rosenbrock-10, Ackley-10, Sphere-10 (test de cohérence)
      * Knapsack-50, Knapsack-100, Knapsack-200 (optimum exact par programmation dynamique)
  - 30 exécutions indépendantes par algorithme et par instance avec graines appariées
    et populations initiales strictement identiques.
  - Métriques d'évaluation :
      * Meilleure valeur (best), Moyenne (mean), Écart-type (std), Médiane (median)
      * Taux de succès (success_rate) selon seuils étalonnés préalablement sur graines >= 1000
        (et écart relatif <= 1% à l'optimum DP pour le sac à dos)
      * AUC de la courbe d'erreur (effort cumulé)
      * Évaluations pour atteindre la cible (AES)
      * Temps CPU en secondes (time.process_time)
  - Analyse statistique rigoureuse :
      * Test de Wilcoxon des rangs signés apparié BILATÉRAL (two-sided)
      * Correction séquentielle de Holm (Holm-Bonferroni) pour tests multiples
      * Taille d'effet A12 de Vargha-Delaney
      * Tableau récapitulatif Gagne / Égalité / Perd (Win / Tie / Loss)
  - Étude de sensibilité multi-paramétrique (eta, eps_min, beta)
  - Parallélisation (multiprocessing) et option d'exécution rapide (--quick)
  - Exportation systématique des résultats en CSV et JSON.
"""

import os
import sys
import time
import json
import csv
import argparse
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# Inclusion du chemin racine pour exécution directe
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problems.benchmark import get_problem, BaseProblem, AVAILABLE_PROBLEMS
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.ga_nash import GANash
from src.algorithms.baselines import GAUniformRandom, GAProbabilityMatching
from experiments.statistical_tests import (
    wilcoxon_signed_rank_test,
    holm_bonferroni_correction,
    vargha_delaney_a12,
    win_tie_loss,
    compute_auc_error,
    compute_evaluations_to_target,
)


# ===========================================================================
# Seuils de succès étalonnés par étude pilote (graines >= 1000)
# ===========================================================================
# Note : Pour les fonctions continues, les seuils sont choisis d'après le pilote
# de sorte que le GA de base obtienne un taux de succès intermédiaire (~40-50%).
# Pour le sac à dos, l'exigence formelle est un écart relatif <= 1% à l'optimum DP.

CALIBRATED_SUCCESS_THRESHOLDS: Dict[str, float] = {
    "sphere-10": 0.0010,
    "rastrigin-10": 6.0,
    "rosenbrock-10": 7.4,
    "ackley-10": 0.40,
}


def get_success_threshold(problem: BaseProblem) -> float:
    """Retourne le seuil de succès formel pour le problème considéré."""
    prob_name = problem.name.lower()
    if prob_name in CALIBRATED_SUCCESS_THRESHOLDS:
        return CALIBRATED_SUCCESS_THRESHOLDS[prob_name]
    elif prob_name.startswith("knapsack"):
        # Écart relatif <= 1% à l'optimum DP : fitness <= -0.99 * optimum_value
        opt_val = getattr(problem, "optimum_value", None)
        if opt_val is not None:
            return -0.99 * float(opt_val)
        return float(problem.optimum) if problem.optimum is not None else 0.0
    return 0.0


# ===========================================================================
# Exécution unitaire d'un algorithme pour une graine donnée
# ===========================================================================

def run_single_experiment_task(
    algo_name: str,
    prob_name: str,
    seed: int,
    pop_size: int,
    max_gen: int,
    initial_pop: np.ndarray,
    algo_kwargs: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Exécute un seul run d'un algorithme sur un problème avec population initiale imposée.
    Cette fonction est au niveau module pour être sérialisable avec multiprocessing.
    """
    if algo_kwargs is None:
        algo_kwargs = {}

    algo_map = {
        "GA_standard": GeneticAlgorithm,
        "GA_uniform": GAUniformRandom,
        "GA_probability_matching": GAProbabilityMatching,
        "GA_nash": GANash,
    }
    algo_cls = algo_map[algo_name]

    problem = get_problem(prob_name)
    threshold = get_success_threshold(problem)

    algo = algo_cls(
        problem=problem,
        pop_size=pop_size,
        max_gen=max_gen,
        seed=seed,
        **algo_kwargs,
    )

    # Mesure stricte du temps CPU du processus
    t_cpu_start = time.process_time()
    result = algo.run(initial_pop=initial_pop)
    cpu_time = time.process_time() - t_cpu_start

    best_fitness = float(result["best_fitness"])
    best_history = [float(f) for f in result["best_history"]]
    mean_history = [float(f) for f in result["mean_history"]]

    # Optimum de référence pour le calcul de l'erreur
    optimum = float(problem.optimum) if problem.optimum is not None else 0.0
    auc = compute_auc_error(best_history, optimum)

    max_evals = pop_size * (max_gen + 1)
    evals_to_target = compute_evaluations_to_target(
        best_history=best_history,
        target_threshold=threshold,
        pop_size=pop_size,
        max_evals=max_evals,
    )
    is_success = bool(best_fitness <= threshold)

    out = {
        "algo_name": algo_name,
        "prob_name": prob_name,
        "seed": seed,
        "best_fitness": best_fitness,
        "best_history": best_history,
        "mean_history": mean_history,
        "evaluations": result["evaluations"],
        "evaluations_to_target": evals_to_target,
        "auc_error": auc,
        "cpu_time": cpu_time,
        "is_success": is_success,
    }

    # Données spécifiques à GA-Nash
    if algo_name == "GA_nash":
        out["strategy_history"] = result.get("strategy_history", [])
        out["nash_gap_history"] = result.get("nash_gap_history", [])
        out["pure_nash_history"] = result.get("pure_nash_history", [])
        out["final_payoff_matrix"] = result.get("final_payoff_matrix", [])

    return out


# ===========================================================================
# Campagne Complète
# ===========================================================================

def run_campaign(
    problems: Optional[List[str]] = None,
    n_runs: int = 30,
    pop_size: int = 50,
    max_gen: int = 200,
    seed_base: int = 42,
    output_dir: str = "results",
    use_parallel: bool = True,
    quick: bool = False,
) -> Dict[str, Any]:
    """
    Exécute la campagne complète d'expériences comparatives.

    Paramètres
    ----------
    problems : List[str]
        Liste des instances à évaluer.
    n_runs : int
        Nombre d'exécutions indépendantes par configuration (défaut 30).
    pop_size : int
        Taille de la population.
    max_gen : int
        Nombre de générations.
    seed_base : int
        Graine de départ (les runs utilisent seed_base + r).
    output_dir : str
        Dossier de sauvegarde des résultats.
    use_parallel : bool
        Activation du parallélisme multi-cœur.
    quick : bool
        Mode rapide pour vérification du pipeline.
    """
    if quick:
        problems = ["rastrigin-10", "knapsack-50"]
        n_runs = 3
        max_gen = 30
    elif problems is None:
        problems = [
            "rastrigin-10",
            "rosenbrock-10",
            "ackley-10",
            "sphere-10",
            "knapsack-50",
            "knapsack-100",
            "knapsack-200",
        ]

    os.makedirs(output_dir, exist_ok=True)
    algo_names = ["GA_standard", "GA_uniform", "GA_probability_matching", "GA_nash"]

    print("=" * 70)
    print(f"  Lancement de la campagne : {len(problems)} instances × {len(algo_names)} algorithmes × {n_runs} runs")
    print(f"  Budget par run : {pop_size * (max_gen + 1)} évaluations (pop_size={pop_size}, max_gen={max_gen})")
    print(f"  Parallélisme : {'Actif' if use_parallel else 'Inactif'}")
    print("=" * 70)

    # 1. Préparation de toutes les tâches avec appariement des populations initiales
    tasks = []
    # Dict pour stocker les résultats bruts : raw_results[prob_name][algo_name] = [run_0, ..., run_29]
    raw_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        p: {a: [] for a in algo_names} for p in problems
    }

    for prob_name in problems:
        prob_instance = get_problem(prob_name)
        for r in range(n_runs):
            seed = seed_base + r
            # Génération unique de la population initiale pour ce run et ce problème
            init_rng = np.random.default_rng(seed)
            if prob_instance.is_binary:
                init_pop = np.array([prob_instance.random_solution(init_rng)
                                     for _ in range(pop_size)], dtype=float)
            else:
                init_pop = init_rng.uniform(prob_instance.lb, prob_instance.ub,
                                            size=(pop_size, prob_instance.dim))

            for a_name in algo_names:
                tasks.append((a_name, prob_name, seed, pop_size, max_gen, init_pop.copy(), {}))

    # 2. Exécution des tâches (parallèle ou séquentielle)
    t_start = time.time()
    total_tasks = len(tasks)
    completed_count = 0

    if use_parallel and total_tasks > 4:
        max_workers = min(os.cpu_count() or 4, 8)
        print(f"  Exécution distribuée sur {max_workers} processus...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    run_single_experiment_task,
                    t[0], t[1], t[2], t[3], t[4], t[5], t[6]
                ): t for t in tasks
            }
            for future in as_completed(future_to_task):
                res = future.result()
                raw_results[res["prob_name"]][res["algo_name"]].append(res)
                completed_count += 1
                if completed_count % max(1, total_tasks // 10) == 0 or completed_count == total_tasks:
                    print(f"  Progression : {completed_count}/{total_tasks} runs terminés ({(completed_count/total_tasks)*100:.1f}%)")
    else:
        print("  Exécution séquentielle...")
        for t in tasks:
            res = run_single_experiment_task(t[0], t[1], t[2], t[3], t[4], t[5], t[6])
            raw_results[res["prob_name"]][res["algo_name"]].append(res)
            completed_count += 1
            if completed_count % max(1, total_tasks // 5) == 0:
                print(f"  Progression : {completed_count}/{total_tasks} runs terminés")

    elapsed_wall = time.time() - t_start
    print(f"\n  Campagne terminée en {elapsed_wall:.2f} s ({elapsed_wall/60:.2f} min).")

    # Trier les runs par graine pour garantir un ordre apparié strict
    for p in problems:
        for a in algo_names:
            raw_results[p][a].sort(key=lambda d: d["seed"])

    # 3. Calcul des métriques statistiques agrégées
    summary_metrics: List[Dict[str, Any]] = []
    aggregated_results: Dict[str, Any] = {}

    for prob_name in problems:
        aggregated_results[prob_name] = {}
        prob_obj = get_problem(prob_name)
        threshold = get_success_threshold(prob_obj)

        for a_name in algo_names:
            runs = raw_results[prob_name][a_name]
            bests = np.array([r["best_fitness"] for r in runs])
            times_cpu = np.array([r["cpu_time"] for r in runs])
            aucs = np.array([r["auc_error"] for r in runs])
            evals_target = np.array([r["evaluations_to_target"] for r in runs])
            successes = np.array([r["is_success"] for r in runs])

            # Trajectoire de convergence moyenne
            histories = [r["best_history"] for r in runs]
            mean_convergence = np.mean(histories, axis=0).tolist()
            std_convergence = np.std(histories, axis=0).tolist()

            agg = {
                "problem": prob_name,
                "algorithm": a_name,
                "n_runs": len(runs),
                "threshold": threshold,
                "best_min": float(np.min(bests)),
                "best_mean": float(np.mean(bests)),
                "best_std": float(np.std(bests)),
                "best_median": float(np.median(bests)),
                "success_rate": float(np.mean(successes) * 100.0),
                "auc_error_mean": float(np.mean(aucs)),
                "auc_error_std": float(np.std(aucs)),
                "evals_to_target_mean": float(np.mean(evals_target)),
                "cpu_time_mean": float(np.mean(times_cpu)),
                "cpu_time_std": float(np.std(times_cpu)),
                "all_best_fitnesses": bests.tolist(),
                "mean_convergence": mean_convergence,
                "std_convergence": std_convergence,
            }

            if a_name == "GA_nash":
                # Proportions moyennes des stratégies à travers les 30 runs
                strat_histories = [r["strategy_history"] for r in runs if "strategy_history" in r]
                if strat_histories and len(strat_histories[0]) > 0:
                    num_gens = len(strat_histories[0])
                    cr_probs_all = []
                    mr_probs_all = []
                    for h in strat_histories:
                        cr_probs_all.append([step["crossover_probs"] for step in h])
                        mr_probs_all.append([step["mutation_probs"] for step in h])
                    agg["mean_crossover_probs"] = np.mean(cr_probs_all, axis=0).tolist()
                    agg["std_crossover_probs"] = np.std(cr_probs_all, axis=0).tolist()
                    agg["mean_mutation_probs"] = np.mean(mr_probs_all, axis=0).tolist()
                    agg["std_mutation_probs"] = np.std(mr_probs_all, axis=0).tolist()

                # Nash gap moyen
                gap_histories = [r["nash_gap_history"] for r in runs if "nash_gap_history" in r]
                if gap_histories:
                    agg["mean_nash_gap"] = np.mean(gap_histories, axis=0).tolist()
                    agg["std_nash_gap"] = np.std(gap_histories, axis=0).tolist()

            aggregated_results[prob_name][a_name] = agg
            summary_metrics.append({
                "problem": prob_name,
                "algorithm": a_name,
                "best_min": f"{agg['best_min']:.4f}",
                "best_mean": f"{agg['best_mean']:.4f}",
                "best_std": f"{agg['best_std']:.4f}",
                "best_median": f"{agg['best_median']:.4f}",
                "success_rate": f"{agg['success_rate']:.1f}%",
                "auc_error": f"{agg['auc_error_mean']:.2f}",
                "evals_to_target": f"{agg['evals_to_target_mean']:.1f}",
                "cpu_time_s": f"{agg['cpu_time_mean']:.3f} ± {agg['cpu_time_std']:.3f}",
            })

    # 4. Tests Statistiques de Wilcoxon avec correction de Holm et A12
    # Comparaison systématique : GA-Nash vs les 3 baselines
    statistical_rows: List[Dict[str, Any]] = []
    comparators = ["GA_standard", "GA_uniform", "GA_probability_matching"]

    for comp in comparators:
        raw_p_values = []
        tests_data = []

        for prob_name in problems:
            nash_scores = np.array(aggregated_results[prob_name]["GA_nash"]["all_best_fitnesses"])
            comp_scores = np.array(aggregated_results[prob_name][comp]["all_best_fitnesses"])

            stat, p_val = wilcoxon_signed_rank_test(nash_scores, comp_scores)
            a12 = vargha_delaney_a12(nash_scores, comp_scores, minimize=True)
            raw_p_values.append(p_val)
            tests_data.append((prob_name, comp, stat, p_val, a12, nash_scores, comp_scores))

        # Correction de Holm pour cette famille de tests
        adj_p_values = holm_bonferroni_correction(raw_p_values)

        for i, (prob_name, comparator, stat, p_val, a12, ns, cs) in enumerate(tests_data):
            p_adj = adj_p_values[i]
            verdict = win_tie_loss(ns, cs, p_adj=p_adj, alpha=0.05, minimize=True)
            statistical_rows.append({
                "problem": prob_name,
                "method": "GA_nash",
                "comparator": comparator,
                "wilcoxon_stat": f"{stat:.1f}",
                "p_value_raw": f"{p_val:.4e}",
                "p_value_holm": f"{p_adj:.4e}",
                "a12_vargha_delaney": f"{a12:.3f}",
                "verdict": verdict,
            })

    # 5. Export des résultats en CSV
    csv_summary_path = os.path.join(output_dir, "summary_metrics.csv")
    with open(csv_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "problem", "algorithm", "best_min", "best_mean", "best_std",
            "best_median", "success_rate", "auc_error", "evals_to_target", "cpu_time_s"
        ])
        writer.writeheader()
        writer.writerows(summary_metrics)
    print(f"  → CSV Métriques sauvegardé : {csv_summary_path}")

    csv_stats_path = os.path.join(output_dir, "statistical_tests.csv")
    with open(csv_stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "problem", "method", "comparator", "wilcoxon_stat",
            "p_value_raw", "p_value_holm", "a12_vargha_delaney", "verdict"
        ])
        writer.writeheader()
        writer.writerows(statistical_rows)
    print(f"  → CSV Tests Statistiques sauvegardé : {csv_stats_path}")

    # 6. Export complet en JSON pour visualisations et rapports
    full_output = {
        "metadata": {
            "n_runs": n_runs,
            "pop_size": pop_size,
            "max_gen": max_gen,
            "seed_base": seed_base,
            "problems": problems,
            "algorithms": algo_names,
            "wall_clock_time_s": elapsed_wall,
        },
        "aggregated": aggregated_results,
        "statistical_tests": statistical_rows,
    }
    json_path = os.path.join(output_dir, "campaign_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"  → JSON Complet sauvegardé : {json_path}")

    return full_output


# ===========================================================================
# Étude de Sensibilité Multi-Paramétrique (eta, eps_min, beta)
# ===========================================================================

def run_sensitivity_analysis(
    problems: Optional[List[str]] = None,
    n_runs: int = 30,
    pop_size: int = 50,
    max_gen: int = 150,
    seed_base: int = 42,
    output_dir: str = "results",
    quick: bool = False,
) -> Dict[str, Any]:
    """
    Étudie la sensibilité de GA-Nash aux 3 hyperparamètres clés du jeu :
      - eta : taux d'apprentissage de la dynamique de réplication
      - eps_min : plancher d'exploration par stratégie
      - beta : facteur d'oubli EMA de la matrice de gains U
    """
    if quick:
        problems = ["rastrigin-10"]
        n_runs = 3
        max_gen = 30
        grid_eta = [0.1, 0.5, 1.0]
        grid_eps = [0.01, 0.05]
        grid_beta = [0.05, 0.1]
    elif problems is None:
        problems = ["rastrigin-10", "rosenbrock-10", "knapsack-50", "knapsack-100"]
        grid_eta = [0.1, 0.5, 1.0, 2.0]
        grid_eps = [0.01, 0.05, 0.10]
        grid_beta = [0.05, 0.1, 0.3]

    print("\n" + "=" * 70)
    print("  Lancement de l'étude de sensibilité multi-paramétrique (eta, eps_min, beta)")
    print("=" * 70)

    sensitivity_records = []
    sens_results: Dict[str, Any] = {"by_param": {}, "records": []}

    # Étude 1 : Variation de eta (avec eps_min=0.05, beta=0.1)
    for prob_name in problems:
        prob = get_problem(prob_name)
        for eta_val in grid_eta:
            bests = []
            for r in range(n_runs):
                s = seed_base + r
                init_rng = np.random.default_rng(s)
                if prob.is_binary:
                    init_pop = np.array([prob.random_solution(init_rng) for _ in range(pop_size)], dtype=float)
                else:
                    init_pop = init_rng.uniform(prob.lb, prob.ub, size=(pop_size, prob.dim))

                nash = GANash(prob, pop_size=pop_size, max_gen=max_gen, eta=eta_val, eps_min=0.05, beta=0.1, seed=s)
                res = nash.run(initial_pop=init_pop)
                bests.append(res["best_fitness"])

            bests = np.array(bests)
            sensitivity_records.append({
                "problem": prob_name,
                "param_tested": "eta",
                "param_value": eta_val,
                "fixed_params": "eps_min=0.05, beta=0.1",
                "mean_best": float(np.mean(bests)),
                "std_best": float(np.std(bests)),
                "median_best": float(np.median(bests)),
            })

    # Étude 2 : Variation de eps_min (avec eta=0.5, beta=0.1)
    for prob_name in problems:
        prob = get_problem(prob_name)
        for eps_val in grid_eps:
            bests = []
            for r in range(n_runs):
                s = seed_base + r
                init_rng = np.random.default_rng(s)
                if prob.is_binary:
                    init_pop = np.array([prob.random_solution(init_rng) for _ in range(pop_size)], dtype=float)
                else:
                    init_pop = init_rng.uniform(prob.lb, prob.ub, size=(pop_size, prob.dim))

                nash = GANash(prob, pop_size=pop_size, max_gen=max_gen, eta=0.5, eps_min=eps_val, beta=0.1, seed=s)
                res = nash.run(initial_pop=init_pop)
                bests.append(res["best_fitness"])

            bests = np.array(bests)
            sensitivity_records.append({
                "problem": prob_name,
                "param_tested": "eps_min",
                "param_value": eps_val,
                "fixed_params": "eta=0.5, beta=0.1",
                "mean_best": float(np.mean(bests)),
                "std_best": float(np.std(bests)),
                "median_best": float(np.median(bests)),
            })

    # Étude 3 : Variation de beta (avec eta=0.5, eps_min=0.05)
    for prob_name in problems:
        prob = get_problem(prob_name)
        for beta_val in grid_beta:
            bests = []
            for r in range(n_runs):
                s = seed_base + r
                init_rng = np.random.default_rng(s)
                if prob.is_binary:
                    init_pop = np.array([prob.random_solution(init_rng) for _ in range(pop_size)], dtype=float)
                else:
                    init_pop = init_rng.uniform(prob.lb, prob.ub, size=(pop_size, prob.dim))

                nash = GANash(prob, pop_size=pop_size, max_gen=max_gen, eta=0.5, eps_min=0.05, beta=beta_val, seed=s)
                res = nash.run(initial_pop=init_pop)
                bests.append(res["best_fitness"])

            bests = np.array(bests)
            sensitivity_records.append({
                "problem": prob_name,
                "param_tested": "beta",
                "param_value": beta_val,
                "fixed_params": "eta=0.5, eps_min=0.05",
                "mean_best": float(np.mean(bests)),
                "std_best": float(np.std(bests)),
                "median_best": float(np.median(bests)),
            })

    # Sauvegarde CSV
    sens_csv_path = os.path.join(output_dir, "sensitivity_results.csv")
    with open(sens_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "problem", "param_tested", "param_value", "fixed_params",
            "mean_best", "std_best", "median_best"
        ])
        writer.writeheader()
        writer.writerows(sensitivity_records)
    print(f"  → CSV Sensibilité sauvegardé : {sens_csv_path}")

    sens_results["records"] = sensitivity_records
    return sens_results


# ===========================================================================
# Point d'entrée principal en ligne de commande
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lancement de la campagne expérimentale.")
    parser.add_argument("--quick", action="store_true", help="Exécution rapide pour tests unitaires et intégration.")
    parser.add_argument("--runs", type=int, default=30, help="Nombre de runs par configuration.")
    parser.add_argument("--no-parallel", action="store_true", help="Désactive le multiprocessing.")
    args = parser.parse_args()

    out_dir = "results"
    campaign_res = run_campaign(
        n_runs=args.runs,
        output_dir=out_dir,
        use_parallel=not args.no_parallel,
        quick=args.quick,
    )
    sens_res = run_sensitivity_analysis(
        n_runs=args.runs if not args.quick else 3,
        output_dir=out_dir,
        quick=args.quick,
    )
    print("\n[OK] Toutes les expériences sont terminées avec succès.")
