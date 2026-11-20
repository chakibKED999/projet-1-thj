"""
Point d'entrée principal — Démonstration et exécution du projet
=============================================================
Projet IVM2 (USTHB) : Théorie des Jeux dans une Métaheuristique
Hybridation GA-Nash — Responsable : Dr Djamila Dahmani
"""

import sys
import os
import argparse

# Inclusion explicite du répertoire racine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.run_experiments import run_campaign, run_sensitivity_analysis
from experiments.visualization import generate_all_visualizations


def main():
    parser = argparse.ArgumentParser(
        description="Métaheuristique génétique guidée par la théorie des jeux (GA-Nash)."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Lance la campagne expérimentale complète (30 runs, 7 instances, 4 algorithmes) et l'étude de sensibilité.",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Génère uniquement les figures à partir des résultats existants dans 'results/'.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Nombre de runs en mode démonstration (défaut: 5 runs par instance).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  USTHB — M2 IV — Théorie des Jeux dans une Métaheuristique (GA-Nash)")
    print("=" * 70)

    if args.visualize_only:
        print("\n[*] Mode Visualisation : Génération des 13 figures scientifiques...")
        generate_all_visualizations(results_dir="results", figures_dir="results/figures")
        print("\n[OK] Visualisations terminées. Consulter 'results/figures/'.")
        return

    if args.full:
        print("\n[*] Mode Campagne Complète (30 runs par configuration)...")
        run_campaign(n_runs=30, output_dir="results", use_parallel=True)
        run_sensitivity_analysis(n_runs=30, output_dir="results")
        generate_all_visualizations(results_dir="results", figures_dir="results/figures")
        print("\n[OK] Campagne complète et visualisations terminées avec succès.")
    else:
        print(f"\n[*] Mode Démonstration ({args.runs} runs par instance sur rastrigin-10 et knapsack-50)...")
        demo_problems = ["rastrigin-10", "knapsack-50"]
        run_campaign(
            problems=demo_problems,
            n_runs=args.runs,
            pop_size=50,
            max_gen=50,
            output_dir="results",
            use_parallel=True,
        )
        print("\n[*] Génération des figures pour la démonstration...")
        generate_all_visualizations(results_dir="results", figures_dir="results/figures")
        print("\n[OK] Démonstration terminée avec succès. Consultez 'results/' et 'results/figures/'.")


if __name__ == "__main__":
    main()
