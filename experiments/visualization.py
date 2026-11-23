"""
Module : Visualisation des résultats expérimentaux
=================================================
Génère l'ensemble des figures requises pour le rapport (Parties E et F) :
  1. Courbes de convergence moyennes avec bande d'écart-type (4 algorithmes)
  2. Évolution temporelle moyenne des stratégies mixtes (σ₁ et σ₂) sur 30 runs
  3. Évolution du Nash Gap G(σ) au fil des générations
  4. Heatmap de la matrice de gains commune U moyenne issue de la campagne
  5. Courbes d'analyse de sensibilité multi-paramétrique (η, ε_min, β)
"""

import os
import sys
import json
import csv
from typing import Dict, List, Any, Optional

# Assurer l'inclusion de la racine du projet dans sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Mode headless pour exécution en arrière-plan
import matplotlib.pyplot as plt

# Configuration esthétique globale
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 200,
    "lines.linewidth": 1.8,
})

# Palette de couleurs cohérente
ALGO_COLORS = {
    "GA_standard": "#E74C3C",              # Rouge corail
    "GA_uniform": "#7F8C8D",               # Gris neutre
    "GA_probability_matching": "#E67E22",  # Orange ambré
    "GA_nash": "#27AE60",                  # Vert émeraude
}

ALGO_LABELS = {
    "GA_standard": "GA standard (statique)",
    "GA_uniform": "GA uniforme (aléatoire)",
    "GA_probability_matching": "GA adaptatif (AOS)",
    "GA_nash": "GA-Nash (théorie des jeux)",
}


def load_campaign_data(json_path: str = "results/campaign_results.json") -> Dict[str, Any]:
    """Charge les données agrégées de la campagne depuis le fichier JSON."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Fichier de campagne introuvable : {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# 1. Courbes de convergence moyennes avec bandes d'écart-type
# ===========================================================================

def plot_single_convergence(
    problem_name: str,
    agg_problem: Dict[str, Any],
    output_path: str,
) -> str:
    """Trace la convergence moyenne des 4 algorithmes sur un problème donné."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for algo_key, color in ALGO_COLORS.items():
        if algo_key not in agg_problem:
            continue
        data = agg_problem[algo_key]
        mean_conv = np.array(data["mean_convergence"])
        std_conv = np.array(data["std_convergence"])
        gens = np.arange(1, len(mean_conv) + 1)

        label = ALGO_LABELS.get(algo_key, algo_key)
        line_style = "--" if algo_key == "GA_standard" else "-"
        ax.plot(gens, mean_conv, label=label, color=color, linestyle=line_style)
        ax.fill_between(
            gens,
            mean_conv - std_conv,
            mean_conv + std_conv,
            color=color,
            alpha=0.15,
        )

    ax.set_xlabel("Génération")
    ax.set_ylabel("Meilleure fitness moyenne")
    ax.set_title(f"Convergence moyenne (30 runs appariés) — {problem_name}")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", framealpha=0.9)

    # Si problème continu à minimum 0 (sphere, rastrigin, ackley), échelle semi-log
    if "sphere" in problem_name or "ackley" in problem_name or "rastrigin" in problem_name:
        if np.all(np.array(agg_problem["GA_standard"]["mean_convergence"]) > 0):
            ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_all_convergence_grid(
    campaign_data: Dict[str, Any],
    output_dir: str = "results/figures",
) -> str:
    """Génère une grille 2x4 synthétique de convergence pour les 7 instances."""
    problems = list(campaign_data["aggregated"].keys())
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for idx, prob_name in enumerate(problems):
        ax = axes[idx]
        agg_prob = campaign_data["aggregated"][prob_name]

        for algo_key, color in ALGO_COLORS.items():
            if algo_key not in agg_prob:
                continue
            data = agg_prob[algo_key]
            mean_conv = np.array(data["mean_convergence"])
            gens = np.arange(1, len(mean_conv) + 1)
            ax.plot(gens, mean_conv, label=ALGO_LABELS[algo_key], color=color,
                    linestyle="--" if algo_key == "GA_standard" else "-")

        ax.set_title(prob_name, fontweight="bold")
        ax.set_xlabel("Génération")
        ax.set_ylabel("Fitness")
        ax.grid(True, linestyle=":", alpha=0.5)

        if "sphere" in prob_name or "ackley" in prob_name:
            ax.set_yscale("log")

    # Masquer les sous-graphes inutilisés et y placer la légende globale
    for extra_idx in range(len(problems), len(axes)):
        axes[extra_idx].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    legend_idx = len(problems) if len(problems) < len(axes) else -1
    axes[legend_idx].legend(handles, labels, loc="center", fontsize=11, framealpha=0.9)

    n_runs = campaign_data.get("metadata", {}).get("n_runs", 30)
    fig.suptitle(f"Courbes de convergence moyennes ({n_runs} runs indépendants appariés)", fontsize=14, y=0.98)
    fig.tight_layout()
    grid_path = os.path.join(output_dir, "convergence_grid_all_problems.png")
    fig.savefig(grid_path)
    plt.close(fig)
    return grid_path


# ===========================================================================
# 2. Évolution temporelle moyenne des stratégies mixtes (σ₁, σ₂) et Nash Gap
# ===========================================================================

def plot_strategy_dynamics_and_gap(
    campaign_data: Dict[str, Any],
    problem_name: str,
    output_dir: str = "results/figures",
) -> str:
    """
    Trace pour GA-Nash :
      - Sous-figure 1 : Proportions moyennes de croisement (σ₁) sur 30 runs avec ruban std.
      - Sous-figure 2 : Proportions moyennes de mutation (σ₂) sur 30 runs avec ruban std.
      - Sous-figure 3 : Évolution du Nash Gap G(σ) au fil des générations avec ruban std.
    """
    from src.game_theory.nash_game import OperatorGame

    nash_agg = campaign_data["aggregated"][problem_name]["GA_nash"]
    mean_cr = np.array(nash_agg["mean_crossover_probs"])  # (G, 4)
    std_cr = np.array(nash_agg["std_crossover_probs"])
    mean_mr = np.array(nash_agg["mean_mutation_probs"])   # (G, 4)
    std_mr = np.array(nash_agg["std_mutation_probs"])
    mean_gap = np.array(nash_agg["mean_nash_gap"])        # (G,)
    std_gap = np.array(nash_agg["std_nash_gap"])

    cr_vals = OperatorGame.STRATEGIES_CROSSOVER
    mr_vals = OperatorGame.STRATEGIES_MUTATION
    gens = np.arange(1, len(mean_gap) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Palette bleue pour J1 (Croisement)
    cmap_blue = plt.cm.Blues(np.linspace(0.45, 0.95, len(cr_vals)))
    for j, val in enumerate(cr_vals):
        axes[0].plot(gens, mean_cr[:, j], label=f"cr = {val:.2f}", color=cmap_blue[j], linewidth=2)
        axes[0].fill_between(gens, mean_cr[:, j] - std_cr[:, j], mean_cr[:, j] + std_cr[:, j],
                             color=cmap_blue[j], alpha=0.15)
    axes[0].set_title("Joueur 1 : Stratégies de Croisement (σ₁)")
    axes[0].set_xlabel("Génération")
    axes[0].set_ylabel("Probabilité de sélection σ₁")
    axes[0].set_ylim(-0.02, 0.65)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(loc="upper right")

    # Palette orange pour J2 (Mutation)
    cmap_orange = plt.cm.Oranges(np.linspace(0.45, 0.95, len(mr_vals)))
    for j, val in enumerate(mr_vals):
        axes[1].plot(gens, mean_mr[:, j], label=f"mr = {val:.2f}", color=cmap_orange[j], linewidth=2)
        axes[1].fill_between(gens, mean_mr[:, j] - std_mr[:, j], mean_mr[:, j] + std_mr[:, j],
                             color=cmap_orange[j], alpha=0.15)
    axes[1].set_title("Joueur 2 : Stratégies de Mutation (σ₂)")
    axes[1].set_xlabel("Génération")
    axes[1].set_ylabel("Probabilité de sélection σ₂")
    axes[1].set_ylim(-0.02, 0.65)
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(loc="upper right")

    # Sous-figure 3 : Nash Gap G(σ)
    axes[2].plot(gens, mean_gap, label="Nash Gap G(σ)", color="#8E44AD", linewidth=2)
    axes[2].fill_between(gens, np.maximum(0, mean_gap - std_gap), mean_gap + std_gap,
                         color="#8E44AD", alpha=0.18)
    axes[2].set_title("Dynamique de Nash Gap : G(σ) → 0")
    axes[2].set_xlabel("Génération")
    axes[2].set_ylabel("Ecart à l'équilibre G(σ)")
    axes[2].grid(True, linestyle=":", alpha=0.5)
    axes[2].legend(loc="upper right")

    n_runs = campaign_data.get("metadata", {}).get("n_runs", 30)
    fig.suptitle(f"Dynamique évolutive et convergence vers l'équilibre de Nash — {problem_name} ({n_runs} runs)",
                 fontsize=13, y=0.98)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"strategies_and_gap_{problem_name}.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ===========================================================================
# 3. Heatmap de la matrice de gains U moyenne
# ===========================================================================

def plot_mean_payoff_matrix(
    problem_names: List[str] = ["ackley-10", "knapsack-100"],
    output_dir: str = "results/figures",
    n_runs: int = 5,
    seed_base: int = 42,
    max_gen: int = 50,
) -> str:
    """
    Calcule et trace la heatmap de la matrice de gains commune U moyenne
    pour les problèmes représentatifs.
    """
    from src.algorithms.ga_nash import GANash
    from src.problems.benchmark import get_problem
    from src.game_theory.nash_game import OperatorGame

    fig, axes = plt.subplots(1, len(problem_names), figsize=(6.5 * len(problem_names), 5.2))
    if len(problem_names) == 1:
        axes = [axes]

    cr_labels = [f"cr={v:.2f}" for v in OperatorGame.STRATEGIES_CROSSOVER]
    mr_labels = [f"mr={v:.2f}" for v in OperatorGame.STRATEGIES_MUTATION]

    for ax, p_name in zip(axes, problem_names):
        prob = get_problem(p_name)
        U_mats = []
        for r in range(n_runs):
            s = seed_base + r
            init_rng = np.random.default_rng(s)
            if prob.is_binary:
                init_pop = np.array([prob.random_solution(init_rng) for _ in range(50)], dtype=float)
            else:
                init_pop = init_rng.uniform(prob.lb, prob.ub, size=(50, prob.dim))
            model = GANash(prob, pop_size=50, max_gen=max_gen, seed=s)
            res = model.run(initial_pop=init_pop)
            U_mats.append(np.array(res["final_payoff_matrix"]))

        U_mean = np.mean(U_mats, axis=0)

        im = ax.imshow(U_mean, cmap="YlGnBu", aspect="auto", vmin=0.2, vmax=0.7)
        ax.set_xticks(range(len(mr_labels)))
        ax.set_xticklabels(mr_labels)
        ax.set_yticks(range(len(cr_labels)))
        ax.set_yticklabels(cr_labels)
        ax.set_xlabel("Joueur 2 : Taux de Mutation (mr)")
        ax.set_ylabel("Joueur 1 : Taux de Croisement (cr)")
        ax.set_title(f"Matrice de Gains U moyenne — {p_name}")

        # Annotation des valeurs dans les cellules
        for i in range(len(cr_labels)):
            for j in range(len(mr_labels)):
                val = U_mean[i, j]
                text_col = "white" if val > 0.45 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=text_col, fontweight="bold")

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Matrice de Gains Commune U issue de la coordination (Jeu à intérêts communs)",
                 fontsize=13, y=0.98)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "payoff_matrix_heatmaps.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ===========================================================================
# 4. Courbes de sensibilité multi-paramétrique (η, ε_min, β)
# ===========================================================================

def plot_sensitivity_analysis(
    csv_path: str = "results/sensitivity_results.csv",
    output_dir: str = "results/figures",
) -> str:
    """Trace les courbes d'analyse de sensibilité pour les trois hyperparamètres clés."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier de sensibilité introuvable : {csv_path}")

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "problem": row["problem"],
                "param_tested": row["param_tested"],
                "param_value": float(row["param_value"]),
                "fixed_params": row["fixed_params"],
                "mean_best": float(row["mean_best"]),
                "std_best": float(row["std_best"]),
                "median_best": float(row["median_best"]),
            })

    params = ["eta", "eps_min", "beta"]
    param_titles = {
        "eta": "Taux d'apprentissage réplicateur (η)",
        "eps_min": "Plancher d'exploration (ε_min)",
        "beta": "Facteur d'oubli EMA (β)",
    }
    param_xlabels = {
        "eta": "Valeur de η (apprentissage)",
        "eps_min": "Valeur de ε_min (plancher)",
        "beta": "Valeur de β (mémoire EMA)",
    }

    # Problèmes étudiés
    problems = sorted(list(set(r["problem"] for r in records)))
    prob_colors = {
        "rastrigin-10": "#E74C3C",
        "rosenbrock-10": "#3498DB",
        "knapsack-50": "#2ECC71",
        "knapsack-100": "#9B59B6",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, p_param in zip(axes, params):
        for prob in problems:
            prob_recs = [r for r in records if r["problem"] == prob and r["param_tested"] == p_param]
            prob_recs.sort(key=lambda x: x["param_value"])
            if not prob_recs:
                continue

            x_vals = [r["param_value"] for r in prob_recs]
            y_vals = [r["mean_best"] for r in prob_recs]
            y_errs = [r["std_best"] for r in prob_recs]

            # Inverser signe pour sac à dos pour afficher la valeur positive maximisée
            if "knapsack" in prob:
                y_vals = [-v for v in y_vals]

            label = f"{prob} (valeur)" if "knapsack" in prob else f"{prob} (erreur)"
            ax.errorbar(
                x_vals,
                y_vals,
                yerr=y_errs,
                fmt="o-",
                label=label,
                color=prob_colors.get(prob, "#333333"),
                capsize=4,
                linewidth=1.8,
                markersize=6,
            )

        ax.set_title(param_titles[p_param], fontweight="bold")
        ax.set_xlabel(param_xlabels[p_param])
        ax.set_ylabel("Fitness moyenne (30 runs)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Étude de Sensibilité Multi-Paramétrique de GA-Nash (30 runs par point)",
                 fontsize=14, y=0.98)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "sensitivity_analysis.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ===========================================================================
# 5. Point d'entrée principal pour génération complète
# ===========================================================================

def generate_all_visualizations(
    results_dir: str = "results",
    figures_dir: str = "results/figures",
) -> List[str]:
    """Exécute la chaîne complète de génération de graphiques."""
    os.makedirs(figures_dir, exist_ok=True)
    generated_files = []

    print("=" * 65)
    print("  Génération des figures haute résolution pour le rapport...")
    print("=" * 65)

    json_path = os.path.join(results_dir, "campaign_results.json")
    campaign_data = load_campaign_data(json_path)

    # 1. Grille complète de convergence
    grid_path = plot_all_convergence_grid(campaign_data, figures_dir)
    generated_files.append(grid_path)
    print(f"  [OK] Grille de convergence : {grid_path}")

    # 2. Courbes de convergence individuelles
    for prob_name, agg_prob in campaign_data["aggregated"].items():
        out_single = os.path.join(figures_dir, f"convergence_{prob_name}.png")
        plot_single_convergence(prob_name, agg_prob, out_single)
        generated_files.append(out_single)
    print(f"  [OK] {len(campaign_data['aggregated'])} courbes de convergence individuelles générées.")

    # 3. Évolution des stratégies et Nash gap
    target_gap_probs = [p for p in ["ackley-10", "knapsack-100", "rastrigin-10", "knapsack-50"] if p in campaign_data["aggregated"]]
    if not target_gap_probs:
        target_gap_probs = list(campaign_data["aggregated"].keys())
    for p in target_gap_probs:
        gap_path = plot_strategy_dynamics_and_gap(campaign_data, p, figures_dir)
        generated_files.append(gap_path)
        print(f"  [OK] Dynamique de stratégies & Nash gap ({p}) : {gap_path}")

    # 4. Heatmaps de la matrice de gains commune U
    heat_probs = [p for p in ["ackley-10", "knapsack-100"] if p in campaign_data["aggregated"]]
    if not heat_probs:
        heat_probs = list(campaign_data["aggregated"].keys())[:2]
    max_gen_heat = campaign_data.get("metadata", {}).get("max_gen", 50)
    heat_path = plot_mean_payoff_matrix(heat_probs, figures_dir, max_gen=max_gen_heat)
    generated_files.append(heat_path)
    print(f"  [OK] Heatmaps matrice de gains U : {heat_path}")

    # 5. Étude de sensibilité multi-paramétrique
    sens_csv = os.path.join(results_dir, "sensitivity_results.csv")
    if os.path.exists(sens_csv) and os.path.getsize(sens_csv) > 50:
        try:
            sens_path = plot_sensitivity_analysis(sens_csv, figures_dir)
            generated_files.append(sens_path)
            print(f"  [OK] Étude de sensibilité : {sens_path}")
        except Exception as e:
            print(f"  [!] Note: Tracé de sensibilité ignoré ({e})")

    print("=" * 65)
    print(f"  Total : {len(generated_files)} figures générées dans '{figures_dir}'.")
    print("=" * 65)
    return generated_files


if __name__ == "__main__":
    generate_all_visualizations()
