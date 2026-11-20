"""Script de test rapide."""
import sys, os
sys.path.insert(0, '.')

from src.problems.benchmark import get_problem, KnapsackProblem
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.ga_nash import GANash
from experiments.run_experiments import run_experiment
import numpy as np

print('=== Test rapide GA-Nash ===')
problem = get_problem('rastrigin', dim=10)

# Test GA standard
ga = GeneticAlgorithm(problem=problem, pop_size=30, max_gen=50, seed=0)
r_ga = ga.run()
print(f'GA standard  : best={r_ga["best_fitness"]:.4f}')

# Test GA-Nash
ga_nash = GANash(problem=problem, pop_size=30, max_gen=50, seed=0)
r_nash = ga_nash.run()
print(f'GA-Nash      : best={r_nash["best_fitness"]:.4f}')
print(f'Strat sigma1 : {[f"{p:.3f}" for p in r_nash["final_strategies"]["exploration_probs"]]}')
print(f'Strat sigma2 : {[f"{p:.3f}" for p in r_nash["final_strategies"]["exploitation_probs"]]}')

# Test run_experiment
exp = run_experiment(GANash, problem, n_runs=3, pop_size=30, max_gen=50)
print(f'Exp moy={exp["mean"]:.4f} std={exp["std"]:.4f}')

# Test knapsack
kp = KnapsackProblem(n_items=10, seed=0)
sol = kp.random_solution(np.random.default_rng(0))
print(f'Knapsack eval: {kp.evaluate(sol):.4f}')

print('=== Tous les tests OK ===')
