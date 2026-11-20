# Théorie des Jeux dans une Métaheuristique : Hybridation GA-Nash

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()
[![Tests](https://img.shields.io/badge/Tests-40%2F40%20Passed-brightgreen.svg)]()

Ce projet universitaire (USTHB, Master 2 Informatique Visuelle, module *Théorie des Jeux*, sous la direction du **Dr Djamila Dahmani**) implémente, évalue et analyse une métaheuristique génétique hybride guidée par la théorie des jeux évolutionnaire : **GA-Nash**.

---

## 📌 Correspondance avec les Exigences du Sujet de Projet

Conformément à l'énoncé du projet, le travail est structuré selon les parties officielles (l'énoncé ne comportant pas de partie D) :

| Partie de l'Énoncé | Contenu Réalisé | Emplacement Principal |
| :--- | :--- | :--- |
| **Partie A — État de l'art** | Synthèse rigoureuse (2-3 pages) de 3 approches combinant métaheuristiques et théorie des jeux avec références académiques vérifiables (DOI/JSTOR). | [`RAPPORT.md`](./RAPPORT.md#partie-a--état-de-lart--théorie-des-jeux-et-métaheuristiques) |
| **Partie B — Modélisation du jeu** | Formalisation en jeu sous forme normale à intérêts communs ($u_1 = u_2 = U$), attribution du gain au niveau individuel de l'enfant, dynamique de réplication discrète avec plancher $\varepsilon_{\min}$, détection d'équilibres purs et Nash gap. | [`src/game_theory/nash_game.py`](./src/game_theory/nash_game.py), [`RAPPORT.md`](./RAPPORT.md#partie-b--modélisation-mathématique-du-jeu) |
| **Partie C — Implémentation** | Code modulaire, isolation stricte du générateur pseudo-aléatoire (`default_rng`), wrapper d'évaluation garantissant l'égalité absolue du budget (10 050 évaluations). | `src/`, `tests/`, [`RAPPORT.md`](./RAPPORT.md#partie-c--architecture-et-implémentation-logicielle) |
| **Partie E — Expérimentations** | 2 familles de problèmes (continu dimension 10 et sac à dos binaire 50/100/200 objets résolu par programmation dynamique), 30 runs appariés, tests de Wilcoxon bilatéraux avec correction de Holm et $\hat{A}_{12}$, étude de sensibilité multi-paramétrique ($\eta, \varepsilon_{\min}, \beta$). | [`experiments/`](./experiments/), [`results/`](./results/), [`RAPPORT.md`](./RAPPORT.md#partie-e--protocole-expérimental-et-résultats-statistiques) |
| **Partie F — Analyse et Discussion** | 13 figures haute résolution, courbes moyennes avec intervalles d'écart-type, évolution des proportions $\sigma$, Nash gap, heatmap de $U$, discussion honnête des cas favorables, des égalités et du surcoût CPU $\times 2$. | [`results/figures/`](./results/figures/), [`RAPPORT.md`](./RAPPORT.md#partie-f--analyse-détaillée-visualisations-et-discussion-critique) |

---

## 📂 Structure du Répertoire

```
.
├── main.py                        # Script de démonstration rapide et d'inspection
├── README.md                      # Guide d'installation, reproduction et documentation
├── RAPPORT.md                     # Rapport académique complet (Parties A, B, C, E, F)
├── src/
│   ├── algorithms/
│   │   ├── genetic_algorithm.py   # Algorithme génétique standard (opérateurs réels et binaires)
│   │   ├── ga_nash.py             # Algorithme génétique hybride piloté par le jeu de Nash
│   │   └── baselines.py           # GA Uniforme et GA Adaptatif (Probability Matching AOS)
│   ├── game_theory/
│   │   └── nash_game.py           # Modèle du jeu à intérêts communs, dynamique de réplication, Nash gap
│   └── problems/
│       └── benchmark.py           # Problèmes continus (Rastrigin, Rosenbrock, Ackley, Sphere)
│                                  # Problème combinatoire binaire Sac à dos (solveur DP exact + réparation)
├── experiments/
│   ├── run_experiments.py         # Moteur de campagne expérimentale (multiprocessing, CSV/JSON)
│   ├── statistical_tests.py       # Test de Wilcoxon bilatéral, correction de Holm, effet Vargha-Delaney A12
│   └── visualization.py           # Génération des 13 graphiques vectoriels / haute résolution
├── results/
│   ├── summary_metrics.csv        # Métriques agrégées sur 30 runs pour les 4 algorithmes
│   ├── statistical_tests.csv      # P-valeurs de Wilcoxon ajustées Holm et verdicts Win/Tie/Loss
│   ├── sensitivity_results.csv    # Résultats complets de l'étude de sensibilité (1 200 runs)
│   ├── campaign_results.json      # Données brutes et trajectoires de convergence complètes
│   └── figures/                   # 13 graphiques haute résolution générés
└── tests/
    ├── test_phase1.py             # Tests d'hygiène, reproductibilité et égalité de budget
    ├── test_phase2.py             # Tests sac à dos DP exact, opérateur réparateur et parsing
    ├── test_phase3.py             # Tests du composant théorie des jeux (intérêts communs, Nash gap, gains)
    └── test_phase4.py             # Tests des baselines, appariement, Wilcoxon bilatéral, Holm et A12
```

---

## ⚙️ Prérequis et Installation

- **Python** $\ge 3.8$
- Bibliothèques requises :
  - `numpy` : manipulations vectorielles et générateur aléatoire `np.random.default_rng`
  - `scipy` : test non-paramétrique de Wilcoxon (`scipy.stats.wilcoxon`)
  - `matplotlib` : génération des graphiques et visualisations (mode headless `Agg`)

### Justification de l'utilisation de `scipy` et `matplotlib`
Conformément aux règles du projet :
- `scipy` est exclusivement employé dans `experiments/statistical_tests.py` pour le calcul exact et asymptotique du test de Wilcoxon apparié bilatéral, garantissant une rigueur statistique standardisée sans réimplémenter une approximation fragile des rangs signés.
- `matplotlib` est exclusivement employé dans `experiments/visualization.py` pour sauvegarder sur disque les 13 figures scientifiques illustrant la convergence, la dynamique de réplication, la décroissance du Nash gap, les matrices de gains $U$ et les courbes de sensibilité.

### Installation de l'environnement virtuel (optionnel)
```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

pip install numpy scipy matplotlib
```

---

## 🚀 Reproduction des Résultats et Commandes

### 1. Exécution de la Suite Complète de Tests Unitaires (40 tests)
Tous les composants font l'objet d'une validation unitaire stricte :
```bash
python -m unittest discover -s tests -v
```
*Sortie attendue : 40 tests exécutés avec succès (`OK`).*

### 2. Exécution Rapide de Validation du Pipeline (Mode `--quick`)
Pour tester l'intégralité du pipeline d'expérimentation en moins de 5 secondes :
```bash
python experiments/run_experiments.py --quick
```

### 3. Exécution de la Campagne Expérimentale Complète (840 runs + 1 200 runs de sensibilité)
Pour reproduire l'ensemble des données chiffrées de `results/` :
```bash
python experiments/run_experiments.py --runs 30
```
*Cette commande utilise le parallélisme multi-cœur (`ProcessPoolExecutor`). Les fichiers `summary_metrics.csv`, `statistical_tests.csv`, `sensitivity_results.csv` et `campaign_results.json` sont mis à jour.*

### 4. Génération de l'Ensemble des Figures
Pour produire les 13 graphiques haute résolution dans `results/figures/` :
```bash
python experiments/visualization.py
```

### 5. Démonstration Interactive et Inspection Rapide
```bash
python main.py
```

---

## 📊 Résumé des Résultats Scientifiques Réels

Les 4 algorithmes ont été évalués à budget strictement identique (**10 050 évaluations**, population de 50 individus sur 200 générations) sur 30 exécutions appariées par graine.

### Bilan Statistique Face au GA Standard (Wilcoxon Bilatéral avec Correction de Holm)

| Problème | Nature | Moyenne GA Standard | Moyenne GA-Nash | $p_{\text{Holm}}$ | $\hat{A}_{12}$ | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **rastrigin-10** | Continu multimodal | 6.0570 | **5.6404** | 0.5561 | 0.533 | **Tie (Égalité)** |
| **rosenbrock-10** | Continu vallée | 7.4068 | **7.0772** | 0.3819 | 0.600 | **Tie (Égalité)** |
| **ackley-10** | Continu trou d'attraction | 0.4888 | **0.2139** | $1.30 \times 10^{-7}$ | 0.933 | **Win (GA-Nash gagne)** |
| **sphere-10** | Continu convexe | 0.0012 | **0.0006** | $3.81 \times 10^{-3}$ | 0.733 | **Win (GA-Nash gagne)** |
| **knapsack-50** | Combinatoire 0/1 | -1058.33 | **-1059.17** | $4.04 \times 10^{-2}$ | 0.733 | **Win (GA-Nash gagne)** |
| **knapsack-100** | Combinatoire 0/1 | -1897.63 | **-1910.53** | $1.48 \times 10^{-4}$ | 0.817 | **Win (GA-Nash gagne)** |
| **knapsack-200** | Combinatoire 0/1 | -3704.70 | **-3801.50** | $1.15 \times 10^{-5}$ | 0.967 | **Win (GA-Nash gagne)** |

- **Bilan Global face au GA standard** : **5 Victoires, 2 Égalités, 0 Défaite**.
- **Bilan Global face aux deux baselines adaptatives univariées** : **14 Victoires sur 14** ($p_{\text{Holm}} < 0.05, \hat{A}_{12} \ge 0.70$).
- **Surcoût Computationnel** : GA-Nash consomme environ $+100\%$ de temps CPU ($\times 2$, $\approx 2.8\,\text{s}$ vs $1.3\,\text{s}$), justifié dès lors que la fonction objectif est plus coûteuse qu'une simple fonction analytique.

Pour l'analyse scientifique détaillée, la justification théorique formelle et la discussion critique complète, veuillez vous référer à [`RAPPORT.md`](./RAPPORT.md).
