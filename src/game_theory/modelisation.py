"""
Rapport de Modélisation — Partie B du Projet
=============================================
Ce fichier documente formellement la modélisation du jeu théorique.
(Peut être consulté indépendamment du code)

TITRE : Algorithme Génétique hybridé avec Équilibre de Nash (GA-Nash)
AUTEUR : [Votre nom]
DATE   : Novembre 2026
"""

MODELISATION = """
=============================================================================
PARTIE B : MODÉLISATION DU JEU
=============================================================================

─── 1. JOUEURS ──────────────────────────────────────────────────────────────

Γ = {1, 2}

Joueur 1 (Γ₁) — Opérateur d'Exploration :
  Représente le mécanisme de croisement (crossover) du GA.
  Son rôle est de maintenir la diversité de la population et d'explorer
  de nouvelles régions de l'espace de recherche.

Joueur 2 (Γ₂) — Opérateur d'Exploitation :
  Représente le mécanisme de mutation locale.
  Son rôle est de raffiner les solutions existantes et d'exploiter
  les bonnes régions de l'espace.


─── 2. STRATÉGIES ───────────────────────────────────────────────────────────

S₁ = {0.50, 0.65, 0.80, 0.95}   → taux de croisement (crossover rate)
     |S₁| = 4 stratégies pures

S₂ = {0.01, 0.05, 0.10, 0.20}   → taux de mutation locale (mutation rate)
     |S₂| = 4 stratégies pures

Les stratégies mixtes :
  σ₁ ∈ Δ(S₁)  : distribution de probabilités sur S₁
  σ₂ ∈ Δ(S₂)  : distribution de probabilités sur S₂
  σ₁ et σ₂ sont mises à jour chaque génération.


─── 3. GAINS ────────────────────────────────────────────────────────────────

Soient f_before = (f₁,...,fₙ) les fitness de la population avant génération g,
       f_after  = (f₁',...,fₙ') les fitness après génération g.

Gain du Joueur 1 (Exploration) :
  u₁(s₁, s₂) = (best_before - best_after) / (|best_before| + ε)
  où best = min de la population (meilleur individu).

Gain du Joueur 2 (Exploitation) :
  u₂(s₁, s₂) = (mean_before - mean_after) / (|mean_before| + ε)

Normalisation : les gains sont clippés dans [-1, +1].

Mise à jour par moyenne mobile exponentielle (EMA) :
  A[s₁,s₂] ← (1 - α) · A[s₁,s₂] + α · u₁    avec α = learning_rate

Justification : l'EMA gère la non-stationnarité — le paysage de gains
change au cours de l'optimisation.


─── 4. RÈGLE DE DÉCISION ────────────────────────────────────────────────────

Méthode : Fictitious Play (jeu fictif) → approximation de l'Équilibre de Nash

À chaque génération g :
  (1) Calculer : v₁(sᵢ) = Σⱼ A[i,j] · σ₂[j]   pour chaque sᵢ ∈ S₁
                 v₂(sⱼ) = Σᵢ B[i,j] · σ₁[i]   pour chaque sⱼ ∈ S₂
  (2) Mise à jour : σ₁ ← Softmax(v₁)
                    σ₂ ← Softmax(v₂)
  (3) Échantillonnage : (s₁*, s₂*) ~ (σ₁, σ₂)
  (4) Application des opérateurs avec les paramètres (cr=s₁*, mr=s₂*)

Propriété utilisée (Chap. 3, Théorème Nash 1951) :
  Tout jeu fini admet un équilibre en stratégies mixtes.
  → La convergence de Fictitious Play vers l'équilibre de Nash est
    garantie pour les jeux à somme nulle (Robinson 1951).


─── 5. CE QUE LE JEU MODIFIE ────────────────────────────────────────────────

Sans jeu (GA standard) :
  cr et mr sont des constantes fixées avant l'optimisation.
  Problème : un cr élevé aide en début (exploration) mais nuit en fin
             (destruction des bonnes solutions).

Avec jeu (GA-Nash) :
  cr et mr évoluent chaque génération selon l'équilibre de Nash approché.
  - En début d'optimisation : fitness bruyant → gains d'exploration élevés
    → σ₁ favorise les cr élevés, σ₂ favorise les mr élevés (diversification)
  - En fin d'optimisation : gains d'exploitation dominent
    → σ₁ favorise les cr modérés, σ₂ favorise les mr faibles (raffinement)

Cette adaptation dynamique est précisément ce qu'un équilibre de Nash
capture : chaque opérateur maximise son gain compte tenu de ce que
fait l'autre (propriété de meilleure réponse, Définition 3.1, Chap. 3).


─── 6. JUSTIFICATION DU CHOIX DU MODÈLE ────────────────────────────────────

Les jeux en stratégies mixtes (Chap. 3) sont adaptés car :
  a) L'existence de l'équilibre est garantie (Théorème Nash 1951, p.84)
     même quand les intérêts des opérateurs sont partiellement opposés.
  b) La propriété d'indifférence au support (p.75) assure qu'à l'équilibre
     toutes les stratégies actives sont équivalentes → balance optimale.
  c) La formulation est suffisamment générale pour s'appliquer à n'importe
     quelle métaheuristique avec deux types d'opérateurs.
  d) Le Fictitious Play est simple à implémenter et converge en pratique
     même pour des jeux à somme générale.

=============================================================================
"""

if __name__ == "__main__":
    print(MODELISATION)
