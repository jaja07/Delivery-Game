# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 23:14:29 2026

@author: yasmi
"""

"""
Script d'entraînement de l'agent REINFORCE - Version Corrigée
Corrections :
  1. Sauvegarde du MEILLEUR modèle (pas seulement le dernier)
  2. Baseline pour réduire la variance
  3. Cohérence des hyperparamètres avec l'évaluation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from environment import DeliveryGame
from policy_network import REINFORCEAgent
import os


class Trainer:
    """
    Trainer REINFORCE corrigé.
    """

    def __init__(self,
                 num_episodes: int = 3000,
                 grid_size: int = 8,
                 max_steps: int = 200,        # ← Garder cohérent avec evaluate
                 learning_rate: float = 0.0001,
                 discount_factor: float = 0.95):
        """
        Initialiser le trainer.

        Args:
            num_episodes    : Nombre d'épisodes d'entraînement
            grid_size       : Taille de la grille
            max_steps       : Nombre max d'étapes par épisode
            learning_rate   : Taux d'apprentissage
            discount_factor : Facteur d'actualisation gamma
        """
        self.num_episodes    = num_episodes
        self.grid_size       = grid_size
        self.max_steps       = max_steps
        self.discount_factor = discount_factor

        # ── Environnement et Agent ──────────────────────────────
        self.env = DeliveryGame(grid_size=grid_size, max_steps=max_steps)
        self.agent = REINFORCEAgent(
            state_size=7,
            action_size=5,
            learning_rate=learning_rate,
            discount_factor=discount_factor
        )

        # ── Baseline : moyenne mobile exponentielle ─────────────
        # Soustrait au return pour réduire la variance du gradient
        # Sans changer l'espérance (mathématiquement neutre)
        self.baseline       = 0.0
        self.baseline_alpha = 0.95  # Proche de 1 = change lentement

        # ── Suivi du meilleur modèle ────────────────────────────
        self.best_success_rate = 0.0
        self.best_episode      = 0

        # ── Historiques d'entraînement ──────────────────────────
        self.episode_rewards   = []
        self.episode_lengths   = []
        self.episode_losses    = []
        self.episode_successes = []

    # ────────────────────────────────────────────────────────────
    # BASELINE : Réduit la variance des gradients REINFORCE
    # ────────────────────────────────────────────────────────────
    def update_baseline(self, returns: np.ndarray) -> float:
        """
        Met à jour la baseline avec une moyenne mobile exponentielle.

        Formule : baseline = α * baseline + (1-α) * mean(returns)

        Args:
            returns : Returns calculés pour l'épisode courant
        Returns:
            baseline : Valeur de référence mise à jour
        """
        episode_mean  = np.mean(returns)
        self.baseline = (self.baseline_alpha * self.baseline +
                        (1 - self.baseline_alpha) * episode_mean)
        return self.baseline

    # ────────────────────────────────────────────────────────────
    # CALCUL DES RETURNS ACTUALISÉS
    # ────────────────────────────────────────────────────────────
    def compute_returns(self, rewards: List[float]) -> np.ndarray:
        """
        Calcule les returns actualisés G_t pour chaque timestep.

        G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...

        Args:
            rewards : Liste des récompenses de l'épisode
        Returns:
            returns_normalized : Returns normalisés (moyenne=0, std=1)
        """
        # ── Calcul des returns cumulés (de la fin vers le début) ──
        returns = []
        G = 0.0
        for reward in reversed(rewards):
            G = reward + self.discount_factor * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)

        # ── Soustraction de la baseline ──
        baseline  = self.update_baseline(returns)
        returns   = returns - baseline

        # ── Normalisation pour stabiliser les gradients ──
        std = np.std(returns)
        if std > 1e-8:
            returns = (returns - np.mean(returns)) / std

        return returns

    # ────────────────────────────────────────────────────────────
    # EXÉCUTION D'UN ÉPISODE
    # ────────────────────────────────────────────────────────────
    def run_episode(self) -> Tuple[float, int, float, bool]:
        """
        Exécute un épisode complet d'entraînement.

        Returns:
            (total_reward, length, loss, success)
        """
        # Réinitialiser l'environnement
        state        = self.env.reset()
        states       = []
        actions      = []
        rewards      = []
        total_reward = 0.0
        success      = False

        # ── Boucle d'interaction agent ↔ environnement ──────────
        while not self.env.done:
            states.append(state)

            # ✅ Sélection STOCHASTIQUE (échantillonnage depuis π)
            # C'est ce qui permet l'exploration pendant l'entraînement
            action = self.agent.select_action(state)
            actions.append(action)

            # Exécution de l'action dans l'environnement
            state, reward, done, info = self.env.step(action)
            rewards.append(reward)
            total_reward += reward

            if info['delivered']:
                success = True

        # ── Mise à jour de la politique avec les returns ────────
        loss = self.agent.train_on_episode(states, actions, rewards)

        return total_reward, len(rewards), loss, success

    # ────────────────────────────────────────────────────────────
    # BOUCLE D'ENTRAÎNEMENT PRINCIPALE
    # ────────────────────────────────────────────────────────────
    def train(self):
        """
        Entraîne l'agent sur tous les épisodes.
        Sauvegarde automatiquement le MEILLEUR modèle.
        """
        print("=" * 60)
        print("  Démarrage de l'entraînement REINFORCE")
        print("=" * 60)
        print(f"  Épisodes        : {self.num_episodes}")
        print(f"  Grille          : {self.grid_size}x{self.grid_size}")
        print(f"  Max steps       : {self.max_steps}")
        print(f"  Learning rate   : {self.agent.learning_rate}")
        print(f"  Discount factor : {self.discount_factor}")
        print("=" * 60)

        for episode in range(self.num_episodes):

            # ── Exécuter un épisode ──
            reward, length, loss, success = self.run_episode()

            # ── Enregistrer les statistiques ──
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            self.episode_losses.append(loss)
            self.episode_successes.append(1 if success else 0)

            # ── Sauvegarder le MEILLEUR modèle ─────────────────
            # On attend 100 épisodes pour avoir une moyenne fiable
            if (episode + 1) >= 100:
                # Taux de succès sur les 100 derniers épisodes
                recent_success = np.mean(self.episode_successes[-100:])

                if recent_success > self.best_success_rate:
                    self.best_success_rate = recent_success
                    self.best_episode      = episode + 1

                    # ✅ Sauvegarde du MEILLEUR modèle séparément
                    self.agent.save_model("best_agent.pt")

            # ── Affichage de la progression ─────────────────────
            if (episode + 1) % 50 == 0:
                window      = min(50, episode + 1)
                avg_reward  = np.mean(self.episode_rewards[-window:])
                avg_success = np.mean(self.episode_successes[-window:])
                avg_length  = np.mean(self.episode_lengths[-window:])

                # Indicateur visuel de performance
                if avg_success >= 0.4:
                    indicator = "✅ Bon"
                elif avg_success >= 0.2:
                    indicator = "⚠️  Moyen"
                else:
                    indicator = "❌ Faible"

                print(f"\n📊 Épisode {episode + 1}/{self.num_episodes}")
                print(f"   Récompense moyenne : {avg_reward:+.2f}")
                print(f"   Taux de succès     : {avg_success:.1%} {indicator}")
                print(f"   Longueur moyenne   : {avg_length:.1f} steps")
                print(f"   Meilleur succès    : {self.best_success_rate:.1%}"
                      f" (épisode {self.best_episode})")

        print("\n" + "=" * 60)
        print("  ✅ Entraînement terminé !")
        print(f"  Meilleur taux de succès : {self.best_success_rate:.1%}")
        print(f"  Atteint à l'épisode     : {self.best_episode}")
        print("=" * 60)

        # ── Sauvegarder aussi le modèle final ───────────────────
        self.agent.save_model("trained_agent.pt")
        print("  Modèle final sauvegardé  : trained_agent.pt")
        print("  Meilleur modèle sauvegardé : best_agent.pt")

    # ────────────────────────────────────────────────────────────
    # ÉVALUATION INTÉGRÉE (dans le trainer)
    # ────────────────────────────────────────────────────────────
    def evaluate(self, num_episodes: int = 100) -> Tuple[float, float, float]:
        """
        Évalue l'agent avec les DEUX modes pour diagnostiquer l'écart.

        ⚠️ CORRECTION CLEF :
        Compare stochastique vs déterministe pour identifier
        la source de l'écart de performance.

        Args:
            num_episodes : Nombre d'épisodes d'évaluation
        Returns:
            (avg_reward, success_rate, avg_length) mode stochastique
        """
        print("\n" + "=" * 60)
        print("  ÉVALUATION (meilleur modèle)")
        print("=" * 60)

        # ── Charger le meilleur modèle ──────────────────────────
        if os.path.exists("best_agent.pt"):
            self.agent.load_model("best_agent.pt")
            print("  ✅ Meilleur modèle chargé (best_agent.pt)")
        else:
            print("  ⚠️  Modèle final utilisé (best_agent.pt introuvable)")

        results = {}

        # ── Tester les deux modes ────────────────────────────────
        for mode in ["stochastic", "deterministic"]:
            eval_rewards, eval_lengths, eval_successes = [], [], []

            for _ in range(num_episodes):
                state     = self.env.reset()
                ep_reward = 0.0
                success   = False
                length    = 0

                while not self.env.done:
                    probs = self.agent.policy_network.get_action_probabilities(state)

                    if mode == "stochastic":
                        # ✅ Même comportement qu'à l'entraînement
                        action = np.random.choice(len(probs), p=probs)
                    else:
                        # Argmax pur (peut causer l'écart !)
                        action = np.argmax(probs)

                    state, reward, done, info = self.env.step(action)
                    ep_reward += reward
                    length    += 1

                    if info['delivered']:
                        success = True

                eval_rewards.append(ep_reward)
                eval_lengths.append(length)
                eval_successes.append(1 if success else 0)

            results[mode] = {
                "avg_reward"   : np.mean(eval_rewards),
                "success_rate" : np.mean(eval_successes),
                "avg_length"   : np.mean(eval_lengths)
            }

        # ── Affichage comparatif ─────────────────────────────────
        stoch = results["stochastic"]
        det   = results["deterministic"]

        print(f"\n{'Métrique':<25} {'Stochastique':>15} {'Déterministe':>15}")
        print("-" * 55)
        print(f"{'Récompense moyenne':<25} "
              f"{stoch['avg_reward']:>+14.2f} "
              f"{det['avg_reward']:>+14.2f}")
        print(f"{'Taux de succès':<25} "
              f"{stoch['success_rate']:>14.1%} "
              f"{det['success_rate']:>14.1%}")
        print(f"{'Longueur moyenne':<25} "
              f"{stoch['avg_length']:>14.1f} "
              f"{det['avg_length']:>14.1f}")

        # ── Diagnostic automatique (suite) ───────────────────────
        gap = stoch['success_rate'] - det['success_rate']
        print("\n" + "=" * 60)
        print("  DIAGNOSTIC :")

        if gap > 0.15:
            print(f"  ⚠️  Écart stoch/det = {gap:.1%}")
            print("  → Politique pas assez concentrée")
            print("  → L'agent dépend trop de la chance")
            print("  → Solutions :")
            print("     1. Augmenter num_episodes (ex: 5000)")
            print("     2. Réduire entropy_coeff (ex: 0.001)")
            print("     3. Utiliser mode stochastique en évaluation")

        elif det['success_rate'] < 0.10:
            print(f"  ❌ Déterministe très faible ({det['success_rate']:.1%})")
            print("  → L'agent n'a pas vraiment appris")
            print("  → Solutions :")
            print("     1. Vérifier le state (positions relatives ?)")
            print("     2. Vérifier le reward shaping")
            print("     3. Augmenter num_episodes")

        elif stoch['success_rate'] > 0.40:
            print(f"  ✅ Bonnes performances !")
            print(f"  → Stochastique : {stoch['success_rate']:.1%}")
            print(f"  → Déterministe : {det['success_rate']:.1%}")
            print(f"  → Écart        : {gap:.1%} (acceptable)")

        else:
            print(f"  🔶 Performances moyennes")
            print(f"  → Stochastique : {stoch['success_rate']:.1%}")
            print(f"  → Déterministe : {det['success_rate']:.1%}")
            print(f"  → Continuer l'entraînement recommandé")

        print("=" * 60)

        return (
            stoch['avg_reward'],
            stoch['success_rate'],
            stoch['avg_length']
        )

    # ────────────────────────────────────────────────────────────
    # GRAPHIQUES D'ENTRAÎNEMENT
    # ────────────────────────────────────────────────────────────
    def plot_results(self, save_path: str = "training_results.png"):
        """
        Trace les courbes de résultats d'entraînement.

        Génère 4 graphiques :
        1. Récompenses par épisode
        2. Taux de succès (moyenne mobile)
        3. Longueur des épisodes
        4. Perte d'entraînement

        Args:
            save_path : Chemin pour sauvegarder l'image
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            'Résultats Entraînement REINFORCE - Delivery Game',
            fontsize=16, fontweight='bold'
        )

        window = 50  # Fenêtre pour la moyenne mobile

        # ── 1. Récompenses par épisode ────────────────────────────
        ax = axes[0, 0]
        ax.plot(
            self.episode_rewards,
            linewidth=0.8, alpha=0.4,
            color='blue', label='Par épisode'
        )

        # Moyenne mobile
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(
                self.episode_rewards,
                np.ones(window) / window,
                mode='valid'
            )
            ax.plot(
                range(window - 1, len(self.episode_rewards)),
                moving_avg,
                linewidth=2, color='red',
                label=f'Moyenne mobile ({window} ep.)'
            )

        # Ligne zéro
        ax.axhline(y=0, color='black', linestyle='--',
                  alpha=0.3, linewidth=1)

        ax.set_xlabel('Épisode')
        ax.set_ylabel('Récompense totale')
        ax.set_title('Récompense par épisode')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── 2. Taux de succès ─────────────────────────────────────
        ax = axes[0, 1]

        if len(self.episode_successes) >= window:
            success_avg = np.convolve(
                self.episode_successes,
                np.ones(window) / window,
                mode='valid'
            ) * 100

            ax.plot(
                range(window - 1, len(self.episode_successes)),
                success_avg,
                linewidth=2, color='green',
                label=f'Moyenne mobile ({window} ep.)'
            )

            # Zone colorée sous la courbe
            ax.fill_between(
                range(window - 1, len(self.episode_successes)),
                success_avg,
                alpha=0.2, color='green'
            )

        # Ligne objectif 50%
        ax.axhline(y=50, color='orange', linestyle='--',
                  alpha=0.7, linewidth=1.5, label='Objectif 50%')

        # Meilleur taux
        if self.best_success_rate > 0:
            ax.axhline(
                y=self.best_success_rate * 100,
                color='red', linestyle=':',
                alpha=0.7, linewidth=1.5,
                label=f'Meilleur: {self.best_success_rate:.1%}'
            )

        ax.set_xlabel('Épisode')
        ax.set_ylabel('Taux de succès (%)')
        ax.set_title('Taux de succès (moyenne mobile)')
        ax.set_ylim([0, 105])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── 3. Longueur des épisodes ──────────────────────────────
        ax = axes[1, 0]
        ax.plot(
            self.episode_lengths,
            linewidth=0.8, alpha=0.4,
            color='orange', label='Par épisode'
        )

        if len(self.episode_lengths) >= window:
            length_avg = np.convolve(
                self.episode_lengths,
                np.ones(window) / window,
                mode='valid'
            )
            ax.plot(
                range(window - 1, len(self.episode_lengths)),
                length_avg,
                linewidth=2, color='darkorange',
                label=f'Moyenne mobile ({window} ep.)'
            )

        # Ligne max_steps
        ax.axhline(
            y=self.max_steps,
            color='red', linestyle='--',
            alpha=0.5, linewidth=1,
            label=f'Max steps ({self.max_steps})'
        )

        ax.set_xlabel('Épisode')
        ax.set_ylabel("Nombre d'étapes")
        ax.set_title('Longueur des épisodes')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── 4. Perte d'entraînement ───────────────────────────────
        ax = axes[1, 1]
        ax.plot(
            self.episode_losses,
            linewidth=0.8, alpha=0.4,
            color='purple', label='Par épisode'
        )

        if len(self.episode_losses) >= window:
            loss_avg = np.convolve(
                self.episode_losses,
                np.ones(window) / window,
                mode='valid'
            )
            ax.plot(
                range(window - 1, len(self.episode_losses)),
                loss_avg,
                linewidth=2, color='darkviolet',
                label=f'Moyenne mobile ({window} ep.)'
            )

        ax.set_xlabel('Épisode')
        ax.set_ylabel('Perte (Loss)')
        ax.set_title("Perte d'entraînement")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # ── Sauvegarde ────────────────────────────────────────────
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Graphiques sauvegardés : {save_path}")
        plt.close()

    # ────────────────────────────────────────────────────────────
    # SAUVEGARDE / CHARGEMENT
    # ────────────────────────────────────────────────────────────
    def save_agent(self, filepath: str = "trained_agent.pt"):
        """
        Sauvegarde l'agent entraîné.

        Args:
            filepath : Chemin de sauvegarde
        """
        self.agent.save_model(filepath)

    def load_agent(self, filepath: str = "trained_agent.pt"):
        """
        Charge un agent sauvegardé.

        Args:
            filepath : Chemin du modèle à charger
        """
        self.agent.load_model(filepath)


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
def main():
    """
    Fonction principale d'entraînement.

    ✅ Chemins Windows compatibles (pas de /mnt/...)
    ✅ Dossier outputs créé automatiquement
    """

    # ── Dossier de sortie ─────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ── Création du Trainer ───────────────────────────────────────
    trainer = Trainer(
        num_episodes    = 3000,
        grid_size       = 8,
        max_steps       = 200,
        learning_rate   = 0.0001,
        discount_factor = 0.95
    )

    # ── Entraînement ──────────────────────────────────────────────
    trainer.train()

    # ── Évaluation avec diagnostic ────────────────────────────────
    trainer.evaluate(num_episodes=100)

    # ── Graphiques ────────────────────────────────────────────────
    plot_path = os.path.join(output_dir, "training_results.png")
    trainer.plot_results(save_path=plot_path)

    # ── Sauvegarde finale ─────────────────────────────────────────
    final_path = os.path.join(output_dir, "trained_agent.pt")
    trainer.save_agent(filepath=final_path)

    print("\n" + "=" * 60)
    print("  ✅ Tout terminé !")
    print(f"  📁 Fichiers dans : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()