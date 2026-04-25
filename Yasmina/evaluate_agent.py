# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 23:03:32 2026

@author: yasmi
"""

"""
Script d'évaluation - Version Corrigée
Corrections :
  1. Mode STOCHASTIQUE par défaut (cohérent avec l'entraînement)
  2. max_steps=200 (cohérent avec train_agent.py)
  3. Charge best_agent.pt (meilleur modèle, pas le dernier)
  4. Diagnostic automatique de l'écart train/eval
  5. Comparaison stochastique vs déterministe
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from environment import DeliveryGame
from policy_network import REINFORCEAgent
import os
import time


class AgentEvaluator:
    """
    Évaluateur corrigé de l'agent entraîné.

    Corrections principales :
    - Utilise le mode STOCHASTIQUE (même qu'à l'entraînement)
    - max_steps identique à l'entraînement (200)
    - Charge le MEILLEUR modèle (best_agent.pt)
    - Diagnostic automatique
    """

    def __init__(self, agent_path: str = None):
        """
        Initialise l'évaluateur.

        ✅ CORRECTION : max_steps=200 (cohérent avec train_agent.py)

        Args:
            agent_path : Chemin vers le modèle entraîné
        """
        # ✅ max_steps=200 identique à l'entraînement
        self.env = DeliveryGame(grid_size=8, max_steps=200)
        self.agent = REINFORCEAgent(
            state_size=7,
            action_size=5
        )
        self.agent_trained = False

        # ── Chargement du modèle ─────────────────────────────────
        if agent_path:
            self._load_best_model(agent_path)

    def _load_best_model(self, agent_path: str):
        """
        Charge le meilleur modèle disponible.

        ✅ CORRECTION : Priorité à best_agent.pt sur trained_agent.pt

        Args:
            agent_path : Chemin demandé par l'utilisateur
        """
        # Priorité : best_agent.pt > agent_path demandé
        candidates = [
            "best_agent.pt",    # ✅ Meilleur modèle (sauvegardé pendant train)
            agent_path,         # Modèle demandé
            "trained_agent.pt"  # Modèle final (fallback)
        ]

        for path in candidates:
            if path and os.path.exists(path):
                try:
                    self.agent.load_model(path)
                    self.agent_trained = True
                    print(f"✅ Modèle chargé : {path}")
                    return
                except Exception as e:
                    print(f"⚠️  Échec chargement {path} : {e}")

        print("❌ Aucun modèle trouvé. Agent non entraîné utilisé.")

    # ────────────────────────────────────────────────────────────
    # SÉLECTION D'ACTION CORRIGÉE
    # ────────────────────────────────────────────────────────────
    def _select_action(self,
                       state: np.ndarray,
                       mode: str = "stochastic") -> int:
        """
        Sélectionne une action selon le mode choisi.

        ✅ CORRECTION PRINCIPALE :
        - "stochastic"    → même comportement qu'à l'entraînement
        - "deterministic" → argmax (peut causer l'écart !)
        - "temperature"   → compromis entre les deux

        Args:
            state : État actuel
            mode  : "stochastic" | "deterministic" | "temperature"
        Returns:
            action : Indice de l'action choisie
        """
        probs = self.agent.policy_network.get_action_probabilities(state)

        if mode == "stochastic":
            # ✅ Échantillonnage = même comportement qu'entraînement
            action = np.random.choice(len(probs), p=probs)

        elif mode == "deterministic":
            # Argmax pur (peut causer l'écart train/eval !)
            action = np.argmax(probs)

        elif mode == "temperature":
            # Softmax avec température T=0.5
            # Plus concentré que stochastique, moins que déterministe
            temperature = 0.5
            log_probs   = np.log(probs + 1e-8) / temperature
            probs_temp  = np.exp(log_probs - np.max(log_probs))
            probs_temp  = probs_temp / probs_temp.sum()
            action      = np.random.choice(len(probs_temp), p=probs_temp)

        else:
            action = np.random.choice(len(probs), p=probs)

        return action

    # ────────────────────────────────────────────────────────────
    # ÉVALUATION PRINCIPALE
    # ────────────────────────────────────────────────────────────
    def evaluate_agent(self,
                       num_episodes: int = 100,
                       mode: str = "stochastic",
                       use_policy: bool = True) -> Tuple[List, List, List]:
        """
        Évalue l'agent sur plusieurs épisodes.

        ✅ CORRECTION : mode="stochastic" par défaut

        Args:
            num_episodes : Nombre d'épisodes d'évaluation
            mode         : "stochastic" | "deterministic" | "temperature"
            use_policy   : True = agent entraîné, False = aléatoire
        Returns:
            (rewards, lengths, successes)
        """
        rewards, lengths, successes = [], [], []

        label = f"Agent {'entraîné' if use_policy else 'aléatoire'} [{mode}]"
        print(f"\n📊 Évaluation : {label} sur {num_episodes} épisodes...")

        for episode in range(num_episodes):
            state        = self.env.reset()
            ep_reward    = 0.0
            ep_length    = 0
            success      = False

            while not self.env.done:

                if use_policy and self.agent_trained:
                    # ✅ Mode configurable (stochastique par défaut)
                    action = self._select_action(state, mode=mode)
                else:
                    # Agent aléatoire (baseline de comparaison)
                    action = np.random.randint(0, self.env.action_space_size)

                state, reward, done, info = self.env.step(action)
                ep_reward += reward
                ep_length += 1

                if info['delivered']:
                    success = True

            rewards.append(ep_reward)
            lengths.append(ep_length)
            successes.append(1 if success else 0)

            if (episode + 1) % 25 == 0:
                recent_success = np.mean(successes[-25:])
                print(f"   Épisode {episode + 1:>3}/{num_episodes} "
                      f"| Succès récents : {recent_success:.1%}")

        return rewards, lengths, successes

    # ────────────────────────────────────────────────────────────
    # DIAGNOSTIC : COMPARE LES MODES
    # ────────────────────────────────────────────────────────────
    def diagnose_gap(self, num_episodes: int = 100):
        """
        Diagnostique l'écart entraînement/évaluation.

        Compare les 3 modes pour identifier la source du problème.

        Args:
            num_episodes : Épisodes par mode
        """
        print("\n" + "=" * 60)
        print("  DIAGNOSTIC : Écart Entraînement / Évaluation")
        print("=" * 60)

        modes = [
            ("stochastic",    "Même qu'entraînement ✅"),
            ("temperature",   "Semi-déterministe    🔶"),
            ("deterministic", "Argmax pur           ⚠️ "),
        ]

        results = {}

        for mode, description in modes:
            r, l, s = self.evaluate_agent(
                num_episodes=num_episodes,
                mode=mode,
                use_policy=True
            )
            results[mode] = {
                "success_rate": np.mean(s),
                "avg_reward"  : np.mean(r),
                "avg_length"  : np.mean(l)
            }

            emoji = ("✅" if np.mean(s) > 0.3
                     else "⚠️ " if np.mean(s) > 0.1
                     else "❌")

            print(f"\n  Mode : {description}")
            print(f"  {emoji} Succès    : {np.mean(s):.1%}")
            print(f"  📊 Récompense : {np.mean(r):+.2f}")
            print(f"  📏 Longueur   : {np.mean(l):.1f} steps")

        # ── Diagnostic automatique ───────────────────────────────
        stoch_rate = results["stochastic"]["success_rate"]
        det_rate   = results["deterministic"]["success_rate"]
        gap        = stoch_rate - det_rate

        print("\n" + "=" * 60)
        print("  CONCLUSION :")
        print("=" * 60)

        if gap > 0.20:
            print(f"  ⚠️  Écart stoch/det = {gap:.1%} → ÉLEVÉ")
            print("  → La politique n'est pas assez concentrée")
            print("  → L'agent dépend trop de la chance")
            print("  → Solutions recommandées :")
            print("     1. Augmenter num_episodes (ex: 5000)")
            print("     2. Ajouter entropy bonus dans train_on_episode")
            print("     3. Utiliser mode stochastique en évaluation")
        elif det_rate < 0.10:
            print(f"  ❌ Déterministe très faible ({det_rate:.1%})")
            print("  → L'agent n'a pas vraiment appris")
            print("  → L'agent n'a pas vraiment appris")
            print("  → Solutions recommandées :")
            print("     1. Vérifier le state (positions relatives ?)")
            print("     2. Vérifier le reward shaping")
            print("     3. Augmenter num_episodes")
        else:
            print(f"  ✅ Performances cohérentes entre les modes")
            print(f"  → Stochastique : {stoch_rate:.1%}")
            print(f"  → Déterministe : {det_rate:.1%}")
            print(f"  → Écart        : {gap:.1%} (acceptable)")

        print("=" * 60)
        return results

    # ────────────────────────────────────────────────────────────
    # COMPARAISON AGENT ENTRAÎNÉ VS ALÉATOIRE
    # ────────────────────────────────────────────────────────────
    def compare_agents(self, num_episodes: int = 100) -> dict:
        """
        Compare l'agent entraîné avec un agent aléatoire.

        ✅ CORRECTION : Utilise mode stochastique pour l'agent entraîné

        Args:
            num_episodes : Épisodes par agent
        Returns:
            Dictionnaire avec les résultats des deux agents
        """
        print("\n" + "=" * 60)
        print("  Comparaison : Agent Entraîné vs Agent Aléatoire")
        print("=" * 60)

        # ── Agent entraîné (mode stochastique) ───────────────────
        trained_r, trained_l, trained_s = self.evaluate_agent(
            num_episodes=num_episodes,
            mode="stochastic",      # ✅ Cohérent avec l'entraînement
            use_policy=True
        )

        # ── Agent aléatoire (baseline) ────────────────────────────
        random_r, random_l, random_s = self.evaluate_agent(
            num_episodes=num_episodes,
            use_policy=False
        )

        # ── Affichage des résultats ───────────────────────────────
        trained_success = np.mean(trained_s) * 100
        random_success  = np.mean(random_s)  * 100

        print(f"\n{'Métrique':<28} {'Entraîné':>12} {'Aléatoire':>12}")
        print("-" * 54)
        print(f"{'Récompense moyenne':<28} "
              f"{np.mean(trained_r):>+11.2f} "
              f"{np.mean(random_r):>+11.2f}")
        print(f"{'Écart-type récompense':<28} "
              f"{np.std(trained_r):>12.2f} "
              f"{np.std(random_r):>12.2f}")
        print(f"{'Taux de succès (%)':<28} "
              f"{trained_success:>11.1f}% "
              f"{random_success:>11.1f}%")
        print(f"{'Longueur moyenne (steps)':<28} "
              f"{np.mean(trained_l):>12.1f} "
              f"{np.mean(random_l):>12.1f}")

        # ── Amélioration relative ─────────────────────────────────
        if random_success > 0:
            improvement = ((trained_success - random_success)
                           / random_success * 100)
            print(f"{'Amélioration relative':<28} "
                  f"{improvement:>+11.1f}%")

        print("=" * 60)

        return {
            'trained': {
                'rewards'  : trained_r,
                'lengths'  : trained_l,
                'successes': trained_s
            },
            'random': {
                'rewards'  : random_r,
                'lengths'  : random_l,
                'successes': random_s
            }
        }

    # ────────────────────────────────────────────────────────────
    # VISUALISATION D'UN ÉPISODE
    # ────────────────────────────────────────────────────────────
    def visualize_episode(self,
                          save_path: str = None,
                          mode: str = "stochastic"):
        """
        Enregistre et visualise la trajectoire d'un épisode.

        Args:
            save_path : Chemin pour sauvegarder l'image
            mode      : Mode de sélection d'action
        """
        state         = self.env.reset()
        positions     = [self.env.robot_pos]
        actions_taken = []
        rewards_steps = []
        has_pkg_steps = []

        print(f"\n🎬 Enregistrement d'un épisode (mode={mode})...")

        while not self.env.done:
            action = self._select_action(state, mode=mode)
            state, reward, done, info = self.env.step(action)

            positions.append(self.env.robot_pos)
            actions_taken.append(action)
            rewards_steps.append(reward)
            has_pkg_steps.append(info['has_package'])

        total_reward = sum(rewards_steps)
        success      = any(info['delivered']
                          for info in [{'delivered': self.env.done
                          and self.env.has_package}])

        print(f"   Récompense totale : {total_reward:.2f}")
        print(f"   Nombre de steps   : {len(rewards_steps)}")

        # ── Création de la figure ─────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f'Visualisation Épisode - Mode {mode} '
            f'(Récompense: {total_reward:.1f})',
            fontsize=14, fontweight='bold'
        )

        # ── Graphique 1 : Trajectoire ─────────────────────────────
        ax = axes[0]
        positions_array = np.array(positions)

        # Colorier la trajectoire selon has_package
        for i in range(len(positions) - 1):
            color = 'orange' if (i < len(has_pkg_steps)
                                 and has_pkg_steps[i]) else 'blue'
            ax.plot(
                [positions_array[i, 0], positions_array[i+1, 0]],
                [positions_array[i, 1], positions_array[i+1, 1]],
                color=color, linewidth=2, alpha=0.7
            )

        # Points de la trajectoire
        ax.scatter(positions_array[:, 0], positions_array[:, 1],
                  c='blue', s=20, alpha=0.5, zorder=3)

        # Départ, colis, objectif
        ax.plot(positions[0][0], positions[0][1],
                'go', markersize=12, label='Départ', zorder=5)
        ax.plot(self.env.package_pos[0], self.env.package_pos[1],
                'y^', markersize=12, label='Colis', zorder=5)
        ax.plot(self.env.goal_pos[0], self.env.goal_pos[1],
                'r*', markersize=15, label='Objectif', zorder=5)

        # Obstacles
        for obs in self.env.obstacles:
            ax.add_patch(plt.Rectangle(
                (obs[0] - 0.5, obs[1] - 0.5), 1, 1,
                color='gray', alpha=0.5
            ))

        # Légende trajectoire
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue',   label='Sans colis'),
            Line2D([0], [0], color='orange', label='Avec colis'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        ax.set_xlim(-0.5, self.env.grid_size - 0.5)
        ax.set_ylim(-0.5, self.env.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Trajectoire du robot')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

        # ── Graphique 2 : Récompenses par step ───────────────────
        ax = axes[1]
        colors_reward = ['green' if r > 0 else 'red'
                        for r in rewards_steps]
        ax.bar(range(len(rewards_steps)), rewards_steps,
               color=colors_reward, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Ligne cumulative
        cumulative = np.cumsum(rewards_steps)
        ax2 = ax.twinx()
        ax2.plot(cumulative, color='purple', linewidth=2,
                label='Cumulatif')
        ax2.set_ylabel('Récompense cumulée', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')

        ax.set_xlabel('Étape')
        ax.set_ylabel('Récompense par étape')
        ax.set_title(f'Récompenses (Total: {total_reward:.1f})')
        ax.grid(True, alpha=0.3)

        # ── Graphique 3 : Distribution des actions ────────────────
        ax = axes[2]
        action_names  = ['Haut', 'Bas', 'Gauche', 'Droite', 'Rester']
        action_counts = [actions_taken.count(i) for i in range(5)]
        colors_action = ['#FF6B6B', '#4ECDC4', '#45B7D1',
                        '#96CEB4', '#FFEAA7']

        bars = ax.bar(action_names, action_counts,
                     color=colors_action, alpha=0.8)

        # Valeurs sur les barres
        for bar, count in zip(bars, action_counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontsize=10
            )

        ax.set_xlabel('Action')
        ax.set_ylabel('Nombre d\'utilisations')
        ax.set_title('Distribution des actions')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Visualisation sauvegardée : {save_path}")

        plt.close()

    # ────────────────────────────────────────────────────────────
    # GRAPHIQUES DE COMPARAISON
    # ────────────────────────────────────────────────────────────
    def plot_comparison(self,
                        comparison_data: dict,
                        save_path: str = None):
        """
        Trace les graphiques de comparaison entraîné vs aléatoire.

        Args:
            comparison_data : Résultats de compare_agents()
            save_path       : Chemin pour sauvegarder l'image
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            'Comparaison : Agent Entraîné vs Agent Aléatoire',
            fontsize=16, fontweight='bold'
        )

        trained = comparison_data['trained']
        random  = comparison_data['random']

        # ── 1. Distribution des récompenses ──────────────────────
        ax = axes[0, 0]
        ax.hist(trained['rewards'], bins=20, alpha=0.7,
                label='Entraîné', color='green')
        ax.hist(random['rewards'],  bins=20, alpha=0.7,
                label='Aléatoire', color='red')
        ax.axvline(np.mean(trained['rewards']), color='darkgreen',
                  linestyle='--', linewidth=2,
                  label=f"Moy. entraîné: {np.mean(trained['rewards']):.1f}")
        ax.axvline(np.mean(random['rewards']), color='darkred',
                  linestyle='--', linewidth=2,
                  label=f"Moy. aléatoire: {np.mean(random['rewards']):.1f}")
        ax.set_xlabel('Récompense totale')
        ax.set_ylabel('Fréquence')
        ax.set_title('Distribution des récompenses')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── 2. Taux de succès (barres) ────────────────────────────
        ax = axes[0, 1]
        success_rates = [
            np.mean(trained['successes']) * 100,
            np.mean(random['successes'])  * 100
        ]
        bars = ax.bar(
            ['Entraîné\n(stochastique)', 'Aléatoire'],
            success_rates,
            color=['green', 'red'],
            alpha=0.7,
            width=0.5
        )
        for bar, rate in zip(bars, success_rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 1,
                f'{rate:.1f}%',
                ha='center', va='bottom',
                fontsize=12, fontweight='bold'
            )
        ax.set_ylabel('Taux de succès (%)')
        ax.set_title('Taux de succès')
        ax.set_ylim([0, 110])
        ax.grid(True, alpha=0.3, axis='y')

        # ── 3. Distribution des longueurs ─────────────────────────
        ax = axes[1, 0]
        ax.hist(trained['lengths'], bins=20, alpha=0.7,
                label='Entraîné', color='green')
        ax.hist(random['lengths'],  bins=20, alpha=0.7,
                label='Aléatoire', color='red')
        ax.axvline(np.mean(trained['lengths']), color='darkgreen',
                  linestyle='--', linewidth=2)
        ax.axvline(np.mean(random['lengths']), color='darkred',
                  linestyle='--', linewidth=2)
        ax.set_xlabel("Nombre d'étapes")
        ax.set_ylabel('Fréquence')
        ax.set_title("Distribution de la longueur des épisodes")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── 4. Boîte à moustaches ─────────────────────────────────
        ax = axes[1, 1]
        bp = ax.boxplot(
            [trained['rewards'], random['rewards']],
            labels=['Entraîné\n(stochastique)', 'Aléatoire'],
            patch_artist=True,
            notch=True
        )

        # Colorer les boîtes
        colors_box = ['green', 'red']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Colorer les médianes
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        ax.set_ylabel('Récompense totale')
        ax.set_title('Boîte à moustaches des récompenses')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Comparaison sauvegardée : {save_path}")

        plt.close()

    # ────────────────────────────────────────────────────────────
    # DÉMONSTRATION PYGAME
    # ────────────────────────────────────────────────────────────
    def play_with_visualization(self,
                                window_width: int = 480,
                                fps: int = 5,
                                mode: str = "stochastic"):
        """
        Affiche le jeu avec pygame pendant que l'agent joue.

        ✅ CORRECTION : mode stochastique par défaut
        ✅ CORRECTION : Affichage enrichi (phase, stats)

        Args:
            window_width : Largeur de la fenêtre pygame
            fps          : Images par seconde
            mode         : Mode de sélection d'action
        """
        pygame.init()

        cell_size     = window_width // self.env.grid_size
        info_height   = 120
        window_height = window_width + info_height

        window = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption(
            f"Agent Entraîné - Mode {mode}"
        )

        clock      = pygame.time.Clock()
        font_large = pygame.font.Font(None, 28)
        font_small = pygame.font.Font(None, 22)

        # ── Réinitialiser l'environnement ─────────────────────────
        state        = self.env.reset()
        total_reward = 0.0
        running      = True
        step_count   = 0

        print(f"\n🎮 Démonstration pygame (mode={mode})...")
        print("   Fermer la fenêtre pour terminer.")

        while running and not self.env.done:

            # ── Gestion des événements ────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

            # ── Sélection et exécution de l'action ───────────────
            action = self._select_action(state, mode=mode)
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            step_count   += 1

            # ── Rendu du jeu ──────────────────────────────────────
            window.fill((240, 240, 240))
            self.env.render(window, cell_size=cell_size)

            # ── Panneau d'informations ────────────────────────────
            info_y = window_width + 5

            # Phase actuelle
            phase = "📦 Phase 1 : Aller chercher le colis"
            if info['has_package']:
                phase = "🎯 Phase 2 : Livrer le colis"

            phase_surf = font_small.render(phase, True, (0, 0, 0))
            window.blit(phase_surf, (10, info_y))

            # Récompense et steps
            reward_text = font_large.render(
                f"Récompense : {total_reward:+.1f}",
                True, (0, 100, 0) if total_reward > 0 else (200, 0, 0)
            )
            window.blit(reward_text, (10, info_y + 25))

            step_text = font_small.render(
                f"Étape : {step_count} / {self.env.max_steps}",
                True, (0, 0, 0)
            )
            window.blit(step_text, (10, info_y + 55))

            # Mode d'évaluation
            mode_text = font_small.render(
                f"Mode : {mode}",
                True, (100, 100, 100)
            )
            window.blit(mode_text, (10, info_y + 78))

            # Statut final
            if done:
                if info['delivered']:
                    status      = "✅ Livraison réussie !"
                    status_color = (0, 180, 0)
                else:
                    status      = "❌ Échec (temps écoulé)"
                    status_color = (200, 0, 0)

                status_surf = font_large.render(status, True, status_color)
                window.blit(status_surf, (10, info_y + 95))

            pygame.display.flip()
            clock.tick(fps)

        # Pause finale pour voir le résultat
        if running:
            time.sleep(1.5)

        pygame.quit()
        print(f"   Jeu terminé | Récompense : {total_reward:+.1f}"
              f" | Steps : {step_count}")


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
def main():
    """
    Fonction principale d'évaluation.

    ✅ CORRECTIONS appliquées :
    1. Charge best_agent.pt en priorité
    2. Utilise mode stochastique
    3. Diagnostic automatique de l'écart
    4. Chemins Windows compatibles
    """

    # ── Dossier de sortie ─────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ── Chargement de l'évaluateur ────────────────────────────────
    print("=" * 60)
    print("  ÉVALUATION DE L'AGENT REINFORCE")
    print("=" * 60)

    evaluator = AgentEvaluator(agent_path="trained_agent.pt")

    # ── Diagnostic de l'écart train/eval ─────────────────────────
    evaluator.diagnose_gap(num_episodes=100)

    # ── Comparaison entraîné vs aléatoire ────────────────────────
    comparison_data = evaluator.compare_agents(num_episodes=100)

    # ── Visualisation d'un épisode ────────────────────────────────
    evaluator.visualize_episode(
        save_path=os.path.join(output_dir, "episode_visualization.png"),
        mode="stochastic"
    )

    # ── Graphiques de comparaison ─────────────────────────────────
    evaluator.plot_comparison(
        comparison_data,
        save_path=os.path.join(output_dir, "agent_comparison.png")
    )

    # ── Démonstration pygame ──────────────────────────────────────
    try:
        print("\n🎮 Lancement de la démonstration pygame...")
        evaluator.play_with_visualization(
            window_width=480,
            fps=5,
            mode="stochastic"   # ✅ Mode stochastique
        )
    except Exception as e:
        print(f"⚠️  Pygame non disponible : {e}")


if __name__ == "__main__":
    main()