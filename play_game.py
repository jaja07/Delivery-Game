# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 23:29:45 2026

@author: yasmi
"""

"""
Script de jeu manuel - Delivery Game
Permet à un humain de jouer au jeu avec les touches du clavier.
Ou de regarder l'agent IA jouer en démonstration.

Contrôles :
    Flèches ↑↓←→ : Déplacer le robot
    Espace        : Rester sur place
    R             : Nouveau jeu
    A             : Mode IA (regarder l'agent jouer)
    Q             : Quitter
"""

import pygame
import numpy as np
import os
import sys
from environment import DeliveryGame
from policy_network import REINFORCEAgent


# ────────────────────────────────────────────────────────────────
# CONSTANTES
# ────────────────────────────────────────────────────────────────
WINDOW_WIDTH  = 480
WINDOW_HEIGHT = 600
GRID_SIZE     = 8
CELL_SIZE     = WINDOW_WIDTH // GRID_SIZE
INFO_HEIGHT   = WINDOW_HEIGHT - WINDOW_WIDTH

FPS_HUMAN = 60   # FPS pour le mode humain
FPS_AI    = 5    # FPS pour le mode IA (plus lent pour voir)

# ── Couleurs ──────────────────────────────────────────────────────
WHITE      = (255, 255, 255)
BLACK      = (0,   0,   0)
GRAY       = (180, 180, 180)
DARK_GRAY  = (80,  80,  80)
BLUE       = (0,   100, 255)
LIGHT_BLUE = (173, 216, 230)
GREEN      = (0,   200, 0)
DARK_GREEN = (0,   140, 0)
YELLOW     = (255, 220, 0)
ORANGE     = (255, 140, 0)
RED        = (220, 0,   0)
PURPLE     = (150, 0,   200)
BG_COLOR   = (245, 245, 245)


class DeliveryGamePlayer:
    """
    Interface pygame pour jouer au Delivery Game.

    Modes disponibles :
    - Mode Humain  : contrôle au clavier
    - Mode IA      : l'agent entraîné joue automatiquement
    """

    def __init__(self,
                 agent_path: str = None,
                 grid_size: int = GRID_SIZE,
                 max_steps: int = 200):
        """
        Initialise le jeu pygame.

        Args:
            agent_path : Chemin vers le modèle entraîné (None = pas d'IA)
            grid_size  : Taille de la grille
            max_steps  : Nombre max d'étapes par épisode
        """
        # ── Environnement ─────────────────────────────────────────
        self.env       = DeliveryGame(
            grid_size=grid_size,
            max_steps=max_steps
        )
        self.grid_size = grid_size
        self.max_steps = max_steps

        # ── Agent IA ──────────────────────────────────────────────
        self.agent         = None
        self.agent_loaded  = False
        self._load_agent(agent_path)

        # ── Pygame ────────────────────────────────────────────────
        pygame.init()
        pygame.display.set_caption("Delivery Game")

        self.window = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )
        self.clock  = pygame.time.Clock()

        # ── Polices ───────────────────────────────────────────────
        self.font_title  = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 26)
        self.font_small  = pygame.font.Font(None, 22)

        # ── État du jeu ───────────────────────────────────────────
        self.mode          = "human"   # "human" ou "ai"
        self.total_reward  = 0.0
        self.step_count    = 0
        self.game_over     = False
        self.success       = False
        self.message       = ""
        self.message_timer = 0

        # ── Statistiques de session ───────────────────────────────
        self.session_games    = 0
        self.session_wins     = 0
        self.session_rewards  = []

    # ────────────────────────────────────────────────────────────
    # CHARGEMENT DE L'AGENT
    # ────────────────────────────────────────────────────────────
    def _load_agent(self, agent_path: str):
        """
        Charge l'agent IA si disponible.

        Priorité : best_agent.pt > agent_path > trained_agent.pt

        Args:
            agent_path : Chemin demandé par l'utilisateur
        """
        self.agent = REINFORCEAgent(
            state_size=7,
            action_size=5
        )

        # Candidats par ordre de priorité
        candidates = [
            "best_agent.pt",
            agent_path,
            "trained_agent.pt",
            os.path.join("outputs", "best_agent.pt"),
            os.path.join("outputs", "trained_agent.pt"),
        ]

        for path in candidates:
            if path and os.path.exists(path):
                try:
                    self.agent.load_model(path)
                    self.agent_loaded = True
                    print(f"✅ Agent IA chargé : {path}")
                    return
                except Exception as e:
                    print(f"⚠️  Échec chargement {path} : {e}")

        print("⚠️  Aucun agent IA trouvé. Mode humain uniquement.")

    # ────────────────────────────────────────────────────────────
    # RESET DU JEU
    # ────────────────────────────────────────────────────────────
    def _reset_game(self):
        """Réinitialise le jeu pour une nouvelle partie."""
        self.state        = self.env.reset()
        self.total_reward = 0.0
        self.step_count   = 0
        self.game_over    = False
        self.success      = False
        self.session_games += 1

        self._show_message("Nouvelle partie !")

    # ────────────────────────────────────────────────────────────
    # MESSAGE TEMPORAIRE
    # ────────────────────────────────────────────────────────────
    def _show_message(self, msg: str, duration: int = 90):
        """
        Affiche un message temporaire à l'écran.

        Args:
            msg      : Message à afficher
            duration : Durée en frames
        """
        self.message       = msg
        self.message_timer = duration

    # ────────────────────────────────────────────────────────────
    # ACTION IA
    # ────────────────────────────────────────────────────────────
    def _get_ai_action(self) -> int:
        """
        Obtient l'action de l'agent IA (mode stochastique).

        Returns:
            action : Indice de l'action (0-4)
        """
        if not self.agent_loaded:
            return np.random.randint(0, 5)

        probs  = self.agent.policy_network.get_action_probabilities(
            self.state
        )
        # ✅ Mode stochastique (cohérent avec l'entraînement)
        action = np.random.choice(len(probs), p=probs)
        return action

    # ────────────────────────────────────────────────────────────
    # GESTION DES ÉVÉNEMENTS CLAVIER
    # ────────────────────────────────────────────────────────────
    def _handle_events(self) -> tuple:
        """
        Gère les événements pygame (clavier, fermeture).

        Returns:
            (running, action)
            running : False si on quitte
            action  : Action choisie (-1 si aucune)
        """
        action  = -1   # -1 = pas d'action
        running = True

        for event in pygame.event.get():

            # ── Fermeture de la fenêtre ───────────────────────────
            if event.type == pygame.QUIT:
                running = False

            # ── Touches du clavier ────────────────────────────────
            elif event.type == pygame.KEYDOWN:

                # Quitter
                if event.key == pygame.K_q:
                    running = False

                # Nouveau jeu
                elif event.key == pygame.K_r:
                    self._reset_game()

                # Basculer mode IA / Humain
                elif event.key == pygame.K_a:
                    if self.agent_loaded:
                        self.mode = "ai" if self.mode == "human" else "human"
                        mode_name = "IA 🤖" if self.mode == "ai" else "Humain 👤"
                        self._show_message(f"Mode : {mode_name}")
                    else:
                        self._show_message("⚠️ Aucun agent IA chargé !")

                # ── Contrôles de déplacement (mode humain) ────────
                elif self.mode == "human" and not self.game_over:

                    if event.key == pygame.K_UP:
                        action = 0   # Haut
                    elif event.key == pygame.K_DOWN:
                        action = 1   # Bas
                    elif event.key == pygame.K_LEFT:
                        action = 2   # Gauche
                    elif event.key == pygame.K_RIGHT:
                        action = 3   # Droite
                    elif event.key == pygame.K_SPACE:
                        action = 4   # Rester

        return running, action

    # ────────────────────────────────────────────────────────────
    # MISE À JOUR DU JEU
    # ────────────────────────────────────────────────────────────
    def _update(self, action: int):
        """
        Met à jour l'état du jeu avec l'action choisie.

        Args:
            action : Indice de l'action (0-4), -1 = aucune
        """
        if self.game_over or action == -1:
            return

        # ── Exécuter l'action ─────────────────────────────────────
        self.state, reward, done, info = self.env.step(action)
        self.total_reward += reward
        self.step_count   += 1

        # ── Vérifier les événements ───────────────────────────────
        if info['picked_package']:
            self._show_message("📦 Colis récupéré ! Livrez-le !")

        if info['delivered']:
            self.success   = True
            self.game_over = True
            self.session_wins += 1
            self.session_rewards.append(self.total_reward)
            self._show_message(
                f"🎉 Livraison réussie ! +{self.total_reward:.0f} pts"
            )

        if done and not self.success:
            self.game_over = True
            self.session_rewards.append(self.total_reward)
            self._show_message("❌ Temps écoulé ! Appuyez sur R")

    # ────────────────────────────────────────────────────────────
    # RENDU DE LA GRILLE
    # ────────────────────────────────────────────────────────────
    def _draw_grid(self):
        """Dessine la grille de jeu."""

        # ── Fond de la grille ─────────────────────────────────────
        grid_rect = pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_WIDTH)
        pygame.draw.rect(self.window, WHITE, grid_rect)

        # ── Obstacles ─────────────────────────────────────────────
        for obs_x, obs_y in self.env.obstacles:
            rect = pygame.Rect(
                obs_x * CELL_SIZE,
                obs_y * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )
            pygame.draw.rect(self.window, DARK_GRAY, rect)

            # Texture hachurée pour les murs
            for i in range(0, CELL_SIZE, 8):
                pygame.draw.line(
                    self.window, GRAY,
                    (obs_x * CELL_SIZE, obs_y * CELL_SIZE + i),
                    (obs_x * CELL_SIZE + i, obs_y * CELL_SIZE),
                    1
                )

        # ── Zone objectif (vert) ──────────────────────────────────
        gx, gy = self.env.goal_pos
        goal_rect = pygame.Rect(
            gx * CELL_SIZE, gy * CELL_SIZE,
            CELL_SIZE, CELL_SIZE
        )
        pygame.draw.rect(self.window, GREEN, goal_rect)
        pygame.draw.rect(self.window, DARK_GREEN, goal_rect, 3)

        # Lettre G
        g_surf = self.font_medium.render("G", True, WHITE)
        self.window.blit(g_surf, (
            gx * CELL_SIZE + CELL_SIZE // 3,
            gy * CELL_SIZE + CELL_SIZE // 4
        ))

        # ── Colis (jaune) si pas encore ramassé ───────────────────
        if not self.env.has_package:
            px, py = self.env.package_pos
            pkg_rect = pygame.Rect(
                px * CELL_SIZE, py * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )
            pygame.draw.rect(self.window, YELLOW, pkg_rect)
            pygame.draw.rect(self.window, ORANGE, pkg_rect, 3)

            # Icône colis
            p_surf = self.font_medium.render("📦", True, BLACK)
            self.window.blit(p_surf, (
                px * CELL_SIZE + 4,
                py * CELL_SIZE + 4
            ))

        # ── Robot ─────────────────────────────────────────────────
        rx, ry      = self.env.robot_pos
        robot_color = ORANGE if self.env.has_package else BLUE
        robot_rect  = pygame.Rect(
            rx * CELL_SIZE, ry * CELL_SIZE,
            CELL_SIZE, CELL_SIZE
        )
        pygame.draw.rect(self.window, robot_color, robot_rect)
        pygame.draw.rect(self.window, BLACK, robot_rect, 2)

        # Icône robot
        r_surf = self.font_medium.render("🤖", True, WHITE)
        self.window.blit(r_surf, (
            rx * CELL_SIZE + 2,
            ry * CELL_SIZE + 4
        ))

        # ── Lignes de la grille ───────────────────────────────────
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                self.window, GRAY,
                (x * CELL_SIZE, 0),
                (x * CELL_SIZE, WINDOW_WIDTH), 1
            )
        for y in range(self.grid_size + 1):
            pygame.draw.line(
                self.window, GRAY,
                (0, y * CELL_SIZE),
                (WINDOW_WIDTH, y * CELL_SIZE), 1
            )

    # ────────────────────────────────────────────────────────────
    # RENDU DU PANNEAU D'INFORMATIONS
    # ────────────────────────────────────────────────────────────
    def _draw_info_panel(self):
        """
        Dessine le panneau d'informations sous la grille.
        Affiche : mode, score, steps, phase, stats session.
        """
        # ── Fond du panneau ───────────────────────────────────────
        panel_rect = pygame.Rect(
            0, WINDOW_WIDTH,
            WINDOW_WIDTH, INFO_HEIGHT
        )
        pygame.draw.rect(self.window, BG_COLOR, panel_rect)
        pygame.draw.line(
            self.window, DARK_GRAY,
            (0, WINDOW_WIDTH),
            (WINDOW_WIDTH, WINDOW_WIDTH), 2
        )

        y_offset = WINDOW_WIDTH + 8

        # ── Ligne 1 : Mode + Score ────────────────────────────────
        mode_label = "🤖 IA" if self.mode == "ai" else "👤 Humain"
        mode_color = PURPLE if self.mode == "ai" else BLUE

        mode_surf = self.font_medium.render(
            f"Mode : {mode_label}", True, mode_color
        )
        self.window.blit(mode_surf, (10, y_offset))

        score_color = (0, 150, 0) if self.total_reward >= 0 else RED
        score_surf  = self.font_medium.render(
            f"Score : {self.total_reward:+.1f}",
            True, score_color
        )
        self.window.blit(score_surf, (WINDOW_WIDTH - 160, y_offset))

        y_offset += 26

        # ── Ligne 2 : Steps + Phase ───────────────────────────────
        steps_surf = self.font_small.render(
            f"Étapes : {self.step_count} / {self.max_steps}",
            True, DARK_GRAY
        )
        self.window.blit(steps_surf, (10, y_offset))

        # Barre de progression des steps
        bar_x     = 160
        bar_w     = 150
        bar_h     = 12
        bar_rect  = pygame.Rect(bar_x, y_offset + 2, bar_w, bar_h)
        fill_w    = int(bar_w * self.step_count / self.max_steps)
        fill_color = GREEN if fill_w < bar_w * 0.7 else ORANGE \
                     if fill_w < bar_w * 0.9 else RED

        pygame.draw.rect(self.window, LIGHT_BLUE, bar_rect)
        pygame.draw.rect(
            self.window, fill_color,
            pygame.Rect(bar_x, y_offset + 2, fill_w, bar_h)
        )
        pygame.draw.rect(self.window, DARK_GRAY, bar_rect, 1)

        y_offset += 24

        # ── Ligne 3 : Phase actuelle ──────────────────────────────
        if not self.game_over:
            if not self.env.has_package:
                phase      = "📦 Phase 1 : Récupérer le colis"
                phase_color = ORANGE
            else:
                phase      = "🎯 Phase 2 : Livrer le colis"
                phase_color = DARK_GREEN
        else:
            if self.success:
                phase       = "🎉 Livraison réussie !"
                phase_color = DARK_GREEN
            else:
                phase       = "❌ Temps écoulé ! [R] Rejouer"
                phase_color = RED

        phase_surf = self.font_small.render(phase, True, phase_color)
        self.window.blit(phase_surf, (10, y_offset))

        y_offset += 24

        # ── Ligne 4 : Stats de session ────────────────────────────
        win_rate = (self.session_wins / max(self.session_games, 1)) * 100
        avg_reward = (np.mean(self.session_rewards)
                      if self.session_rewards else 0.0)

        stats_surf = self.font_small.render(
            f"Session : {self.session_wins}/{self.session_games} victoires"
            f" ({win_rate:.0f}%)  |  Moy: {avg_reward:+.1f}",
            True, DARK_GRAY
        )
        self.window.blit(stats_surf, (10, y_offset))

        y_offset += 22

        # ── Ligne 5 : Contrôles ───────────────────────────────────
        controls = "Flèches : Déplacer  |  Espace: Rester  |  R: Rejouer  |  A: IA/Humain  |  Q: Quitter"
        ctrl_surf = self.font_small.render(controls, True, GRAY)
        self.window.blit(ctrl_surf, (10, y_offset))

        # ── Message temporaire ────────────────────────────────────
        if self.message_timer > 0:
            alpha = min(255, self.message_timer * 4)

            # Fond semi-transparent
            msg_surf = self.font_title.render(
                self.message, True, WHITE
            )
            msg_w    = msg_surf.get_width()
            msg_x    = (WINDOW_WIDTH - msg_w) // 2
            msg_y    = WINDOW_WIDTH // 2 - 20

            # Rectangle de fond
            bg_surf = pygame.Surface(
                (msg_w + 20, 40), pygame.SRCALPHA
            )
            bg_surf.fill((0, 0, 0, 160))
            self.window.blit(bg_surf, (msg_x - 10, msg_y - 5))
            self.window.blit(msg_surf, (msg_x, msg_y))

            self.message_timer -= 1

    # ────────────────────────────────────────────────────────────
    # ÉCRAN DE DÉMARRAGE
    # ────────────────────────────────────────────────────────────
    def _draw_start_screen(self):
        """Affiche l'écran de démarrage."""
        self.window.fill(BG_COLOR)

        # Titre
        title_surf = self.font_title.render(
            "DELIVERY GAME", True, BLUE
        )
        self.window.blit(title_surf, (
            (WINDOW_WIDTH - title_surf.get_width()) // 2, 80
        ))

        # Sous-titre
        sub_surf = self.font_medium.render(
            "Livrez le colis à la zone objectif !",
            True, DARK_GRAY
        )
        self.window.blit(sub_surf, (
            (WINDOW_WIDTH - sub_surf.get_width()) // 2, 130
        ))

        # Instructions
        instructions = [
            ("Flèches",  "Déplacer le robot"),
            ("Espace",    "Rester sur place"),
            ("R",         "Nouveau jeu"),
            ("A",         "Basculer mode IA / Humain"),
            ("Q",         "Quitter"),
        ]

        y = 200
        for key, desc in instructions:
            key_surf = self.font_medium.render(
                f"{key:<10}", True, BLUE
            )
            desc_surf = self.font_medium.render(
                f": {desc}", True, DARK_GRAY
            )
            self.window.blit(key_surf, (100, y))
            self.window.blit(desc_surf, (210, y))
            y += 32

        # Légende couleurs
        legend = [
            (BLUE,   "Robot"),
            (ORANGE, "Robot + Colis"),
            (YELLOW, "Colis"),
            (GREEN,  "Zone objectif"),
            (DARK_GRAY, "Obstacle"),
        ]

        y += 20
        legend_title = self.font_medium.render(
            "Légende :", True, DARK_GRAY
        )
        self.window.blit(legend_title, (100, y))
        y += 30

        for color, label in legend:
            pygame.draw.rect(
                self.window, color,
                pygame.Rect(100, y, 20, 20)
            )
            pygame.draw.rect(
                self.window, BLACK,
                pygame.Rect(100, y, 20, 20), 1
            )
            label_surf = self.font_small.render(
                label, True, DARK_GRAY
            )
            self.window.blit(label_surf, (130, y + 2))
            y += 28

        # Agent IA
        ai_status = (
            "Agent IA disponible (A pour activer)"
            if self.agent_loaded
            else "Aucun agent IA (entraînez d'abord)"
        )
        ai_color = DARK_GREEN if self.agent_loaded else RED
        ai_surf  = self.font_small.render(ai_status, True, ai_color)
        self.window.blit(ai_surf, (
            (WINDOW_WIDTH - ai_surf.get_width()) // 2,
            WINDOW_HEIGHT - 60
        ))

        # Appuyer pour commencer
        start_surf = self.font_medium.render(
            "Appuyez sur ENTRÉE pour commencer",
            True, DARK_GRAY
        )
        self.window.blit(start_surf, (
            (WINDOW_WIDTH - start_surf.get_width()) // 2,
            WINDOW_HEIGHT - 30
        ))

        pygame.display.flip()

    # ────────────────────────────────────────────────────────────
    # BOUCLE PRINCIPALE
    # ────────────────────────────────────────────────────────────
    def run(self):
        """
        Lance la boucle principale du jeu.
        Gère l'écran de démarrage, le jeu humain et le mode IA.
        """

        # ── Écran de démarrage ────────────────────────────────────
        waiting = True
        while waiting:
            self._draw_start_screen()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        waiting = False
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()

        # ── Initialiser la première partie ────────────────────────
        self._reset_game()

        # ── Boucle de jeu principale ──────────────────────────────
        running = True

        while running:

            # ── Gestion des événements ────────────────────────────
            running, action = self._handle_events()

            # ── Mode IA : action automatique ──────────────────────
            if self.mode == "ai" and not self.game_over:
                action = self._get_ai_action()

            # ── Mise à jour du jeu ────────────────────────────────
            self._update(action)

            # ── Rendu ─────────────────────────────────────────────
            self.window.fill(BG_COLOR)
            self._draw_grid()
            self._draw_info_panel()
            pygame.display.flip()

            # ── FPS selon le mode ─────────────────────────────────
            fps = FPS_AI if self.mode == "ai" else FPS_HUMAN
            self.clock.tick(fps)

        # ── Fin du jeu ────────────────────────────────────────────
        self._print_session_stats()
        pygame.quit()

    # ────────────────────────────────────────────────────────────
    # STATISTIQUES DE SESSION
    # ────────────────────────────────────────────────────────────
    def _print_session_stats(self):
        """Affiche les statistiques de la session de jeu."""
        print("\n" + "=" * 50)
        print("  📊 STATISTIQUES DE SESSION")
        print("=" * 50)
        print(f"  Parties jouées  : {self.session_games}")
        print(f"  Victoires       : {self.session_wins}")

        win_rate = (self.session_wins /
                    max(self.session_games, 1)) * 100
        print(f"  Taux de succès  : {win_rate:.1f}%")

        if self.session_rewards:
            print(f"  Score moyen     : "
                  f"{np.mean(self.session_rewards):+.1f}")
            print(f"  Meilleur score  : "
                  f"{max(self.session_rewards):+.1f}")

        print("=" * 50)


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
def main():
    """
    Fonction principale.
    Lance le jeu en mode humain avec option IA.
    """
    print("=" * 50)
    print("DELIVERY GAME")
    print("=" * 50)
    print("  Contrôles :")
    print("  Flèches : Déplacer le robot")
    print("  Espace : Rester sur place")
    print("  R      : Nouveau jeu")
    print("  A      : Basculer mode IA/Humain")
    print("  Q      : Quitter")
    print("=" * 50)

    # ── Lancer le jeu ─────────────────────────────────────────────
    game = DeliveryGamePlayer(
        agent_path="best_agent.pt",
        grid_size=8,
        max_steps=200
    )
    game.run()


if __name__ == "__main__":
    main()