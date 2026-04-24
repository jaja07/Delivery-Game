# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:38:34 2026

@author: yasmi
"""

"""
Delivery Game Environment - Version Corrigée v2
Corrections :
  1. Pénalité de temps plus forte
  2. Pénalité pour mouvements répétitifs
  3. Reward shaping plus précis
  4. Pénalité pour tourner en rond
"""

import numpy as np
import pygame
from enum import Enum
from typing import Tuple, Dict, Any
from collections import deque


class Actions(Enum):
    """Actions possibles pour le robot"""
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3
    STAY  = 4


class DeliveryGame:
    """
    Environnement du jeu Delivery Game - Version Corrigée v2.

    Corrections principales :
    - Pénalité de temps calibrée
    - Détection des mouvements répétitifs
    - Reward shaping directionnel
    - Historique des positions
    """

    def __init__(self,
                 grid_size: int = 8,
                 max_steps: int = 200,
                 num_obstacles: int = 5,
                 seed: int = None):

        self.grid_size     = grid_size
        self.max_steps     = max_steps
        self.num_obstacles = num_obstacles

        self.action_space_size = 5
        self.state_space_size  = 7

        if seed is not None:
            np.random.seed(seed)

        # ── Variables d'état ──────────────────────────────────────
        self.robot_pos    = None
        self.package_pos  = None
        self.goal_pos     = None
        self.obstacles    = set()
        self.has_package  = False
        self.current_step = 0
        self.done         = False

        # ── Historique des positions (détection boucles) ──────────
        # Garde les 8 dernières positions
        self.position_history = deque(maxlen=8)

        # ── Distance précédente (pour reward shaping) ─────────────
        self.prev_dist_to_target = 0

        self.reset()

    # ────────────────────────────────────────────────────────────
    # GÉNÉRATION
    # ────────────────────────────────────────────────────────────
    def _generate_obstacles(self):
        """Génère des obstacles aléatoires."""
        self.obstacles = set()
        attempts = 0
        while len(self.obstacles) < self.num_obstacles and attempts < 200:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            self.obstacles.add((x, y))
            attempts += 1

    def _generate_random_position(self,
                                   excluded: list = None) -> Tuple[int, int]:
        """Génère une position libre non occupée."""
        excluded = excluded or []
        attempts = 0
        while attempts < 1000:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            if (x, y) not in self.obstacles and (x, y) not in excluded:
                return (x, y)
            attempts += 1

        # Fallback
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.obstacles and (x, y) not in excluded:
                    return (x, y)
        return (0, 0)

    # ────────────────────────────────────────────────────────────
    # RESET
    # ────────────────────────────────────────────────────────────
    def reset(self) -> np.ndarray:
        """Réinitialise l'environnement."""
        self._generate_obstacles()

        self.robot_pos   = self._generate_random_position(excluded=[])
        self.package_pos = self._generate_random_position(
            excluded=[self.robot_pos]
        )
        self.goal_pos    = self._generate_random_position(
            excluded=[self.robot_pos, self.package_pos]
        )

        self.has_package  = False
        self.current_step = 0
        self.done         = False

        # ── Réinitialiser l'historique ────────────────────────────
        self.position_history.clear()
        self.position_history.append(self.robot_pos)

        # ── Distance initiale à la cible ──────────────────────────
        self.prev_dist_to_target = self._dist_to_target()

        return self._get_state()

    # ────────────────────────────────────────────────────────────
    # DISTANCE À LA CIBLE COURANTE
    # ────────────────────────────────────────────────────────────
    def _dist_to_target(self) -> float:
        """
        Calcule la distance Manhattan à la cible courante.
        Cible = colis si pas encore ramassé, objectif sinon.
        """
        rx, ry = self.robot_pos

        if not self.has_package:
            tx, ty = self.package_pos
        else:
            tx, ty = self.goal_pos

        return abs(tx - rx) + abs(ty - ry)

    # ────────────────────────────────────────────────────────────
    # DÉTECTION DE BOUCLES
    # ────────────────────────────────────────────────────────────
    def _is_looping(self) -> bool:
        """
        Détecte si l'agent tourne en rond.

        Retourne True si la position actuelle a déjà été
        visitée plusieurs fois dans l'historique récent.
        """
        if len(self.position_history) < 4:
            return False

        # Compter les occurrences de la position actuelle
        current = self.robot_pos
        count   = sum(1 for pos in self.position_history
                      if pos == current)

        # Si la position apparaît 3+ fois → boucle détectée
        return count >= 3

    # ────────────────────────────────────────────────────────────
    # STATE
    # ────────────────────────────────────────────────────────────
    def _get_state(self) -> np.ndarray:
        """
        Encode l'état avec positions relatives normalisées.

        State = [
            dx_to_target / grid,   # Direction X vers cible
            dy_to_target / grid,   # Direction Y vers cible
            dx_to_goal   / grid,   # Direction X vers objectif
            dy_to_goal   / grid,   # Direction Y vers objectif
            has_package,           # 0 ou 1
            dist_to_target / max,  # Distance normalisée cible
            dist_to_goal   / max   # Distance normalisée objectif
        ]
        """
        rx, ry = self.robot_pos

        # Cible selon la phase
        if not self.has_package:
            tx, ty = self.package_pos
        else:
            tx, ty = self.goal_pos

        gx, gy = self.goal_pos

        # Positions relatives normalisées
        dx_target = (tx - rx) / self.grid_size
        dy_target = (ty - ry) / self.grid_size
        dx_goal   = (gx - rx) / self.grid_size
        dy_goal   = (gy - ry) / self.grid_size

        # Distances normalisées
        dist_target = (abs(tx - rx) + abs(ty - ry)) / (2 * self.grid_size)
        dist_goal   = (abs(gx - rx) + abs(gy - ry)) / (2 * self.grid_size)

        return np.array([
            dx_target,
            dy_target,
            dx_goal,
            dy_goal,
            float(self.has_package),
            dist_target,
            dist_goal
        ], dtype=np.float32)

    # ────────────────────────────────────────────────────────────
    # REWARD SHAPING CORRIGÉ
    # ────────────────────────────────────────────────────────────
    def _calculate_reward(self,
                          action: int,
                          picked_package: bool,
                          delivered: bool,
                          hit_wall: bool,
                          moved: bool) -> float:
        """
        Calcule la récompense corrigée.

        Système de récompenses :
        ┌─────────────────────────────────────────────┐
        │ Pénalité de temps        : -0.5  par step   │
        │ Pénalité mur             : -1.0             │
        │ Pénalité boucle          : -2.0             │
        │ Rapprochement cible      : +1.0             │
        │ Éloignement cible        : -1.0             │
        │ Ramasser colis           : +10.0            │
        │ Livrer colis             : +50.0            │
        │ Bonus rapidité livraison : +variable        │
        └─────────────────────────────────────────────┘

        Args:
            action         : Action effectuée
            picked_package : Vient de ramasser le colis ?
            delivered      : Vient de livrer ?
            hit_wall       : A heurté un mur ?
            moved          : S'est effectivement déplacé ?
        """
        reward = 0.0

        # ── 1. Pénalité de temps (plus forte) ────────────────────
        # Force l'agent à être efficace
        reward -= 0.5

        # ── 2. Pénalité pour mur ──────────────────────────────────
        if hit_wall:
            reward -= 1.0

        # ── 3. Pénalité pour boucle ───────────────────────────────
        # Décourage fortement les allers-retours inutiles
        if self._is_looping():
            reward -= 2.0

        # ── 4. Reward shaping directionnel ────────────────────────
        # Récompense/pénalise selon rapprochement ou éloignement
        curr_dist = self._dist_to_target()
        dist_diff = self.prev_dist_to_target - curr_dist

        if dist_diff > 0:
            # ✅ Se rapproche de la cible
            reward += 1.0 * dist_diff
        elif dist_diff < 0:
            # ❌ S'éloigne de la cible
            reward += 1.0 * dist_diff  # Négatif

        # Mettre à jour la distance précédente
        self.prev_dist_to_target = curr_dist

        # ── 5. Bonus ramasser le colis ────────────────────────────
        if picked_package:
            reward += 10.0
            # Reset la distance pour la phase 2
            self.prev_dist_to_target = self._dist_to_target()

        # ── 6. Bonus livraison + bonus rapidité ───────────────────
        if delivered:
            # Bonus de base
            reward += 50.0

            # Bonus rapidité : plus on livre vite, plus c'est récompensé
            steps_remaining = self.max_steps - self.current_step
            speed_bonus     = (steps_remaining / self.max_steps) * 20.0
            reward         += speed_bonus

        return reward

    # ────────────────────────────────────────────────────────────
    # STEP
    # ────────────────────────────────────────────────────────────
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Exécute une action dans l'environnement.

        Args:
            action : Indice de l'action (0-4)
        Returns:
            (state, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Épisode terminé. Appelez reset().")

        action = action % self.action_space_size

        # ── Calcul du mouvement ───────────────────────────────────
        dx, dy = 0, 0
        if   action == Actions.UP.value:    dy = -1
        elif action == Actions.DOWN.value:  dy =  1
        elif action == Actions.LEFT.value:  dx = -1
        elif action == Actions.RIGHT.value: dx =  1
        # STAY → dx=dy=0

        new_x = self.robot_pos[0] + dx
        new_y = self.robot_pos[1] + dy

        # ── Vérification limites ──────────────────────────────────
        new_x = max(0, min(new_x, self.grid_size - 1))
        new_y = max(0, min(new_y, self.grid_size - 1))

        # ── Vérification obstacles ────────────────────────────────
        hit_wall = False
        moved    = False

        if (new_x, new_y) in self.obstacles:
            hit_wall = True
            # Reste en place
        elif (new_x, new_y) != self.robot_pos:
            self.robot_pos = (new_x, new_y)
            moved = True

        # ── Mise à jour historique ────────────────────────────────
        self.position_history.append(self.robot_pos)

        # ── Collecte du colis ─────────────────────────────────────
        picked_package = False
        if not self.has_package and self.robot_pos == self.package_pos:
            self.has_package = True
            picked_package   = True

        # ── Livraison ─────────────────────────────────────────────
        delivered = False
        if self.has_package and self.robot_pos == self.goal_pos:
            delivered = True
            self.done = True

        # ── Calcul récompense ─────────────────────────────────────
        reward = self._calculate_reward(
            action, picked_package, delivered, hit_wall, moved
        )

        # ── Incrément steps ───────────────────────────────────────
        self.current_step += 1
        if self.current_step >= self.max_steps and not self.done:
            self.done = True

        state = self._get_state()

        info = {
            'has_package'   : self.has_package,
            'picked_package': picked_package,
            'delivered'     : delivered,
            'hit_wall'      : hit_wall,
            'moved'         : moved,
            'step'          : self.current_step,
            'dist_to_target': self._dist_to_target()
        }

        return state, reward, self.done, info

    # ────────────────────────────────────────────────────────────
    # RENDER
    # ────────────────────────────────────────────────────────────
    def render(self, window_surface, cell_size: int = 50):
        """
        Affiche le jeu sur une surface pygame.

        Args:
            window_surface : Surface pygame
            cell_size      : Taille de chaque cellule en pixels
        """
        # ── Couleurs ──────────────────────────────────────────────
        WHITE      = (255, 255, 255)
        BLACK      = (0,   0,   0)
        GRAY       = (180, 180, 180)
        DARK_GRAY  = (80,  80,  80)
        BLUE       = (0,   100, 255)
        GREEN      = (0,   200, 0)
        DARK_GREEN = (0,   140, 0)
        YELLOW     = (255, 220, 0)
        ORANGE     = (255, 140, 0)

        # ── Fond blanc ────────────────────────────────────────────
        grid_rect = pygame.Rect(
            0, 0,
            self.grid_size * cell_size,
            self.grid_size * cell_size
        )
        pygame.draw.rect(window_surface, WHITE, grid_rect)

        # ── Obstacles ─────────────────────────────────────────────
        for obs_x, obs_y in self.obstacles:
            rect = pygame.Rect(
                obs_x * cell_size,
                obs_y * cell_size,
                cell_size, cell_size
            )
            pygame.draw.rect(window_surface, DARK_GRAY, rect)

            # Texture hachurée
            for i in range(0, cell_size, 8):
                pygame.draw.line(
                    window_surface, GRAY,
                    (obs_x * cell_size, obs_y * cell_size + i),
                    (obs_x * cell_size + i, obs_y * cell_size), 1
                )

        # ── Zone objectif (vert) ──────────────────────────────────
        gx, gy    = self.goal_pos
        goal_rect = pygame.Rect(
            gx * cell_size, gy * cell_size,
            cell_size, cell_size
        )
        pygame.draw.rect(window_surface, GREEN, goal_rect)
        pygame.draw.rect(window_surface, DARK_GREEN, goal_rect, 3)

        # Lettre G dans la zone objectif
        try:
            font = pygame.font.Font(None, cell_size // 2 + 4)
            g_surf = font.render("G", True, WHITE)
            window_surface.blit(g_surf, (
                gx * cell_size + cell_size // 3,
                gy * cell_size + cell_size // 4
            ))
        except Exception:
            pass

        # ── Colis (jaune) si pas encore ramassé ───────────────────
        if not self.has_package:
            px, py   = self.package_pos
            pkg_rect = pygame.Rect(
                px * cell_size, py * cell_size,
                cell_size, cell_size
            )
            pygame.draw.rect(window_surface, YELLOW, pkg_rect)
            pygame.draw.rect(window_surface, ORANGE, pkg_rect, 3)

            # Lettre P dans le colis
            try:
                p_surf = font.render("P", True, BLACK)
                window_surface.blit(p_surf, (
                    px * cell_size + cell_size // 3,
                    py * cell_size + cell_size // 4
                ))
            except Exception:
                pass

        # ── Trajectoire récente (visualisation) ───────────────────
        # Affiche les 4 dernières positions en bleu clair
        history_list = list(self.position_history)
        for i, pos in enumerate(history_list[:-1]):
            alpha = int(80 * (i + 1) / len(history_list))
            hx, hy = pos
            trail_rect = pygame.Rect(
                hx * cell_size + cell_size // 4,
                hy * cell_size + cell_size // 4,
                cell_size // 2,
                cell_size // 2
            )
            pygame.draw.rect(
                window_surface,
                (173, 216, 230),  # Bleu clair
                trail_rect
            )

        # ── Robot ─────────────────────────────────────────────────
        rx, ry      = self.robot_pos
        robot_color = ORANGE if self.has_package else BLUE
        robot_rect  = pygame.Rect(
            rx * cell_size, ry * cell_size,
            cell_size, cell_size
        )
        pygame.draw.rect(window_surface, robot_color, robot_rect)
        pygame.draw.rect(window_surface, BLACK, robot_rect, 2)

        # Petit carré jaune si le robot porte le colis
        if self.has_package:
            margin     = cell_size // 4
            small_rect = pygame.Rect(
                rx * cell_size + margin,
                ry * cell_size + margin,
                cell_size - 2 * margin,
                cell_size - 2 * margin
            )
            pygame.draw.rect(window_surface, YELLOW, small_rect)

        # Lettre R dans le robot
        try:
            r_surf = font.render("R", True, WHITE)
            window_surface.blit(r_surf, (
                rx * cell_size + cell_size // 3,
                ry * cell_size + cell_size // 4
            ))
        except Exception:
            pass

        # ── Lignes de la grille ───────────────────────────────────
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                window_surface, GRAY,
                (x * cell_size, 0),
                (x * cell_size, self.grid_size * cell_size), 1
            )
        for y in range(self.grid_size + 1):
            pygame.draw.line(
                window_surface, GRAY,
                (0, y * cell_size),
                (self.grid_size * cell_size, y * cell_size), 1
            )

    def get_info(self) -> Dict[str, Any]:
        """Retourne les informations sur l'état actuel."""
        return {
            'robot_pos'   : self.robot_pos,
            'package_pos' : self.package_pos,
            'goal_pos'    : self.goal_pos,
            'has_package' : self.has_package,
            'current_step': self.current_step,
            'max_steps'   : self.max_steps,
            'done'        : self.done,
            'is_looping'  : self._is_looping()
        }