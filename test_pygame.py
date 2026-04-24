"""
RescueBot Environment — Pygame
================================
Delivery game : pick up survivors and bring them to evacuation zones.

Lancer :
    python rescue_bot_env.py              # mode humain (clavier)
    python rescue_bot_env.py --random     # politique aléatoire

Dépendances :
    pip install pygame numpy
"""

import pygame
import numpy as np
import argparse
import sys
import random
from enum import IntEnum

# ─── Constantes ────────────────────────────────────────────────────────────────

GRID_W, GRID_H = 20, 20
CELL            = 36          # pixels par cellule
HUD_H           = 80          # hauteur de la barre d'info en bas
WIN_W           = GRID_W * CELL
WIN_H           = GRID_H * CELL + HUD_H

FPS             = 10          # frames par seconde (mode auto)

# Couleurs
C = {
    "bg":       (10,  25,  40),
    "empty":    (15,  30,  48),
    "wall":     (45,  35,  20),
    "danger":   (60,  15,  10),
    "evac":     (10,  40,  28),
    "grid":     (12,  22,  35),
    "robot":    (13, 148, 136),
    "robot_c":  (20, 184, 166),   # robot portant quelqu'un
    "survivor": (245, 158,  11),
    "rescued":  (34, 197,  94),
    "hud_bg":   ( 8,  16,  28),
    "text":     (226, 232, 240),
    "muted":    (100, 116, 139),
    "accent":   ( 94, 234, 212),
    "danger_glow": (220, 50, 20),
    "evac_border": (13, 148, 136),
    "fire1":    (255, 100, 30),
    "fire2":    (255, 180, 50),
}

# ─── Types de cellules ─────────────────────────────────────────────────────────

class Cell(IntEnum):
    EMPTY  = 0
    WALL   = 1
    DANGER = 2
    EVAC   = 3

# ─── Actions ───────────────────────────────────────────────────────────────────

class Action(IntEnum):
    UP     = 0
    DOWN   = 1
    LEFT   = 2
    RIGHT  = 3
    PICKUP = 4
    DROP   = 5

ACTION_DELTAS = {
    Action.UP:    (-1,  0),
    Action.DOWN:  ( 1,  0),
    Action.LEFT:  ( 0, -1),
    Action.RIGHT: ( 0,  1),
}

# ─── Carte ─────────────────────────────────────────────────────────────────────

def build_map() -> np.ndarray:
    """Construit une carte 20x20 avec obstacles, dangers et zones d'évacuation."""
    grid = np.zeros((GRID_H, GRID_W), dtype=np.int8)

    def fill(r, c, h, w, t):
        grid[r:r+h, c:c+w] = t

    # Bordures
    grid[0, :]  = Cell.WALL
    grid[-1, :] = Cell.WALL
    grid[:, 0]  = Cell.WALL
    grid[:, -1] = Cell.WALL

    # Murs intérieurs (bâtiments détruits)
    walls = [
        (2, 3, 5, 2), (2, 8, 3, 2), (2, 14, 6, 2),
        (7, 3, 2, 5), (8, 10, 2, 3), (6, 16, 4, 2),
        (12, 2, 2, 4), (11, 8, 3, 2), (12, 13, 3, 3),
        (15, 5, 3, 2), (15, 10, 2, 4), (14, 16, 3, 2),
        (4, 5, 1, 3), (9, 14, 2, 2), (17, 3, 1, 4),
    ]
    for (r, c, h, w) in walls:
        fill(r, c, h, w, Cell.WALL)

    # Zones de danger (feu, radiation)
    dangers = [
        (3, 6, 2, 2), (5, 12, 2, 3), (9, 4, 2, 2),
        (13, 7, 2, 2), (16, 13, 2, 3), (7, 17, 3, 1),
        (4, 16, 2, 2), (11, 5, 1, 3),
    ]
    for (r, c, h, w) in dangers:
        fill(r, c, h, w, Cell.DANGER)

    # Points d'évacuation
    evac_positions = [
        (1, 1), (1, GRID_W - 3),
        (GRID_H - 2, 1), (GRID_H - 2, GRID_W - 3),
        (GRID_H // 2, GRID_W // 2),
    ]
    for (r, c) in evac_positions:
        if grid[r, c] == Cell.EMPTY:
            grid[r, c] = Cell.EVAC

    return grid

# ─── Environnement ─────────────────────────────────────────────────────────────

class RescueBotEnv:
    """
    Environnement MDP pour RescueBot.

    Observation : vecteur numpy de taille 490
        - 484 : fenêtre 11x11 encodée sur 4 canaux (obstacle, danger, survivant, évac)
        - 6   : variables globales (Δr_surv, Δc_surv, Δr_evac, Δc_evac, carrying, n_remaining)

    Actions : 6 (UP, DOWN, LEFT, RIGHT, PICKUP, DROP)

    Récompenses :
        +10  pickup survivant
        +25  dépôt au point d'évacuation
        -5   entrer dans une zone de danger
        -0.5 chaque pas de temps
        -1   action invalide (marcher dans un mur)
        +5   bonus fin d'épisode si au moins 1 évacué
    """

    NUM_SURVIVORS  = 5
    MAX_STEPS      = 500
    PERCEPTION_R   = 5   # rayon de la fenêtre de perception (11x11)

    def __init__(self, seed: int = None):
        self.rng   = random.Random(seed)
        self.grid  = build_map()
        self._free_cells = [
            (r, c)
            for r in range(GRID_H)
            for c in range(GRID_W)
            if self.grid[r, c] == Cell.EMPTY
        ]
        self.reset()

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self):
        pos = self.rng.choice(self._free_cells)
        self.robot_r, self.robot_c = pos
        self.carrying = False

        positions = self.rng.sample(
            [p for p in self._free_cells if p != pos],
            self.NUM_SURVIVORS
        )
        self.survivors = [{"r": r, "c": c, "rescued": False} for r, c in positions]

        self.steps    = 0
        self.rescued  = 0
        self.done     = False
        return self._get_obs()

    def step(self, action: int):
        assert not self.done, "Appelle reset() avant de continuer."
        action  = Action(action)
        reward  = -0.5
        info    = {}
        self.steps += 1

        if action in ACTION_DELTAS:
            dr, dc = ACTION_DELTAS[action]
            nr, nc = self.robot_r + dr, self.robot_c + dc
            if self._in_bounds(nr, nc) and self.grid[nr, nc] != Cell.WALL:
                self.robot_r, self.robot_c = nr, nc
                if self.grid[nr, nc] == Cell.DANGER:
                    reward -= 5.0
                    info["event"] = "danger"
            else:
                reward -= 1.0   # action invalide
                info["event"] = "wall_hit"

        elif action == Action.PICKUP:
            picked = self._survivor_at(self.robot_r, self.robot_c)
            if picked is not None and not self.carrying:
                picked["rescued"] = True
                self.carrying = True
                reward += 10.0
                info["event"] = "pickup"
            else:
                reward -= 1.0

        elif action == Action.DROP:
            if self.carrying and self.grid[self.robot_r, self.robot_c] == Cell.EVAC:
                self.carrying = False
                self.rescued += 1
                reward += 25.0
                info["event"] = "drop_success"
            else:
                reward -= 1.0

        # Fin d'épisode
        all_rescued = all(s["rescued"] for s in self.survivors)
        if all_rescued or self.steps >= self.MAX_STEPS:
            if self.rescued > 0:
                reward += 5.0
            self.done = True

        obs = self._get_obs()
        return obs, reward, self.done, info

    def get_action_mask(self) -> np.ndarray:
        """Retourne un masque booléen sur les 6 actions (True = valide)."""
        mask = np.ones(6, dtype=bool)
        for a, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = self.robot_r + dr, self.robot_c + dc
            if not self._in_bounds(nr, nc) or self.grid[nr, nc] == Cell.WALL:
                mask[a] = False
        surv = self._survivor_at(self.robot_r, self.robot_c)
        mask[Action.PICKUP] = (surv is not None) and (not self.carrying)
        mask[Action.DROP]   = (
            self.carrying and
            self.grid[self.robot_r, self.robot_c] == Cell.EVAC
        )
        return mask

    @property
    def obs_size(self):
        side = 2 * self.PERCEPTION_R + 1
        return side * side * 4 + 6

    @property
    def n_actions(self):
        return 6

    # ── Observation ────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        P = self.PERCEPTION_R
        side = 2 * P + 1

        # 4 canaux : obstacle, danger, survivant, évac
        channels = np.zeros((4, side, side), dtype=np.float32)
        for dr in range(-P, P + 1):
            for dc in range(-P, P + 1):
                nr, nc = self.robot_r + dr, self.robot_c + dc
                ri, ci = dr + P, dc + P
                if not self._in_bounds(nr, nc):
                    channels[0, ri, ci] = 1.0  # hors carte = obstacle
                    continue
                cell = self.grid[nr, nc]
                if cell == Cell.WALL:
                    channels[0, ri, ci] = 1.0
                elif cell == Cell.DANGER:
                    channels[1, ri, ci] = 1.0
                elif cell == Cell.EVAC:
                    channels[3, ri, ci] = 1.0
                s = self._survivor_at(nr, nc)
                if s is not None:
                    channels[2, ri, ci] = 1.0

        local_obs = channels.flatten()   # 484 valeurs

        # Variables globales
        active = [s for s in self.survivors if not s["rescued"]]
        if active:
            nearest = min(active, key=lambda s: abs(s["r"] - self.robot_r) + abs(s["c"] - self.robot_c))
            dr_s = np.clip((nearest["r"] - self.robot_r) / GRID_H, -1, 1)
            dc_s = np.clip((nearest["c"] - self.robot_c) / GRID_W, -1, 1)
        else:
            dr_s, dc_s = 0.0, 0.0

        evac_cells = [(r, c) for r in range(GRID_H) for c in range(GRID_W) if self.grid[r, c] == Cell.EVAC]
        if evac_cells:
            nearest_evac = min(evac_cells, key=lambda p: abs(p[0] - self.robot_r) + abs(p[1] - self.robot_c))
            dr_e = np.clip((nearest_evac[0] - self.robot_r) / GRID_H, -1, 1)
            dc_e = np.clip((nearest_evac[1] - self.robot_c) / GRID_W, -1, 1)
        else:
            dr_e, dc_e = 0.0, 0.0

        global_obs = np.array([
            dr_s, dc_s,
            dr_e, dc_e,
            float(self.carrying),
            len(active) / self.NUM_SURVIVORS,
        ], dtype=np.float32)

        return np.concatenate([local_obs, global_obs])   # 490 valeurs

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _in_bounds(self, r, c) -> bool:
        return 0 <= r < GRID_H and 0 <= c < GRID_W

    def _survivor_at(self, r, c):
        for s in self.survivors:
            if not s["rescued"] and s["r"] == r and s["c"] == c:
                return s
        return None


# ─── Rendu Pygame ──────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self, env: RescueBotEnv):
        pygame.init()
        self.env    = env
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("RescueBot — Policy Gradient Environment")
        self.clock  = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 11)
        self.font_m = pygame.font.SysFont("monospace", 13, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 16, bold=True)
        self._fire_tick = 0

    def draw(self, total_reward: float = 0.0, last_event: str = ""):
        self._fire_tick += 1
        self.screen.fill(C["bg"])
        self._draw_grid()
        self._draw_evac_zones()
        self._draw_danger_zones()
        self._draw_survivors()
        self._draw_robot()
        self._draw_perception_overlay()
        self._draw_hud(total_reward, last_event)
        pygame.display.flip()

    def _draw_grid(self):
        for r in range(GRID_H):
            for c in range(GRID_W):
                rect = pygame.Rect(c * CELL, r * CELL, CELL, CELL)
                cell = self.env.grid[r, c]
                if cell == Cell.WALL:
                    pygame.draw.rect(self.screen, C["wall"], rect)
                    # Texture débris
                    for _ in range(3):
                        bx = rect.x + random.randint(2, CELL - 6)
                        by = rect.y + random.randint(2, CELL - 6)
                        pygame.draw.rect(self.screen, (70, 55, 30), (bx, by, 5, 2))
                elif cell == Cell.EMPTY:
                    pygame.draw.rect(self.screen, C["empty"], rect)
                pygame.draw.rect(self.screen, C["grid"], rect, 1)

    def _draw_danger_zones(self):
        tick = self.env.steps
        for r in range(GRID_H):
            for c in range(GRID_W):
                if self.env.grid[r, c] == Cell.DANGER:
                    rect = pygame.Rect(c * CELL, r * CELL, CELL, CELL)
                    pygame.draw.rect(self.screen, C["danger"], rect)
                    # Effet feu animé
                    alpha = 120 + int(60 * abs(((tick + r + c) % 20) / 10 - 1))
                    s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                    s.fill((*C["danger_glow"], alpha))
                    self.screen.blit(s, rect.topleft)
                    # Flammes
                    cx, cy = rect.centerx, rect.centery
                    h = CELL // 2 - 2 + int(4 * abs(((tick * 2 + r) % 10) / 5 - 1))
                    points = [(cx - 4, cy + 4), (cx, cy - h), (cx + 4, cy + 4)]
                    pygame.draw.polygon(self.screen, C["fire1"], points)
                    points2 = [(cx - 2, cy + 2), (cx, cy - h + 4), (cx + 2, cy + 2)]
                    pygame.draw.polygon(self.screen, C["fire2"], points2)

    def _draw_evac_zones(self):
        for r in range(GRID_H):
            for c in range(GRID_W):
                if self.env.grid[r, c] == Cell.EVAC:
                    rect = pygame.Rect(c * CELL, r * CELL, CELL, CELL)
                    pygame.draw.rect(self.screen, C["evac"], rect)
                    pygame.draw.rect(self.screen, C["evac_border"], rect, 2)
                    # Croix médicale
                    cx, cy = rect.centerx, rect.centery
                    hw = 4
                    pygame.draw.rect(self.screen, C["accent"], (cx - hw, cy - 1, hw * 2, 3))
                    pygame.draw.rect(self.screen, C["accent"], (cx - 1, cy - hw, 3, hw * 2))
                    txt = self.font_s.render("EVAC", True, C["accent"])
                    self.screen.blit(txt, (rect.x + 2, rect.bottom - 13))

    def _draw_survivors(self):
        for s in self.env.survivors:
            if s["rescued"]:
                continue
            cx = s["c"] * CELL + CELL // 2
            cy = s["r"] * CELL + CELL // 2
            r  = CELL // 2 - 4
            pygame.draw.circle(self.screen, C["survivor"], (cx, cy), r)
            pygame.draw.circle(self.screen, (252, 211, 77), (cx, cy), r, 2)
            # Icône personne
            pygame.draw.circle(self.screen, (30, 20, 5), (cx, cy - 3), 3)
            pygame.draw.line(self.screen, (30, 20, 5), (cx, cy - 1), (cx, cy + 5), 2)
            txt = self.font_s.render("S", True, (30, 20, 5))
            self.screen.blit(txt, (cx - 3, cy + 3))

    def _draw_robot(self):
        env = self.env
        cx  = env.robot_c * CELL + CELL // 2
        cy  = env.robot_r * CELL + CELL // 2
        r   = CELL // 2 - 3
        col = C["robot_c"] if env.carrying else C["robot"]
        # Corps
        pygame.draw.circle(self.screen, col, (cx, cy), r)
        pygame.draw.circle(self.screen, C["accent"], (cx, cy), r, 2)
        # Yeux
        pygame.draw.circle(self.screen, (255, 255, 255), (cx - 4, cy - 2), 3)
        pygame.draw.circle(self.screen, (255, 255, 255), (cx + 4, cy - 2), 3)
        pygame.draw.circle(self.screen, (10, 10, 10), (cx - 4, cy - 2), 1)
        pygame.draw.circle(self.screen, (10, 10, 10), (cx + 4, cy - 2), 1)
        # Bouche
        pygame.draw.arc(self.screen, (255, 255, 255),
                        (cx - 4, cy + 1, 8, 5), 3.14, 0, 2)
        # Survivant porté
        if env.carrying:
            pygame.draw.circle(self.screen, C["survivor"], (cx + r - 2, cy - r + 2), 4)
        # Label
        lbl = self.font_s.render("BOT", True, (10, 10, 10))
        self.screen.blit(lbl, (cx - 8, cy + 6))

    def _draw_perception_overlay(self):
        """Dessine la fenêtre de perception du robot (11x11)."""
        P   = self.env.PERCEPTION_R
        env = self.env
        s   = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        s.fill((94, 234, 212, 18))
        for dr in range(-P, P + 1):
            for dc in range(-P, P + 1):
                nr, nc = env.robot_r + dr, env.robot_c + dc
                if 0 <= nr < GRID_H and 0 <= nc < GRID_W:
                    self.screen.blit(s, (nc * CELL, nr * CELL))
        # Bordure
        ox = (env.robot_c - P) * CELL
        oy = (env.robot_r - P) * CELL
        pygame.draw.rect(self.screen, (*C["accent"], 180),
                         (ox, oy, CELL * (2*P+1), CELL * (2*P+1)), 1)

    def _draw_hud(self, total_reward: float, last_event: str):
        hud_y = GRID_H * CELL
        pygame.draw.rect(self.screen, C["hud_bg"], (0, hud_y, WIN_W, HUD_H))
        pygame.draw.line(self.screen, C["accent"], (0, hud_y), (WIN_W, hud_y), 1)

        env = self.env
        remaining = sum(1 for s in env.survivors if not s["rescued"])

        items = [
            ("Pas",       str(env.steps)),
            ("Évacués",   str(env.rescued)),
            ("Restants",  str(remaining)),
            ("Porte",     "Oui" if env.carrying else "Non"),
            ("Reward",    f"{total_reward:.1f}"),
            ("Événement", last_event or "—"),
        ]
        x = 10
        for label, val in items:
            lbl = self.font_s.render(label, True, C["muted"])
            val_txt = self.font_m.render(val, True, C["accent"])
            self.screen.blit(lbl,     (x, hud_y + 10))
            self.screen.blit(val_txt, (x, hud_y + 26))
            x += max(lbl.get_width(), val_txt.get_width()) + 20

        # Légende touches
        hint = self.font_s.render(
            "↑↓←→ déplacer  |  P pickup  |  D drop  |  R reset  |  Q quitter",
            True, C["muted"]
        )
        self.screen.blit(hint, (10, hud_y + 58))

    def tick(self, fps: int = FPS):
        self.clock.tick(fps)

    def close(self):
        pygame.quit()


# ─── Boucle principale ─────────────────────────────────────────────────────────

def run_human(env: RescueBotEnv, renderer: Renderer):
    """Mode joueur humain — contrôle au clavier."""
    total_reward = 0.0
    last_event   = ""
    obs          = env.reset()
    renderer.draw(total_reward, last_event)

    while True:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:    action = Action.UP
                if event.key == pygame.K_DOWN:  action = Action.DOWN
                if event.key == pygame.K_LEFT:  action = Action.LEFT
                if event.key == pygame.K_RIGHT: action = Action.RIGHT
                if event.key == pygame.K_p:     action = Action.PICKUP
                if event.key == pygame.K_d:     action = Action.DROP
                if event.key == pygame.K_r:
                    obs = env.reset(); total_reward = 0.0; last_event = "reset"
                if event.key == pygame.K_q:
                    renderer.close(); sys.exit()

        if action is not None:
            obs, reward, done, info = env.step(action)
            total_reward += reward
            last_event    = info.get("event", "")
            if done:
                last_event = "DONE — appuie R pour rejouer"

        renderer.draw(total_reward, last_event)
        renderer.tick(30)


def run_random(env: RescueBotEnv, renderer: Renderer):
    """Mode politique aléatoire — pour tester l'environnement."""
    obs          = env.reset()
    total_reward = 0.0
    last_event   = ""
    episode      = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.close(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                renderer.close(); sys.exit()

        mask   = env.get_action_mask()
        valid  = [a for a in range(6) if mask[a]]
        action = random.choice(valid) if valid else 0

        obs, reward, done, info = env.step(action)
        total_reward += reward
        last_event    = info.get("event", "")

        renderer.draw(total_reward, last_event)
        renderer.tick(FPS)

        if done:
            episode += 1
            print(f"Épisode {episode:4d} | Récompense : {total_reward:7.1f} | Évacués : {env.rescued}")
            obs          = env.reset()
            total_reward = 0.0
            last_event   = "nouvel épisode"


# ─── PolicyNet (pour branchement RL) ──────────────────────────────────────────

POLICY_NET_CODE = '''
# ── À importer dans ton script d'entraînement ──────────────────────────────
import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    """
    Réseau de politique pour RescueBot.
    Entrée  : vecteur d'observation de taille 490
    Sortie  : distribution Categorical sur 6 actions
    """
    def __init__(self, obs_size: int = 490, n_actions: int = 6):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(64, n_actions)

    def forward(self, obs, action_mask=None):
        """
        obs         : Tensor (batch, obs_size)
        action_mask : Tensor bool (batch, n_actions) — True = action valide
        Retourne    : distribution Categorical
        """
        features = self.trunk(obs)
        logits   = self.policy_head(features)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))
        return Categorical(logits=logits)


def compute_returns(rewards, gamma=0.99):
    """Calcule les retours Gₜ avec facteur d'escompte γ."""
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def reinforce_update(policy, optimizer, log_probs, rewards, gamma=0.99):
    """Une mise à jour REINFORCE sur une trajectoire complète."""
    returns = compute_returns(rewards, gamma)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalisation

    loss = -sum(lp * G for lp, G in zip(log_probs, returns))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
'''

# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true", help="Politique aléatoire")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    env      = RescueBotEnv(seed=args.seed)
    renderer = Renderer(env)

    print(f"Taille observation : {env.obs_size}")
    print(f"Nombre d'actions   : {env.n_actions}")
    print()
    print("PolicyNet code disponible dans POLICY_NET_CODE (variable en haut de ce fichier)")
    print()

    if args.random:
        run_random(env, renderer)
    else:
        run_human(env, renderer)