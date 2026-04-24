import numpy as np
import random
from .config import GRID_W, GRID_H, Cell, Action, ACTION_DELTAS

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

    def __init__(self, seed: int = None): # type: ignore
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