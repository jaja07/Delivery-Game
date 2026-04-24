import pygame
import random
from .config import WIN_W, WIN_H, CELL, GRID_H, GRID_W, C, FPS, HUD_H, Cell
from .env import RescueBotEnv

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