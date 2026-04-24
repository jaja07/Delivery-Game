from enum import IntEnum

# Constantes de la grille
GRID_W, GRID_H = 20, 20
CELL           = 36
HUD_H          = 80
WIN_W          = GRID_W * CELL
WIN_H          = GRID_H * CELL + HUD_H
FPS            = 10

# Couleurs
C = {
    "bg":       (10,  25,  40),
    "empty":    (15,  30,  48),
    "wall":     (45,  35,  20),
    "danger":   (60,  15,  10),
    "evac":     (10,  40,  28),
    "grid":     (12,  22,  35),
    "robot":    (13, 148, 136),
    "robot_c":  (20, 184, 166),
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

# Types de cellules
class Cell(IntEnum):
    EMPTY  = 0
    WALL   = 1
    DANGER = 2
    EVAC   = 3

# Actions
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