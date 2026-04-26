import numpy as np
import random
from .config import GRID_W, GRID_H, Cell, Action, ACTION_DELTAS

def build_map() -> np.ndarray:
    """Build a 20x20 map with obstacles, dangers and evacuation zones."""
    grid = np.zeros((GRID_H, GRID_W), dtype=np.int8)

    def fill(r, c, h, w, t):
        """Fill a rectangle in the grid with type t."""
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
            """
            Initialise l'environnement. Si seed est donné, la carte et les positions seront toujours les mêmes.
            """
            self.rng   = random.Random(seed)
            self.grid  = build_map()
            
            self.free_cells = [
                (x, y)
                for x in range(GRID_W)
                for y in range(GRID_H)
                if self.grid[y, x] == Cell.EMPTY
            ]
            self.reset()

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self):
        """
        Remet l'environnement à zéro. Place le robot et les survivants, réinitialise les compteurs.
        Retourne l'observation initiale.
        """
        # Le robot et les survivants sont placés aléatoirement sur des cases vides.
        robot_initial_position = self.rng.choice(self.free_cells)
        self.robot_x, self.robot_y = robot_initial_position
        self.carrying = False

        survivors_initial_positions = self.rng.sample(
            [p for p in self.free_cells if p != robot_initial_position],
            self.NUM_SURVIVORS
        )
        # On crée une liste de dictionnaires pour suivre l'état de chaque survivant
        self.survivors = [{"row": y, "col": x, "rescued": False} for x, y in survivors_initial_positions]
        # Réinitialisation des compteurs
        self.steps    = 0
        self.rescued  = 0
        self.done     = False

        # Initialisation du radar de distance
        self.previous_distance = self._get_target_distance()
        
        return self._get_obs() 

    def step(self, action: int):
        """
        Exécute une action de l'agent et fait évoluer l'environnement d'un pas.

        L'action reçue est convertie en :class:`Action` puis résolue selon trois
        catégories de comportement : déplacement, ramassage et dépôt.

        Paramètres
        ----------
        action : int
                Index d'action attendu par l'algorithme d'apprentissage par renforcement.
                Les valeurs possibles correspondent à :
                - 0 : ``Action.UP``
                - 1 : ``Action.DOWN``
                - 2 : ``Action.LEFT``
                - 3 : ``Action.RIGHT``
                - 4 : ``Action.PICKUP``
                - 5 : ``Action.DROP``

        Retours
        -------
        tuple
                ``(obs, reward, done, info)`` où :
                - ``obs`` est la nouvelle observation de l'environnement,
                - ``reward`` est la récompense instantanée obtenue,
                - ``done`` indique si l'épisode est terminé,
                - ``info`` contient éventuellement un événement descriptif
                    (par exemple ``"wall_hit"``, ``"danger"``, ``"pickup"`` ou
                    ``"drop_success"``).

        Effets de bord
        --------------
        - incrémente le compteur de pas,
        - met à jour la position du robot si le déplacement est valide,
        - modifie l'état des survivants et les compteurs de sauvetage,
        - marque la fin de partie lorsque tous les survivants sont sauvés ou
            lorsque ``MAX_STEPS`` est atteint.

        Notes
        -----
        Un déplacement dans un mur ou une action de ramassage/dépôt invalide
        entraîne une pénalité. Marcher sur une case ``Cell.DANGER`` applique
        également une pénalité supplémentaire.
        """
        assert not self.done
        action  = Action(action)
        reward  = -0.5
        info    = {}
        self.steps += 1

        # --- GESTION DES DÉPLACEMENTS (Haut, Bas, Gauche, Droite) ---
        if action in ACTION_DELTAS:
            # ACTION_DELTAS est défini en (delta_ligne, delta_colonne) = (dr, dc).
            dr, dc = ACTION_DELTAS[action]
            new_y, new_x = self.robot_y + dr, self.robot_x + dc

            if self._in_bounds(new_y, new_x) and self.grid[new_y, new_x] != Cell.WALL:
                self.robot_x, self.robot_y = new_x, new_y
                
                if self.grid[new_y, new_x] == Cell.DANGER:
                    reward -= 5.0
                    info["event"] = "danger"
            else:
                reward -= 1.0
                info["event"] = "wall_hit"

        # --- GESTION DU RAMASSAGE ---
        elif action == Action.PICKUP:
            picked = self._survivor_at(self.robot_y, self.robot_x)
            if picked is not None and not self.carrying:
                picked["rescued"] = True
                self.carrying = True
                reward += 10.0
                info["event"] = "pickup"
                # NOUVEAU : On réinitialise le radar car l'objectif vient de changer !
                self.previous_distance = self._get_target_distance()
            else:
                reward -= 1.0

        # --- GESTION DU DÉPÔT ---
        elif action == Action.DROP:
            if self.carrying and self.grid[self.robot_y, self.robot_x] == Cell.EVAC:
                self.carrying = False
                self.rescued += 1
                reward += 25.0
                info["event"] = "drop_success"
                # NOUVEAU : On réinitialise le radar car l'objectif vient de changer !
                self.previous_distance = self._get_target_distance()
            else:
                reward -= 1.0
        
        # --- REWARD SHAPING : Le jeu du Chaud/Froid ---
        current_distance = self._get_target_distance()
        distance_diff = self.previous_distance - current_distance
        
        # distance_diff est positif si on s'est rapproché, négatif si on s'est éloigné
        reward += distance_diff * 0.2 
        
        self.previous_distance = current_distance

        # --- VÉRIFICATION DE FIN DE PARTIE ---
        all_evacuated = self.rescued >= self.NUM_SURVIVORS

        if all_evacuated or self.steps >= self.MAX_STEPS:
            if self.rescued > 0:
                reward += 5.0
            self.done = True

        obs = self._get_obs()
        
        return obs, reward, self.done, info

    def get_action_mask(self) -> np.ndarray:
        """
        Retourne un masque booléen sur les 6 actions (True = valide).
        Utile pour empêcher le réseau de neurones de calculer des probabilités 
        pour des actions qui sont physiquement impossibles.
        """
        mask = np.ones(6, dtype=bool)
        for a, (dr, dc) in ACTION_DELTAS.items():
            new_y, new_x = self.robot_y + dr, self.robot_x + dc
            if not self._in_bounds(new_y, new_x) or self.grid[new_y, new_x] == Cell.WALL:
                mask[a] = False # Bloque les directions menant à un mur
                
        surv = self._survivor_at(self.robot_y, self.robot_x)
        # On ne peut ramasser que s'il y a un survivant ET qu'on a les mains libres
        mask[Action.PICKUP] = (surv is not None) and (not self.carrying)
        
        # On ne peut déposer que si on porte quelqu'un ET qu'on est sur une zone d'évac
        mask[Action.DROP]   = (
            self.carrying and
            self.grid[self.robot_y, self.robot_x] == Cell.EVAC
        )
        return mask

    @property
    def obs_size(self):
        # 11 * 11 * 4 canaux + 6 variables globales = 490
        side = 2 * self.PERCEPTION_R + 1
        return side * side * 4 + 6

    @property
    def n_actions(self):
        return 6

    # ── Observation ────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
                Construit l'observation courante renvoyée à l'agent.

                L'observation est un vecteur ``np.float32`` de taille ``obs_size`` (490)
                composé de deux blocs concaténés :

                1) Observation locale (484 valeurs)
                     - Fenêtre centrée sur le robot de taille ``(2 * PERCEPTION_R + 1)^2``,
                         soit 11x11 avec ``PERCEPTION_R = 5``.
                     - Encodage en 4 canaux binaires (ordre fixe) :
                         - canal 0 : murs,
                         - canal 1 : zones de danger,
                         - canal 2 : survivants non secourus,
                         - canal 3 : zones d'évacuation.
                     - Les cases hors carte sont traitées comme des murs.
                     - Le tenseur ``(4, 11, 11)`` est aplati en un vecteur de 484 valeurs.

                2) Observation globale (6 valeurs)
                     - ``dr_s, dc_s`` : direction normalisée vers le survivant actif le plus
                         proche (distance de Manhattan), bornée dans ``[-1, 1]``.
                     - ``dr_e, dc_e`` : direction normalisée vers la zone d'évacuation la
                         plus proche, bornée dans ``[-1, 1]``.
                     - ``carrying`` : indicateur binaire (1.0 si le robot porte un survivant,
                         sinon 0.0).
                     - ``remaining_ratio`` : proportion de survivants encore à secourir,
                         calculée comme ``len(active) / NUM_SURVIVORS``.

                Returns
                -------
                np.ndarray
                        Vecteur 1D ``float32`` de taille 490 :
                        ``[local_obs(484), global_obs(6)]``.
        """
        P = self.PERCEPTION_R
        side = 2 * P + 1

        # 1. VISION LOCALE (CNN-style)
        # On crée 4 "calques" (channels) de 11x11, remplis de zéros.
        # Canal 0: Murs, Canal 1: Feu, Canal 2: Survivants, Canal 3: Zones d'évacuation
        channels = np.zeros((4, side, side), dtype=np.float32)
        
        # On scanne les alentours du robot (de -5 à +5 cases)
        for dr in range(-P, P + 1):
            for dc in range(-P, P + 1):
                nr, nc = self.robot_y + dr, self.robot_x + dc # Position réelle sur la carte
                ri, ci = dr + P, dc + P                       # Position dans la "caméra" (0 à 10)
                
                # Si le regard du robot sort de la carte, il voit ça comme un mur (1.0)
                if not self._in_bounds(nr, nc):
                    channels[0, ri, ci] = 1.0  
                    continue
                    
                cell = self.grid[nr, nc]
                if cell == Cell.WALL:
                    channels[0, ri, ci] = 1.0   # Dessine un mur sur le calque 0
                elif cell == Cell.DANGER:
                    channels[1, ri, ci] = 1.0   # Dessine du feu sur le calque 1
                elif cell == Cell.EVAC:
                    channels[3, ri, ci] = 1.0   # Dessine une zone verte sur le calque 3
                    
                s = self._survivor_at(nr, nc)
                if s is not None:
                    channels[2, ri, ci] = 1.0   # Dessine un survivant sur le calque 2

        # Aplatit les 4 grilles 11x11 en une seule longue liste de 484 chiffres (0 ou 1)
        local_obs = channels.flatten()   

        # 2. VISION GLOBALE (GPS / Boussole)
        # Filtre pour ne garder que les survivants non sauvés
        active = [s for s in self.survivors if not s["rescued"]]
        
        if active:
            # Trouve le survivant le plus proche en distance de Manhattan
            nearest = min(active, key=lambda s: abs(s["row"] - self.robot_y) + abs(s["col"] - self.robot_x))
            # Calcule la direction (vecteur) et la normalise entre -1 et 1 (pour que le réseau digère mieux)
            dr_s = np.clip((nearest["row"] - self.robot_y) / GRID_H, -1, 1)
            dc_s = np.clip((nearest["col"] - self.robot_x) / GRID_W, -1, 1)
        else:
            dr_s, dc_s = 0.0, 0.0 # Plus de survivants, le vecteur est nul

        # Même logique pour la zone d'évacuation la plus proche
        evac_cells = [(r, c) for r in range(GRID_H) for c in range(GRID_W) if self.grid[r, c] == Cell.EVAC]
        if evac_cells:
            nearest_evac = min(evac_cells, key=lambda p: abs(p[0] - self.robot_y) + abs(p[1] - self.robot_x))
            dr_e = np.clip((nearest_evac[0] - self.robot_y) / GRID_H, -1, 1)
            dc_e = np.clip((nearest_evac[1] - self.robot_x) / GRID_W, -1, 1)
        else:
            dr_e, dc_e = 0.0, 0.0

        # On compile ces 6 informations abstraites
        global_obs = np.array([
            dr_s, dc_s,                        # Vecteur vers survivant
            dr_e, dc_e,                        # Vecteur vers évacuation
            float(self.carrying),              # 1.0 si on porte quelqu'un, 0.0 sinon
            len(active) / self.NUM_SURVIVORS,  # Ratio de progression (ex: 0.8 s'il en reste 4/5)
        ], dtype=np.float32)

        # On colle les 484 pixels et les 6 variables GPS. Ça donne notre vecteur final de 490.
        return np.concatenate([local_obs, global_obs])   

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _in_bounds(self, r, c) -> bool:
        """Vérifie qu'une coordonnée ne sort pas de la matrice 20x20."""
        return 0 <= r < GRID_H and 0 <= c < GRID_W

    def _survivor_at(self, r, c):
        """Vérifie s'il y a un survivant NON SAUVÉ aux coordonnées données."""
        for s in self.survivors:
            if not s["rescued"] and s["row"] == r and s["col"] == c:
                return s
        return None
    
    def _get_target_distance(self):
        """Calcule la distance de Manhattan vers l'objectif actuel."""
        if self.carrying:
            # S'il porte quelqu'un, l'objectif est l'évacuation
            evac_cells = [(y, x) for y in range(GRID_H) for x in range(GRID_W) if self.grid[y, x] == Cell.EVAC]
            if evac_cells:
                return min(abs(p[0] - self.robot_y) + abs(p[1] - self.robot_x) for p in evac_cells)
            return 0
        else:
            # S'il a les mains vides, l'objectif est le survivant le plus proche
            active = [s for s in self.survivors if not s["rescued"]]
            if active:
                return min(abs(s["row"] - self.robot_y) + abs(s["col"] - self.robot_x) for s in active)
            return 0