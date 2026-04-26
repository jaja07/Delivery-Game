import numpy as np
import random
from .config import GRID_W, GRID_H, Cell, Action, ACTION_DELTAS

def build_map() -> np.ndarray:
    """Build a 20x20 map with obstacles, dangers and evacuation zones."""
    grid = np.zeros((GRID_H, GRID_W), dtype=np.int8)

    def fill(r, c, h, w, t):
        """Fill a rectangle in the grid with type t."""
        grid[r:r+h, c:c+w] = t

    # Borders
    grid[0, :]  = Cell.WALL
    grid[-1, :] = Cell.WALL
    grid[:, 0]  = Cell.WALL
    grid[:, -1] = Cell.WALL

    # Interior walls (destroyed buildings)
    walls = [
        (2, 3, 5, 2), (2, 8, 3, 2), (2, 14, 6, 2),
        (7, 3, 2, 5), (8, 10, 2, 3), (6, 16, 4, 2),
        (12, 2, 2, 4), (11, 8, 3, 2), (12, 13, 3, 3),
        (15, 5, 3, 2), (15, 10, 2, 4), (14, 16, 3, 2),
        (4, 5, 1, 3), (9, 14, 2, 2), (17, 3, 1, 4),
    ]
    for (r, c, h, w) in walls:
        fill(r, c, h, w, Cell.WALL)

    # Danger zones (fire, radiation)
    dangers = [
        (3, 6, 2, 2), (5, 12, 2, 3), (9, 4, 2, 2),
        (13, 7, 2, 2), (16, 13, 2, 3), (7, 17, 3, 1),
        (4, 16, 2, 2), (11, 5, 1, 3),
    ]
    for (r, c, h, w) in dangers:
        fill(r, c, h, w, Cell.DANGER)

    # Evacuation points
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
    MDP environment for RescueBot.

    Observation: 490-element NumPy vector
        - 484: 11x11 window encoded across 4 channels (obstacle, danger, survivor, evacuation)
        - 6: global variables (Δr_surv, Δc_surv, Δr_evac, Δc_evac, carrying, n_remaining)

    Actions: 6 (UP, DOWN, LEFT, RIGHT, PICKUP, DROP)

    Rewards:
        +10  pick up survivor
        +25  drop off at evacuation point
        -5  enter a danger zone
        -0.5 per time step
        -1  invalid action (walking into a wall)
        +5  end-of-episode bonus if at least 1 person evacuated
    """

    NUM_SURVIVORS  = 5
    MAX_STEPS      = 500
    PERCEPTION_R   = 5   # perception window radius (11x11)

    def __init__(self, seed: int = None): # type: ignore
        """
        Initializes the environment. If `seed` is provided, the map and positions will always be the same.
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
        Resets the environment. Places the robot and survivors, and resets the counters.
        Restores the initial state.
        """
        # The robot and survivors are placed randomly on empty cells.
        robot_initial_position = self.rng.choice(self.free_cells)
        self.robot_x, self.robot_y = robot_initial_position
        self.carrying = False

        survivors_initial_positions = self.rng.sample(
            [p for p in self.free_cells if p != robot_initial_position],
            self.NUM_SURVIVORS
        )
        # We create a list of dictionaries to track the status of each survivor
        self.survivors = [{"row": y, "col": x, "rescued": False} for x, y in survivors_initial_positions]

        self.steps    = 0
        self.rescued  = 0
        self.done     = False

        self.previous_distance = self._get_target_distance()
        
        return self._get_obs() 

    def step(self, action: int):
        """
        Executes an action by the agent and advances the environment by one step.

        The received action is converted to :class:`Action` and then resolved according to three
        behavior categories: movement, picking up, and placing.

        Parameters
        ----------
        action : int
            Action index expected by the reinforcement learning algorithm.
            Possible values correspond to:
            - 0 : ``Action.UP``
            - 1 : ``Action.DOWN``
            - 2 : ``Action.LEFT``
            - 3: ``Action.RIGHT``
            - 4: ``Action.PICKUP``
            - 5: ``Action.DROP``

        Returns
        -------
        tuple
            ``(obs, reward, done, info)`` where:
            - ``obs`` is the new observation of the environment,
            - ``reward`` is the instantaneous reward obtained,
            - ``done`` indicates whether the episode is over,
            - ``info`` may contain a descriptive event
                (for example, ``“wall_hit”``, ``‘danger’``, ``“pickup”``, or
                ``“drop_success”``).
        """
        assert not self.done
        action  = Action(action)
        reward  = -0.5
        info    = {}
        self.steps += 1

        # --- NAVIGATION (Up, Down, Left, Right) ---
        if action in ACTION_DELTAS:
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

        # --- COLLECTION MANAGEMENT ---
        elif action == Action.PICKUP:
            picked = self._survivor_at(self.robot_y, self.robot_x)
            if picked is not None and not self.carrying:
                picked["rescued"] = True
                self.carrying = True
                reward += 10.0
                info["event"] = "pickup"
                # Reset the distance tracker because the objective has changed.
                self.previous_distance = self._get_target_distance()
            else:
                reward -= 1.0

        # --- DROP MANAGEMENT ---
        elif action == Action.DROP:
            if self.carrying and self.grid[self.robot_y, self.robot_x] == Cell.EVAC:
                self.carrying = False
                self.rescued += 1
                reward += 25.0
                info["event"] = "drop_success"
                # Reset the distance tracker because the objective has changed.
                self.previous_distance = self._get_target_distance()
            else:
                reward -= 1.0
        
        # --- Reward shaping according to distance ---
        current_distance = self._get_target_distance()
        distance_diff = self.previous_distance - current_distance
        reward += distance_diff * 0.2 
        
        self.previous_distance = current_distance

        # --- End-of-episode check ---
        all_evacuated = self.rescued >= self.NUM_SURVIVORS

        if all_evacuated or self.steps >= self.MAX_STEPS:
            if self.rescued > 0:
                reward += 5.0
            self.done = True

        obs = self._get_obs()
        
        return obs, reward, self.done, info

    def get_action_mask(self) -> np.ndarray:
        """
        Returns a Boolean mask for the 6 actions (True = valid).
        Useful for preventing the neural network from calculating probabilities 
        for actions that are physically impossible.
        """
        mask = np.ones(6, dtype=bool)
        for a, (dr, dc) in ACTION_DELTAS.items():
            new_y, new_x = self.robot_y + dr, self.robot_x + dc
            if not self._in_bounds(new_y, new_x) or self.grid[new_y, new_x] == Cell.WALL:
                mask[a] = False # Block moves that would hit a wall
                
        surv = self._survivor_at(self.robot_y, self.robot_x)
        # The agent can only pick something up if there is a survivor and its hands are free.
        mask[Action.PICKUP] = (surv is not None) and (not self.carrying)
        
        # The agent can only drop if it is carrying someone and standing on an evacuation zone.
        mask[Action.DROP]   = (
            self.carrying and
            self.grid[self.robot_y, self.robot_x] == Cell.EVAC
        )
        return mask

    @property
    def obs_size(self):
        """Return the size of the flattened observation vector."""
        side = 2 * self.PERCEPTION_R + 1
        return side * side * 4 + 6

    @property
    def n_actions(self):
        """Return the number of discrete actions available to the agent."""
        return 6

    # ── Observation ────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Build the current observation vector.

        The observation concatenates a local 11x11 multi-channel view of the map
        and six global features describing the nearest survivor, the nearest
        evacuation zone, whether the robot is carrying a survivor, and the
        fraction of survivors still remaining.
        """
        P = self.PERCEPTION_R
        side = 2 * P + 1

        # 1. VISION LOCALE (CNN-style)
        # Create four 11x11 channels filled with zeros.
        # Channel 0: walls, Channel 1: danger, Channel 2: survivors, Channel 3: evacuation zones.
        channels = np.zeros((4, side, side), dtype=np.float32)
        
        # Scan the area around the robot (from -5 to +5 cells).
        for dr in range(-P, P + 1):
            for dc in range(-P, P + 1):
                nr, nc = self.robot_y + dr, self.robot_x + dc # Actual map position
                ri, ci = dr + P, dc + P                       # Position inside the camera view (0 to 10)
                
                # Anything outside the map is treated as a wall (1.0).
                if not self._in_bounds(nr, nc):
                    channels[0, ri, ci] = 1.0  
                    continue
                    
                cell = self.grid[nr, nc]
                if cell == Cell.WALL:
                    channels[0, ri, ci] = 1.0   # Draw a wall on channel 0.
                elif cell == Cell.DANGER:
                    channels[1, ri, ci] = 1.0   # Draw a danger cell on channel 1.
                elif cell == Cell.EVAC:
                    channels[3, ri, ci] = 1.0   # Draw an evacuation zone on channel 3.
                    
                s = self._survivor_at(nr, nc)
                if s is not None:
                    channels[2, ri, ci] = 1.0   # Draw a survivor on channel 2.

        # Flatten the four 11x11 grids into a single vector of 484 values.
        local_obs = channels.flatten()   

        # 2. GLOBAL VIEW (GPS / compass)
        # Keep only survivors that have not been rescued yet.
        active = [s for s in self.survivors if not s["rescued"]]
        
        if active:
            # Find the closest survivor using Manhattan distance.
            nearest = min(active, key=lambda s: abs(s["row"] - self.robot_y) + abs(s["col"] - self.robot_x))
            # Compute the direction vector and normalize it to [-1, 1].
            dr_s = np.clip((nearest["row"] - self.robot_y) / GRID_H, -1, 1)
            dc_s = np.clip((nearest["col"] - self.robot_x) / GRID_W, -1, 1)
        else:
            dr_s, dc_s = 0.0, 0.0 # No survivors left, so the vector is zero.

        # Use the same logic for the closest evacuation zone.
        evac_cells = [(r, c) for r in range(GRID_H) for c in range(GRID_W) if self.grid[r, c] == Cell.EVAC]
        if evac_cells:
            nearest_evac = min(evac_cells, key=lambda p: abs(p[0] - self.robot_y) + abs(p[1] - self.robot_x))
            dr_e = np.clip((nearest_evac[0] - self.robot_y) / GRID_H, -1, 1)
            dc_e = np.clip((nearest_evac[1] - self.robot_x) / GRID_W, -1, 1)
        else:
            dr_e, dc_e = 0.0, 0.0

        # Assemble the six global features.
        global_obs = np.array([
            dr_s, dc_s,                        # Vector toward the nearest survivor.
            dr_e, dc_e,                        # Vector toward the nearest evacuation zone.
            float(self.carrying),              # 1.0 if carrying a survivor, 0.0 otherwise.
            len(active) / self.NUM_SURVIVORS,  # Progress ratio (for example, 0.8 when 4/5 remain).
        ], dtype=np.float32)

        # Concatenate the 484 local values and the 6 global features.
        return np.concatenate([local_obs, global_obs])   

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _in_bounds(self, r, c) -> bool:
        """Return True when the given coordinate lies inside the 20x20 grid."""
        return 0 <= r < GRID_H and 0 <= c < GRID_W

    def _survivor_at(self, r, c):
        """Return the unrescued survivor located at the given coordinates, if any."""
        for s in self.survivors:
            if not s["rescued"] and s["row"] == r and s["col"] == c:
                return s
        return None
    
    def _get_target_distance(self):
        """Compute the Manhattan distance to the current objective."""
        if self.carrying:
            # When carrying a survivor, the objective is the evacuation zone.
            evac_cells = [(y, x) for y in range(GRID_H) for x in range(GRID_W) if self.grid[y, x] == Cell.EVAC]
            if evac_cells:
                return min(abs(p[0] - self.robot_y) + abs(p[1] - self.robot_x) for p in evac_cells)
            return 0
        else:
            # When empty-handed, the objective is the nearest survivor.
            active = [s for s in self.survivors if not s["rescued"]]
            if active:
                return min(abs(s["row"] - self.robot_y) + abs(s["col"] - self.robot_x) for s in active)
            return 0