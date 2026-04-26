# RescueBot

RescueBot is a grid-based reinforcement learning project built with Pygame and PyTorch.
An agent must navigate a disaster map, pick up survivors, and drop them at evacuation zones while avoiding danger areas and maximizing total reward.

## Current Project Scope

This repository currently contains two RL tracks:

- `rescue_bot/`: main project (20x20 map, 6-action policy, CNN+MLP policy network).
- `Yasmina/`: separate experimental implementation for a different environment setup.

This README documents the main `rescue_bot/` project.

## Repository Structure

```text
Delivery-Game/
├── .venv/
├── LICENSE
├── README.md
├── pyproject.toml
├── uv.lock
├── Version 2-2.zip
├── rescue_bot.egg-info/
├── rescue_bot/
│   ├── __init__.py
│   ├── config.py
│   ├── env.py
│   ├── main.py
│   ├── policy.py
│   ├── renderer.py
│   ├── rescue_bot_policy.pth
│   └── train.ipynb
└── Yasmina/
    ├── environment.py
    ├── evaluate_agent.py
    ├── play_game.py
    ├── policy_network.py
    ├── requirements.txt
    └── train_agent.py
```

## RescueBot MDP Specification

### State Space (Observation)

Each observation is a float32 vector of size 490:

- Local perception: 484 values
  - A centered 11x11 window (radius = 5) around the robot.
  - Encoded as 4 binary channels:
    - channel 0: wall
    - channel 1: danger
    - channel 2: unrescued survivor
    - channel 3: evacuation zone
  - Flattened from shape (4, 11, 11) to 484 values.

- Global features: 6 values
  - Direction to nearest active survivor: (dr_s, dc_s), normalized by grid size and clipped in [-1, 1].
  - Direction to nearest evacuation zone: (dr_e, dc_e), normalized by grid size and clipped in [-1, 1].
  - Carrying flag: 1.0 if carrying a survivor, else 0.0.
  - Remaining survivor ratio: active_survivors / NUM_SURVIVORS.

### Action Space

Discrete action space of size 6:

1. UP
2. DOWN
3. LEFT
4. RIGHT
5. PICKUP
6. DROP

### Transition Dynamics

The transition function is deterministic within an episode:

- Movement actions update the robot position using fixed deltas.
- Invalid movement (wall or out of bounds) keeps position and applies penalty.
- PICKUP succeeds only when the robot is on an unrescued survivor and not already carrying.
- DROP succeeds only when the robot is carrying and standing on an evacuation cell.

Stochasticity comes from episode initialization:

- robot start position sampled from empty cells
- survivor start positions sampled from empty cells
- optional reproducibility through `seed`

### Reward Function

Per step reward starts at -0.5 (time penalty), then event-based terms are added:

- +10.0 for successful PICKUP
- +25.0 for successful DROP at evacuation zone
- -5.0 when entering a danger cell
- -1.0 for invalid movement or invalid PICKUP/DROP
- +5.0 end-of-episode bonus if at least one survivor was evacuated

Distance shaping is also applied every step:

- `r_dist = 0.2 * (d_prev - d_curr)`
- `d_curr` is Manhattan distance to the current objective:
  - nearest active survivor if not carrying
  - nearest evacuation zone if carrying

When the objective changes (successful pickup/drop), the distance tracker is reset before shaping continues.

### Action Mask

The environment exposes a boolean action mask of size 6 (`get_action_mask`):

- Movement actions are masked out if they would hit a wall or leave the grid.
- PICKUP is valid only if an unrescued survivor is on the current cell and the robot is not carrying.
- DROP is valid only if the robot is carrying and standing on an evacuation zone.

This mask is consumed by the policy and invalid logits are set to `-inf`, forcing zero probability for impossible actions.

### Episode Termination

An episode ends when:

- all survivors are evacuated (`rescued >= NUM_SURVIVORS`), or
- max step budget is reached (`MAX_STEPS = 500`).

## Policy Architecture (rescue_bot/policy.py)

`PolicyNet` is a hybrid CNN + MLP policy:

- Input: 490
- Split:
  - local branch: 484 -> reshape to (4, 11, 11)
  - global branch: 6

- Local CNN branch:
  - Conv2d(4, 32, kernel=3, padding=1) + ReLU
  - Conv2d(32, 64, kernel=3, padding=0) + ReLU
  - MaxPool2d(2)
  - Flatten -> 1024

- Global MLP branch:
  - Linear(6, 32) + ReLU

- Fusion head:
  - concat(1024, 32) -> 1056
  - Linear(1056, 256) + ReLU + LayerNorm
  - Linear(256, 128) + ReLU
  - Linear(128, 6 logits)

Output is a Categorical distribution over 6 actions.

## Training Method

Training uses REINFORCE with entropy regularization:

- discounted returns computation (`gamma`, default 0.99)
- return normalization
- policy-gradient loss
- entropy bonus term (`entropy_coef`, default 0.01)

Reference implementation and experiments:

- notebook: `rescue_bot/train.ipynb`
- policy utilities: `rescue_bot/policy.py`

## Rendering and Interaction

The renderer (`rescue_bot/renderer.py`) provides:

- resizable window with internal fixed-size canvas scaling
- map, survivors, robot, danger animation, evacuation symbols
- local perception overlay (11x11)
- HUD with steps, evacuated, remaining, carrying, reward, last event
- clickable Restart button

Human controls:

- Arrow keys: move
- P: pickup
- D: drop
- R: reset
- Q: quit

## Run Modes

Application entry point: `rescue_bot/main.py`

At startup, a menu lets you choose:

- Manual mode (human plays)
- AI mode (load saved model and sample from policy)

CLI arguments:

- `--model`: path to model weights (`rescue_bot_policy.pth` by default)
- `--seed`: environment seed (`42` by default)

## Installation

### Option 1: uv (recommended)

```powershell
git clone <repo-url>
cd Delivery-Game
uv sync
.\.venv\Scripts\Activate.ps1
python rescue_bot/main.py
```

For notebook dependencies:

```powershell
uv sync --group dev
```

### Option 2: pip

```powershell
git clone <repo-url>
cd Delivery-Game
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python rescue_bot/main.py
```

Optional notebook packages:

```powershell
pip install ipykernel jupyter
```

## Console Script

After editable install, you can run:

```powershell
rescue-bot
```

## Dependencies

Defined in `pyproject.toml`:

- numpy
- pygame
- torch
- matplotlib
- pandas
- ipython

Dev group includes:

- jupyter
- ipykernel

## Notes

- `rescue_bot/rescue_bot_policy.pth` is a model artifact currently present in the repository.
- `Yasmina/` contains a separate RL workflow and is not used by `rescue_bot/main.py`.
- `uv.lock` should remain versioned for reproducible environments.
