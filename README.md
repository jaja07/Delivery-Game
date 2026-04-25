# RescueBot

RescueBot est un environnement de simulation RL (Pygame + PyTorch) dans lequel un robot doit retrouver des survivants puis les déposer dans des zones d'evacuation, en evitant murs et dangers.

## Apercu rapide

- Grille: 20x20
- Observation: 490 valeurs
- Actions: 6 (`UP`, `DOWN`, `LEFT`, `RIGHT`, `PICKUP`, `DROP`)
- Environnement principal: [rescue_bot/env.py](rescue_bot/env.py)
- Rendu Pygame: [rescue_bot/renderer.py](rescue_bot/renderer.py)
- Point d'entree: [rescue_bot/main.py](rescue_bot/main.py)

## Specifications MDP (Markov Decision Process)

### 1. Espace d'observation (taille: 490)

L'agent percoit son environnement via un vecteur combinant vision locale et informations globales:

- Vision locale (484 valeurs): une fenetre 11x11 centree sur le robot, encodee sur 4 canaux binaires (obstacles, danger, survivants, evacuation).
- Variables globales (6 valeurs):
	- Direction (delta r, delta c) vers le survivant actif le plus proche.
	- Direction (delta r, delta c) vers la zone d'evacuation la plus proche.
	- Indicateur binaire de portage (le robot transporte un survivant ou non).
	- Ratio de survivants restants.

### 2. Espace d'action (6 actions)

1. `UP`
2. `DOWN`
3. `LEFT`
4. `RIGHT`
5. `PICKUP` (valide uniquement sur une case avec survivant)
6. `DROP` (valide uniquement sur une case d'evacuation)

### 3. Systeme de recompenses

- Depot reussi: `+25.0`
- Ramassage reussi: `+10.0`
- Entree dans une zone de danger: `-5.0`
- Action invalide (mur, etc.): `-1.0`
- Penalite temporelle par pas: `-0.5`
- Bonus de fin d'episode: `+5.0` (si au moins un survivant evacue)

## Architecture du PolicyNet

Le modele de politique est un MLP (Perceptron multicouche) qui consomme le vecteur d'observation de taille 490.

- Couche d'entree: 490 neurones
- Tronc:
	- Dense(256) + ReLU + LayerNorm
	- Dense(128) + ReLU
	- Dense(64) + ReLU
- Tete de sortie: couche lineaire vers 6 actions

L'entrainement suit une approche REINFORCE (Policy Gradient), avec normalisation des retours pour stabiliser l'apprentissage.

## Prerequis

- Python 3.11+
- Git

## Installation et initialisation avec uv (recommande)

1. Cloner le depot puis entrer dans le dossier:

```powershell
git clone <url-du-repo>
cd Delivery-Game
```

2. Synchroniser le projet (creation du `.venv` + installation des dependances depuis `pyproject.toml` et `uv.lock`):

```powershell
uv sync
```

3. Activer l'environnement virtuel (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Lancer le jeu:

```powershell
python rescue_bot/main.py
```

Option notebook (dev):

```powershell
uv sync --group dev
```

Puis selectionner le noyau Python de `.venv` dans VS Code.

## Installation et initialisation avec pip

1. Cloner le depot puis entrer dans le dossier:

```powershell
git clone <url-du-repo>
cd Delivery-Game
```

2. Creer un environnement virtuel:

```powershell
python -m venv .venv
```

3. Activer l'environnement (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Installer le projet et ses dependances:

```powershell
python -m pip install --upgrade pip
pip install -e .
```

5. Lancer le jeu:

```powershell
python rescue_bot/main.py
```

Option notebook avec pip:

```powershell
pip install ipykernel jupyter
```

## Execution

Mode humain:

```powershell
python rescue_bot/main.py
```

Mode aleatoire:

```powershell
python rescue_bot/main.py --random
```

Avec l'entree script du projet (si installe):

```powershell
rescue-bot
```

## Notes uv

- `uv add <package>` met a jour automatiquement `pyproject.toml` et `uv.lock`.
- `uv.lock` doit etre versionne pour garantir des installations reproductibles.