# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 23:11:01 2026

@author: yasmi
"""

"""
Policy Network pour REINFORCE - Version Corrigée
Corrections :
  1. Architecture plus profonde (meilleure capacité)
  2. Normalisation des returns SANS double normalisation
  3. Entropy bonus pour encourager l'exploration
  4. learning_rate accessible comme attribut
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List


class PolicyNetwork(nn.Module):
    """
    Réseau de politique stochastique πθ(a|s).

    ✅ CORRECTION : Architecture plus profonde
    Input(7) → 256 → 256 → 128 → Output(5)
    """

    def __init__(self,
                 state_size: int = 7,
                 action_size: int = 5,
                 hidden_size: int = 256):
        """
        Initialise le réseau de politique.

        Args:
            state_size  : Taille de l'état (7)
            action_size : Nombre d'actions (5)
            hidden_size : Taille des couches cachées
        """
        super(PolicyNetwork, self).__init__()

        self.state_size  = state_size
        self.action_size = action_size

        # ✅ Architecture plus profonde avec Dropout
        self.network = nn.Sequential(
            # Couche 1
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            # Couche 2
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            # Couche 3
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),

            # Sortie
            nn.Linear(hidden_size // 2, action_size)
        )

        # ✅ Initialisation des poids (meilleure convergence)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialise les poids avec Xavier uniform.
        Évite les gradients qui explosent ou disparaissent.
        """
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass : calcule les logits des actions.

        Args:
            state : Tensor (batch_size, state_size)
        Returns:
            logits : Tensor (batch_size, action_size)
        """
        return self.network(state)

    def get_action_probabilities(self,
                                  state: np.ndarray) -> np.ndarray:
        """
        Calcule les probabilités softmax pour un état.

        Args:
            state : État numpy (state_size,)
        Returns:
            probs : Probabilités numpy (action_size,)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # ✅ Mode évaluation pour désactiver le Dropout
        self.eval()
        with torch.no_grad():
            logits = self.forward(state_tensor)
            probs  = torch.softmax(logits, dim=1)
        self.train()

        return probs.squeeze(0).numpy()

    def sample_action(self,
                      state: np.ndarray) -> Tuple[int, float]:
        """
        Échantillonne une action selon la politique.

        Args:
            state : État numpy (state_size,)
        Returns:
            (action, log_prob)
        """
        probs    = self.get_action_probabilities(state)
        action   = np.random.choice(len(probs), p=probs)
        log_prob = np.log(probs[action] + 1e-8)

        return action, log_prob

    def compute_log_probs(self,
                          states: torch.Tensor,
                          actions: torch.Tensor) -> torch.Tensor:
        """
        Calcule les log-probabilités des actions prises.

        Args:
            states  : Tensor (batch_size, state_size)
            actions : Tensor (batch_size,)
        Returns:
            log_probs : Tensor (batch_size,)
        """
        logits    = self.forward(states)
        probs     = torch.softmax(logits, dim=1)
        log_probs = torch.log(
            probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8
        )
        return log_probs

    def compute_entropy(self,
                        states: torch.Tensor) -> torch.Tensor:
        """
        Calcule l'entropie de la politique pour les états donnés.

        H(π) = -Σ π(a|s) * log π(a|s)

        Une entropie élevée = politique plus exploratoire.
        Utilisée comme bonus pour encourager l'exploration.

        Args:
            states : Tensor (batch_size, state_size)
        Returns:
            entropy : Scalaire (entropie moyenne)
        """
        logits  = self.forward(states)
        probs   = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        return entropy


class REINFORCEAgent:
    """
    Agent REINFORCE corrigé.

    ✅ CORRECTIONS :
    1. learning_rate accessible comme attribut
    2. Entropy bonus pour l'exploration
    3. Pas de double normalisation des returns
    4. Gradient clipping conservé
    """

    def __init__(self,
                 state_size: int = 7,
                 action_size: int = 5,
                 learning_rate: float = 0.0001,
                 discount_factor: float = 0.95,
                 entropy_coeff: float = 0.01):
        """
        Initialise l'agent REINFORCE.

        Args:
            state_size      : Taille de l'état
            action_size     : Nombre d'actions
            learning_rate   : Taux d'apprentissage
            discount_factor : Facteur d'actualisation γ
            entropy_coeff   : Coefficient du bonus d'entropie
        """
        # ✅ learning_rate accessible comme attribut
        self.learning_rate   = learning_rate
        self.discount_factor = discount_factor
        self.entropy_coeff   = entropy_coeff

        self.policy_network  = PolicyNetwork(state_size, action_size)
        self.optimizer       = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.policy_network.to(self.device)

    def select_action(self, state: np.ndarray) -> int:
        """
        Sélectionne une action selon la politique stochastique.

        ✅ Toujours stochastique pendant l'entraînement
        (exploration via échantillonnage)

        Args:
            state : État actuel (numpy array)
        Returns:
            action : Indice de l'action choisie (0-4)
        """
        action, _ = self.policy_network.sample_action(state)
        return action

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """
        Calcule les returns cumulés actualisés G_t.

        G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...

        ✅ CORRECTION : Normalisation UNE SEULE FOIS ici
        (supprime la double normalisation avec le Trainer)

        Args:
            rewards : Liste des récompenses de l'épisode
        Returns:
            returns_tensor : Tensor des returns normalisés
        """
        returns = []
        G = 0.0

        # ── Calcul en sens inverse (de la fin vers le début) ──────
        for reward in reversed(rewards):
            G = reward + self.discount_factor * G
            returns.insert(0, G)

        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # ✅ Normalisation UNIQUE (pas de double normalisation)
        if len(returns) > 1:
            mean = returns_tensor.mean()
            std  = returns_tensor.std()
            if std > 1e-8:
                returns_tensor = (returns_tensor - mean) / std

        return returns_tensor

    def train_on_episode(self,
                         states: List[np.ndarray],
                         actions: List[int],
                         rewards: List[float]) -> float:
        """
        Entraîne le réseau sur un épisode complet.

        Mise à jour REINFORCE avec entropy bonus :
        L = -Σ log π(a_t|s_t) * G_t - β * H(π)

        Où :
        - G_t = return actualisé
        - H(π) = entropie de la politique
        - β = entropy_coeff (encourage l'exploration)

        ✅ CORRECTIONS :
        1. Entropy bonus ajouté
        2. Gradient clipping conservé
        3. Pas de double normalisation

        Args:
            states  : Liste des états de l'épisode
            actions : Liste des actions prises
            rewards : Liste des récompenses reçues
        Returns:
            loss : Valeur de la perte pour cet épisode
        """
        # ── Conversion en tensors ─────────────────────────────────
        states_tensor  = torch.FloatTensor(
            np.array(states)
        ).to(self.device)

        actions_tensor = torch.LongTensor(actions).to(self.device)

        # ── Calcul des returns normalisés ─────────────────────────
        returns = self.compute_returns(rewards)

        # ── Log-probabilités des actions prises ───────────────────
        log_probs = self.policy_network.compute_log_probs(
            states_tensor, actions_tensor
        )

        # ── Loss REINFORCE ────────────────────────────────────────
        # L_policy = -E[log π(a|s) * G_t]
        # Le signe moins car on maximise J(θ) = E[G_t]
        policy_loss = -(log_probs * returns).mean()

        # ── Entropy Bonus ─────────────────────────────────────────
        # H(π) = -Σ π(a|s) * log π(a|s)
        # Soustrait de la loss pour MAXIMISER l'entropie
        # → Encourage l'exploration et évite la convergence prématurée
        entropy = self.policy_network.compute_entropy(states_tensor)
        entropy_loss = -self.entropy_coeff * entropy

        # ── Loss totale ───────────────────────────────────────────
        total_loss = policy_loss + entropy_loss

        # ── Mise à jour des paramètres ────────────────────────────
        self.optimizer.zero_grad()
        total_loss.backward()

        # ✅ Gradient clipping (évite les gradients explosifs)
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        return total_loss.item()

    def save_model(self, filepath: str):
        """
        Sauvegarde le modèle entraîné.

        ✅ Sauvegarde aussi les hyperparamètres
        pour pouvoir recharger correctement.

        Args:
            filepath : Chemin de sauvegarde (.pt)
        """
        torch.save({
            # Paramètres du réseau
            'policy_network_state_dict' : self.policy_network.state_dict(),
            'optimizer_state_dict'      : self.optimizer.state_dict(),

            # ✅ Hyperparamètres sauvegardés
            'learning_rate'             : self.learning_rate,
            'discount_factor'           : self.discount_factor,
            'entropy_coeff'             : self.entropy_coeff,
        }, filepath)

        print(f"✅ Modèle sauvegardé : {filepath}")

    def load_model(self, filepath: str):
        """
        Charge un modèle sauvegardé.

        ✅ Restaure aussi les hyperparamètres.

        Args:
            filepath : Chemin du modèle (.pt)
        """
        checkpoint = torch.load(
            filepath,
            map_location=self.device
        )

        # ── Restauration du réseau ────────────────────────────────
        self.policy_network.load_state_dict(
            checkpoint['policy_network_state_dict']
        )
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )

        # ── Restauration des hyperparamètres (si disponibles) ─────
        if 'learning_rate' in checkpoint:
            self.learning_rate   = checkpoint['learning_rate']
        if 'discount_factor' in checkpoint:
            self.discount_factor = checkpoint['discount_factor']
        if 'entropy_coeff' in checkpoint:
            self.entropy_coeff   = checkpoint['entropy_coeff']

        print(f"✅ Modèle chargé : {filepath}")