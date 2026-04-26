import torch
import torch.nn as nn
from torch.distributions import Categorical

import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    """
    Nouveau Réseau de politique avec CNN pour la vision spatiale.
    Entrée  : vecteur d'observation de taille 490
    Sortie  : distribution Categorical sur 6 actions
    """
    def __init__(self, obs_size: int = 490, n_actions: int = 6):
        super().__init__()
        
        # --- 1. L'Extracteur Visuel (CNN) ---
        # Il prend nos 4 grilles 11x11 en entrée
        self.cnn = nn.Sequential(
            # Couche 1 : Garde la taille 11x11 mais passe à 32 filtres de détection
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Couche 2 : Réduit la taille à 9x9 et augmente à 64 filtres
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Couche 3 : Max Pooling pour réduire fortement la dimension (9x9 -> 4x4)
            nn.MaxPool2d(kernel_size=2), 
            nn.Flatten() # Aplatit les features pour les connecter à la suite
        )
        
        # Calcul de la taille de sortie du CNN : 64 filtres * 4 de hauteur * 4 de largeur
        cnn_out_size = 64 * 4 * 4  # = 1024
        
        # --- 2. L'Extracteur Global (GPS) ---
        # Un mini-réseau pour analyser les 6 variables globales
        self.global_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU()
        )
        
        # --- 3. Le Décideur Final (MLP) ---
        # Prend la vision (1024) + le GPS (32) = 1056 entrées
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size + 32, 256),
            nn.ReLU(),
            nn.LayerNorm(256), # Aide à stabiliser l'apprentissage
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, obs, action_mask=None):
        """
        obs : Tensor de taille (batch, 490)
        """
        # 1. Découpage de l'observation
        local_obs = obs[:, :484]  # Les 484 pixels
        global_obs = obs[:, 484:] # Les 6 variables globales
        
        # 2. Reshape pour le CNN: (batch_size, channels, height, width)
        local_obs = local_obs.view(-1, 4, 11, 11)
        
        # 3. Extraction des features
        cnn_features = self.cnn(local_obs)
        global_features = self.global_mlp(global_obs)
        
        # 4. Fusion des deux "cerveaux"
        combined = torch.cat([cnn_features, global_features], dim=1)
        
        # 5. Choix de l'action
        logits = self.fc(combined)
        
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


def reinforce_update(policy, optimizer, log_probs, rewards, entropies, gamma=0.99, entropy_coef=0.01):
    """Mise à jour REINFORCE avec un bonus d'entropie pour encourager l'exploration."""
    returns = compute_returns(rewards, gamma)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalisation

    # 1. Loss classique du Policy Gradient
    policy_loss = -sum(lp * G for lp, G in zip(log_probs, returns))
    
    # 2. Bonus d'Entropie
    # On veut maximiser l'entropie, donc on minimise son opposé dans la Loss
    entropy_loss = -entropy_coef * sum(entropies)
    
    # 3. Loss totale
    loss = policy_loss + entropy_loss
    
    optimizer.zero_grad()
    loss.backward()  # type: ignore
    optimizer.step()
    
    return loss.item() # type: ignore