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
    loss.backward() # type: ignore
    optimizer.step()
    return loss.item() # type: ignore