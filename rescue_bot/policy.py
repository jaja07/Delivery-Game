import torch
import torch.nn as nn
from torch.distributions import Categorical

import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    """
    Policy network using a CNN for spatial perception.
    Input  : observation vector of size 490
    Output : Categorical distribution over 6 actions
    """
    def __init__(self, obs_size: int = 490, n_actions: int = 6):
        super().__init__()
        
        # --- 1. Visual Extractor (CNN) ---
        # Takes the 4 input grids of size 11x11.
        self.cnn = nn.Sequential(
            # Layer 1: keeps 11x11 spatial size and outputs 32 feature maps.
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 2: reduces to 9x9 and increases to 64 feature maps.
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Layer 3: max pooling for strong downsampling (9x9 -> 4x4).
            nn.MaxPool2d(kernel_size=2), 
            nn.Flatten() # Flatten features before feeding the MLP.
        )
        
        # CNN output size: 64 filters * 4 height * 4 width.
        cnn_out_size = 64 * 4 * 4  # = 1024
        
        # --- 2. Global Feature Extractor ---
        # Small MLP for the 6 global features.
        self.global_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU()
        )
        
        # --- 3. Final Decision Head (MLP) ---
        # Concatenates vision (1024) + global features (32) = 1056 inputs.
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size + 32, 256),
            nn.ReLU(),
            nn.LayerNorm(256), # Helps stabilize learning.
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, obs, action_mask=None):
        """
        obs: Tensor of shape (batch, 490)
        """
        # 1. Split the observation into local and global parts.
        local_obs = obs[:, :484]  # 484 local values
        global_obs = obs[:, 484:] # 6 global features
        
        # 2. Reshape for CNN: (batch_size, channels, height, width).
        local_obs = local_obs.view(-1, 4, 11, 11)
        
        # 3. Extract features from both branches.
        cnn_features = self.cnn(local_obs)
        global_features = self.global_mlp(global_obs)
        
        # 4. Fuse both feature streams.
        combined = torch.cat([cnn_features, global_features], dim=1)
        
        # 5. Compute action logits.
        logits = self.fc(combined)
        
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))
            
        return Categorical(logits=logits)


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns G_t with discount factor gamma."""
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def reinforce_update(policy, optimizer, log_probs, rewards, entropies, gamma=0.99, entropy_coef=0.01):
    """Run a REINFORCE update with entropy bonus to encourage exploration."""
    returns = compute_returns(rewards, gamma)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize returns.

    # 1. Standard policy-gradient loss.
    policy_loss = -sum(lp * G for lp, G in zip(log_probs, returns))
    
    # 2. Entropy bonus.
    # We maximize entropy by minimizing its opposite in the loss.
    entropy_loss = -entropy_coef * sum(entropies)
    
    # 3. Total loss.
    loss = policy_loss + entropy_loss
    
    optimizer.zero_grad()
    loss.backward()  # type: ignore
    optimizer.step()
    
    return loss.item() # type: ignore