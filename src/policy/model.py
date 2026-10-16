"""Behaviour-cloning policy: an MLP with a movement head (regression) and a gripper head (classification).

Input normalisation statistics are stored inside the model (as buffers), so they are saved with the
weights and the exact same scaling is used at evaluation time.
"""
import torch
from torch import nn


class BCPolicy(nn.Module):
    def __init__(self, n_obs=11, hidden=256):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(n_obs))
        self.register_buffer("obs_std", torch.ones(n_obs))
        self.body = nn.Sequential(
            nn.Linear(n_obs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.move_head = nn.Linear(hidden, 3)     # x, y, z movement (OSC delta, -1..1)
        self.grip_head = nn.Linear(hidden, 1)     # logit of "gripper closed"

    def set_normalisation(self, mean, std):
        self.obs_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6))

    def forward(self, obs):
        h = self.body((obs - self.obs_mean) / self.obs_std)
        return torch.tanh(self.move_head(h)), self.grip_head(h).squeeze(-1)   # tanh keeps moves in -1..1

    @torch.no_grad()
    def act(self, obs_features):
        """One observation (numpy, 11 values) -> 7-number robosuite action."""
        move, grip_logit = self(torch.as_tensor(obs_features, dtype=torch.float32)[None])
        action = torch.zeros(7)
        action[:3] = move[0]
        action[6] = 1.0 if grip_logit.item() > 0 else -1.0         # logit > 0  <=>  P(closed) > 0.5
        return action.numpy()
