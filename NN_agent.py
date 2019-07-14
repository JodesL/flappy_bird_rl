import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from flappy_bird_engine import FlappyBirdGame


class NNAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_layer = nn.BatchNorm1d(8)
        self.l1 = nn.Linear(8, 15)
        self.l2 = nn.Linear(15, 1)

    def forward(self, x):
        x = torch.FloatTensor(x)
        # x = self.norm_layer(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.sigmoid(x)

        return x




if __name__ == '__main__':
    noob = NNAgent()
    flp_game = FlappyBirdGame()
    n_trials = 10

    results = flp_game.run_epoch(n_trials, noob.forward)

    rewards = torch.FloatTensor(results['rewards'])
    rewards = (rewards - rewards.mean())/rewards.std()
    agent_decisions = results['agent_decisions']
    loss = torch.log(agent_decisions).sum() * (rewards)