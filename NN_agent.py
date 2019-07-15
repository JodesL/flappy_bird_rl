import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from flappy_bird_engine import FlappyBirdGame


class NNAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8, 15)
        self.l2 = nn.Linear(15, 1)

    def forward(self, x):
        x = torch.div(x, torch.tensor(500))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.sigmoid(x)

        return x




if __name__ == '__main__':
    noob = NNAgent()
    flp_game = FlappyBirdGame(display_screen=False)
    number_rounds = 2000
    n_trials = 5
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(noob.parameters(), lr=learning_rate)

    for i in range(number_rounds):
        results = flp_game.run_epoch(n_trials, noob.forward)
        if i % 25 == 0:
            print(results['rewards'].max())
            print(results['actual_decisions'].std())

        rewards = results['rewards']
        rewards = (rewards - rewards.mean())/rewards.std()
        agent_decisions = results['agent_decisions']
        actual_decisions = results['actual_decisions']

        ll = actual_decisions * agent_decisions + (1 - actual_decisions) * (1 - agent_decisions)
        loss = -(torch.log(ll) * rewards).sum()
        optimizer.zero_grad()
        loss.backward()
        if i % 25 == 0:
            print(loss.item())
        optimizer.step()
