import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from flappy_bird_engine import FlappyBirdGame


class NNAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8, 20)
        self.l2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.div(x, torch.tensor(500))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.sigmoid(x)

        return x




if __name__ == '__main__':
    noob = NNAgent()
    noob = torch.load("/home/jonah/PycharmProjects/flappy_bird_rl/noob.pytorch")
    reward_values={
        "positive": 1.0,
        "tick": 0.1,
        "loss": -5.0
    }

    flp_game = FlappyBirdGame(force_fps=True, display_screen=False, reward_values=reward_values, reward_discount=0.99)
    number_rounds = 2000
    n_trials = 1
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(noob.parameters(), lr=learning_rate)

    for i in range(number_rounds):
        results = flp_game.run_epoch(n_trials, noob.forward)
        if i % 50 == 0:
            print(results['rewards'].max())
            print(results['actual_decisions'].std())

        rewards = results['rewards']
        rewards = (rewards - rewards.mean())/rewards.std()
        agent_decisions = results['agent_decisions']
        actual_decisions = results['actual_decisions']

        ll = actual_decisions * agent_decisions + (1 - actual_decisions) * (1 - agent_decisions)  #likelihood of path
        loss = torch.mul(torch.sum(torch.mul(torch.log(ll), rewards)), -1)

        optimizer.zero_grad()
        loss.backward()
        if i % 50 == 0:
            print(loss.item())
            torch.save(noob, "/home/jonah/PycharmProjects/flappy_bird_rl/noob.pytorch")
        optimizer.step()

    torch.save(noob, "/home/jonah/PycharmProjects/flappy_bird_rl/noob.pytorch")