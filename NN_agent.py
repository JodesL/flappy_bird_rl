import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from flappy_bird_engine import FlappyBirdGame


class NNAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8, 120)
        self.l1_2 = nn.Linear(120, 120)
        self.l2 = nn.Linear(120, 1)

    def forward(self, x):
        x = torch.div(x, torch.tensor(500))
        x = F.relu(self.l1(x))
        x = F.relu(self.l1_2(x))
        x = self.l2(x)
        x = torch.sigmoid(x)

        return x


class FlappyBot():
    def __init__(self, NN_agent=None, path=None, flappy_bird_game_params={}):
        self.NN_agent = NNAgent
        self.path = path
        self.game = FlappyBirdGame(**flappy_bird_game_params)
        self.flappy_bird_game_params = flappy_bird_game_params
        if self.path is not None:
            self.NN_agent = torch.load(self.path)
        assert self.NN_agent is not None, 'flappy_bot needs an agent to be initialized'

    def init_game(self, flappy_bird_game_params):
        self.game = FlappyBirdGame(**flappy_bird_game_params)

    def train(self, epochs, trials_per_epoch=1, learning_rate=1e-4, verbose=False):
        optimizer = optim.Adam(self.NN_agent.parameters(), lr=learning_rate)

        for i in range(epochs):
            results = self.game.run_n_trials(trials_per_epoch, self.NN_agent.forward)
            if i % 50 == 0 and verbose:
                print(results['rewards'].max())
                print(results['actual_decisions'].std())

            rewards = results['rewards']
            rewards = (rewards - rewards.mean()) / rewards.std()
            agent_decisions = results['agent_decisions']
            actual_decisions = results['actual_decisions']

            # likelihood of path
            ll = actual_decisions * agent_decisions + (1 - actual_decisions) * (1 - agent_decisions)
            loss = torch.mul(torch.sum(torch.mul(torch.log(ll), rewards)), -1)

            optimizer.zero_grad()
            loss.backward()

            if i % 50 == 0 and verbose:
                print(loss.item())
            if self.path is not None and i % 50 == 0:
                self.save(self.path)

            optimizer.step()

        if self.path is not None:
            self.save(self.path)

    def run_trial(self, n_trials):
        return self.game.run_n_trials(n_trials, self.NN_agent.forward, False)

    def save(self, path):
        torch.save(self.NN_agent, path)


if __name__ == '__main__':
    test_bot_100_path = '/home/jonah/PycharmProjects/flappy_bird_rl/trained_agents/test_bot_100_gap.pytorch'
    flp_dict = {
        'force_fps': True,
        'display_screen': False,
        'reward_values': {
            "positive": 1.0,
            "tick": 0.1,
            "loss": -5.0
        },
        'reward_discount': 0.99
    }

    test_bot = FlappyBot(path=test_bot_100_path,
                         flappy_bird_game_params=flp_dict)

    test_bot.train(epochs=1000000, verbose=True)


    # gap_250_test_bot = FlappyBot(
    #     path="/home/jonah/PycharmProjects/flappy_bird_rl/trained_agents/trained_bot_250_gap.pytorch",
    #     flappy_bird_game_params={'force_fps': False,
    #                              'display_screen': True,
    #                              'reward_values': {
    #                                  "positive": 1.0,
    #                                  "tick": 0.1,
    #                                  "loss": -5.0
    #                              },
    #                              'reward_discount': 0.99,
    #                              'pip_gap': 250})
    # gap_250_test_bot.run_trial(1)
