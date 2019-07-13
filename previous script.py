from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

### Initialize flappy bird game
game = FlappyBird(pipe_gap=100)
p = PLE(game, fps=30, force_fps=True, frame_skip=2, display_screen=True)
actions = p.getActionSet()[::-1]
p.init()


### Define neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #        self.conv1 = nn.Conv2d(1, 5, 5)
        #        self.conv2 = nn.Conv2d(5, 5, 5)
        #        self.pool1 = nn.MaxPool2d(5, 5)
        #        self.pool2 = nn.MaxPool2d(2, 2)
        self.l1 = nn.Linear(8, 120)
        self.l1_2 = nn.Linear(120, 120)
        self.l2 = nn.Linear(120, 1)
        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.xavier_uniform_(self.l2.weight)
        torch.nn.init.xavier_uniform_(self.l1_2.weight)
        self.l1.bias.data.fill_(0)
        self.l2.bias.data.fill_(0)
        self.l1_2.bias.data.fill_(0)

    def forward(self, x):
        #        x = self.pool1(x)
        #        x = self.conv1(x)
        #        x = F.relu(x)
        #        x = self.pool2(x)
        #        x = self.conv2(x)
        #        x = F.relu(x)
        #        x = x.view(-1, 4950)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l1_2(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.sigmoid(x)
        return (x)


policy = Net()

# policy = torch.load("Policy_attempt//attempt3")
optimizer = optim.RMSprop(policy.parameters(), lr=0.005)

### Game environment
epochs = 3000
max_obs = torch.FloatTensor(np.empty((1, 8)))
min_obs = torch.FloatTensor(np.empty((1, 8)))

for s in range(epochs):

    p.reset_game()
    # initialize frame capture size
    observation = torch.FloatTensor(list(game.getGameState().values()))
    observation = observation.view(-1, 8)

    # whole epoch of frames of multiple trials
    experience = torch.FloatTensor()
    experience_returns = torch.FloatTensor()
    experience_log_prob = torch.FloatTensor()

    # all returns associated with each trial
    returns = []
    reward_history = torch.FloatTensor()
    log_prob = torch.FloatTensor()
    memory = torch.FloatTensor()

    trial_number = 0
    iteration = 0

    while trial_number < 10:

        if p.game_over():

            # save experience and returns at trial end
            for r in range(1, (len(reward_history) + 1)):
                if r == 1:
                    returns.append(reward_history[-r])
                else:
                    returns.append(returns[-1] * 0.99 + reward_history[-r])

            returns = returns[::-1]
            returns = torch.FloatTensor(returns)

            if trial_number == 0:
                experience_returns = returns
                experience = memory
                experience_log_prob = log_prob
            else:
                experience_returns = torch.cat((experience_returns, returns))
                experience = torch.cat((experience, memory))
                experience_log_prob = torch.cat((experience_log_prob, log_prob))

            returns = []
            reward_history = torch.FloatTensor()
            memory = torch.FloatTensor()
            log_prob = torch.FloatTensor()
            iteration = 0

            #            time.sleep(0.25)
            trial_number += 1
            p.reset_game()

        observation = torch.FloatTensor(list(game.getGameState().values()))
        observation = observation.view(-1, 8)

        max_obs = torch.max(observation, max_obs)
        min_obs = torch.min(observation, min_obs)

        observation = observation / (max_obs - min_obs)

        prob_up = policy.forward(observation)

        action_int = int(prob_up.bernoulli().tolist()[0][0])
        action = actions[action_int]

        if action_int == 0:
            if iteration == 0:
                log_prob = torch.log(1.000 - prob_up)
            else:
                log_prob = torch.cat((log_prob, torch.log(1.000 - prob_up)))
        else:
            if iteration == 0:
                log_prob = torch.log(prob_up)
            else:
                log_prob = torch.cat((log_prob, torch.log(prob_up)))

        reward = p.act(action)
        reward = torch.FloatTensor([reward])
        if iteration == 0:
            memory = observation
            reward_history = reward
        else:
            memory = torch.cat((memory, observation))
            reward_history = torch.cat((reward_history, reward))

        iteration += 1

    average_returns = experience_returns.mean()
    max_returns = experience_returns.max()
    experience_returns = (experience_returns - experience_returns.mean()) / (experience_returns.std() + 0.0001)

    experience_prob_up = torch.min(torch.FloatTensor(np.ones(1)), torch.exp(experience_log_prob))

    entropy = -torch.mean(
        torch.mul(experience_log_prob, experience_prob_up) + torch.mul(torch.log(1.0001 - experience_prob_up),
                                                                       1 - experience_prob_up))
    loss = torch.sum(torch.mul(experience_returns, experience_log_prob).mul(-1))

    if s == 0:
        avg_loss = loss
    else:
        avg_loss = avg_loss * 0.95 + 0.05 * loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if s % 30 == 0:
        torch.save(policy, "policy_try_{}".format(s))

    print("Epoch:", s)
    print("Average returns:", average_returns)
    print("Max return:", max_returns)
    print("Current Loss:", loss)
    print("Average loss:", avg_loss)
    print("Entropy:", entropy)
    print("_________________________________________")

