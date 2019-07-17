import numpy as np
from random import randint, random
from ple.games.flappybird import FlappyBird
from ple import PLE
import torch


class FlappyBirdGame():
    def __init__(self, reward_values={}, reward_discount=0.99, pip_gap=100, display_screen=True, fps=30, force_fps=True):
        self.game = PLE(FlappyBird(pipe_gap=pip_gap),
                        reward_values=reward_values,
                        fps=fps, force_fps=force_fps, display_screen=display_screen)
        self.game.init()
        self.actions = self.game.getActionSet()
        self.reward_discount = reward_discount

    @staticmethod
    def random_agent(*args, **kwargs):
        return torch.rand(1)

    def calculate_trial_reward(self, rewards_tensor):
        rewards_output = torch.empty(rewards_tensor.shape[0])
        for i in range(rewards_tensor.shape[0]):
            discount_vector = torch.Tensor([self.reward_discount] * (rewards_tensor.shape[0] - i))
            pv_rewards = sum(rewards_tensor[i:] * discount_vector ** torch.FloatTensor(range(rewards_tensor.shape[0] - i)))
            rewards_output[i] = pv_rewards

        rewards_output = rewards_output.reshape((-1, 1))
        return rewards_output

    @staticmethod
    def observation_to_torch_tensor(observation):
        obs_tensor = torch.FloatTensor(
            [observation['player_y'], observation['player_vel'], observation['next_pipe_dist_to_player'],
             observation['next_pipe_top_y'], observation['next_pipe_bottom_y'],
             observation['next_next_pipe_dist_to_player'], observation['next_next_pipe_top_y'],
             observation['next_next_pipe_bottom_y']])

        obs_tensor = obs_tensor.reshape((1, 8))
        return obs_tensor

    def run_trial(self, agent=None, verbose=False):
        if agent is None:
            agent = self.random_agent
        if self.game.game_over():
            self.game.reset_game()
        rewards = torch.empty(0)
        observations = torch.empty((0, 8))
        agent_decisions = torch.empty((0, 1))
        actual_decisions = torch.empty((0, 1))
        while not self.game.game_over():
            observation = self.observation_to_torch_tensor(self.game.getGameState())
            agent_decision = agent(observation)

            actual_decision = torch.bernoulli(agent_decision)
            actual_decision = actual_decision.reshape((1, 1))
            agent_decision = agent_decision.reshape((1, 1))
            if actual_decision == 1:
                action = self.actions[1]
            else:
                action = self.actions[0]

            reward = torch.FloatTensor([self.game.act(action)])

            # reward shaping
            if (observation[0][0] < observation[0][4]) and (observation[0][0] > observation[0][3]):
                reward = torch.add(reward, torch.tensor(0.2))
            else:
                reward = torch.add(reward, torch.tensor(-0.2))

            rewards = torch.cat((rewards, reward))
            observations = torch.cat((observations, observation))
            agent_decisions = torch.cat((agent_decisions, agent_decision))
            actual_decisions = torch.cat((actual_decisions, actual_decision))
            if verbose:
                print(f'action: {action}')
                print(f'observation: {observation}')
                print(f'reward: {reward}')

        return {'observations': observations,
                'rewards': self.calculate_trial_reward(rewards),
                'agent_decisions': agent_decisions,
                'actual_decisions': actual_decisions}

    def run_n_trials(self, n_trials, agent=None):
        out_results = {'observations': torch.empty(0), 'rewards': torch.empty(0),
                       'agent_decisions': torch.empty(0), 'actual_decisions': torch.empty(0)}
        for i in range(n_trials):
            results = self.run_trial(agent)
            out_results['observations'] = torch.cat((out_results['observations'], results['observations']))
            out_results['rewards'] = torch.cat((out_results['rewards'], results['rewards']))
            out_results['agent_decisions'] = torch.cat((out_results['agent_decisions'], results['agent_decisions']))
            out_results['actual_decisions'] = torch.cat((out_results['actual_decisions'], results['actual_decisions']))

        return out_results


if __name__ == '__main__':
    reward_values = {
        "positive": 1.0,
        "tick": 0.1,
        "loss": -5.0
    }
    test = FlappyBirdGame(force_fps=False, reward_values=reward_values, display_screen=True, reward_discount=0.99)
    test.run_n_trials(2)
