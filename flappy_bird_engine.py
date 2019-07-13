import numpy as np
from random import randint
from ple.games.flappybird import FlappyBird
from ple import PLE


class FlappyBirdGame():
    def __init__(self, display_screen=True, fps=30, reward_discount=0.99):
        self.game = PLE(FlappyBird(), fps=fps, display_screen=display_screen)
        self.game.init()
        self.actions = self.game.getActionSet()
        self.reward_discount = reward_discount

    @staticmethod
    def random_agent(observation=None):
        return randint(0, 1)

    def calculate_trial_reward(self, rewards_list):
        rewards_output = [0] * len(rewards_list)
        for i in range(len(rewards_output)):
            discount_vector = [self.reward_discount] * (len(rewards_list) - i)
            rewards_output[i] = sum(np.array(rewards_list[i:]) *
                                    np.array(discount_vector) ** np.array(range(len(rewards_list) - i)))
        return np.array(rewards_output)

    @staticmethod
    def observation_to_array(observation):
        return np.array([observation['player_y'], observation['player_vel'], observation['next_pipe_dist_to_player'],
                         observation['next_pipe_top_y'], observation['next_pipe_bottom_y'],
                         observation['next_next_pipe_dist_to_player'], observation['next_next_pipe_top_y'],
                         observation['next_next_pipe_bottom_y']])

    def reward_function(self, game_return):
        reward = 0
        if game_return == 0:
            reward = 1
        if game_return == -5:
            reward = -5

        return reward

    def run_trial(self, agent=FlappyBirdGame.random_agent):
        if self.game.game_over():
            self.game.reset_game()
        rewards = []
        observations = []
        while not self.game.game_over():
            observation = self.game.getGameState()
            action = self.actions[agent(observation)]
            reward = self.reward_function(self.game.act(action))

            rewards.append(reward)
            observations.append(self.observation_to_array(observation))

            # print(f'action: {action}')
            # print(f'observation: {observation}')
            # print(f'reward: {reward}')
        return {'observations': observations,
                'rewards': self.calculate_trial_reward(rewards)}

    def run_epoch(self, n_trials, agent=FlappyBirdGame.random_agent):
        out_results = {'observations': np.empty(0), 'rewards': np.empty(0)}
        for i in range(n_trials):
            results = self.run_trial(agent)
            out_results['observations']= np.append(out_results['observations'], results['observations'])
            out_results['rewards'] = np.append(out_results['rewards'], results['rewards'])
        return out_results



if __name__ == '__main__':
    test = FlappyBirdGame(display_screen=True)
    test.run_epoch(2)

# test code
# #
# display_screen = True
# fps = 30
# game = PLE(FlappyBird(), fps=fps, display_screen=display_screen)
# # agent = myAgentHere(allowed_actions=p.getActionSet())
# nb_frames=1000
# game.init()
# reward = 0.0
#
# for i in range(nb_frames):
#     if game.game_over():
#            game.reset_game()
#
#     # print(game.getGameState())
#     observation = game.getScreenRGB()
#     action = game.getActionSet()[0]
#     reward = game.act(action)
