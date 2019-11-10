from ple.games.flappybird import FlappyBird
from ple import PLE
import pandas as pd

class SFlappyBirdGame():
    def __init__(self, reward_values={}, pip_gap=100, display_screen=True, fps=30, force_fps=True):
        self.game = PLE(FlappyBird(pipe_gap=pip_gap),
                        reward_values=reward_values,
                        fps=fps, force_fps=force_fps, display_screen=display_screen)
        self.game.init()
        self.actions = self.game.getActionSet()

    def run_trial(self, agent, verbose=False):
        if self.game.game_over():
            self.game.reset_game()

        while not self.game.game_over():
            observation = pd.DataFrame(self.game.getGameState(), index=[0])
            observation.columns = ['player_y_position', 'player_velocity',
                                  'pipe_distance_to_player', 'pipe_top_y_postion',
                                  'pipe_bottom_y_postion', 'next_pipe_distance_to_player',
                                  'next_pipe_top_y_position', 'pipe_bottom_y_position']

            agent_decision = agent(observation)

            if verbose:
                print(f'observation: {observation}'
                      f'agent_decisions: {agent_decision}')

            actual_decision = 1 if agent_decision > 0.5 else 0

            if actual_decision == 1:
                action = self.actions[1]
            else:
                action = self.actions[0]

            self.game.act(action)

    def run_n_trials(self, n_trials, agent=None, sample=True):
        for i in range(n_trials):
            self.run_trial(agent, sample)
