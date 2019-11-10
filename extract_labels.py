from flappy_bird_rl.NN_agent import FlappyBot, NNAgent
from flappy_bird_rl.flappy_bird_engine import FlappyBirdGame
import pickle
import pandas as pd
import xgboost
from ple import PLE
from ple.games.flappybird import FlappyBird
from flappy_bird_rl.simply_flappy_bird_engine import SFlappyBirdGame

game_params = {
    'force_fps': True,
    'display_screen': False
}

trained_bot = FlappyBot(path="trained_agents/test_bot_100_gap.pytorch",
                        flappy_bird_game_params=game_params)
game = FlappyBirdGame(**game_params)
labeled_data = game.run_n_trials(100, trained_bot.NN_agent.forward, False)

with open('labeled_data.pickle', 'wb') as handle:
    pickle.dump(labeled_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

training_data = pd.DataFrame(labeled_data['observations'])
training_data = training_data.astype(float)
training_data.columns = ['player_y_position', 'player_velocity',
                         'pipe_distance_to_player', 'pipe_top_y_postion',
                         'pipe_bottom_y_postion', 'next_pipe_distance_to_player',
                         'next_pipe_top_y_position', 'pipe_bottom_y_position']

labels = pd.DataFrame(labeled_data['actual_decisions'])
labels = labels.astype('float')

training_data['agent_action'] = labels

training_data.to_csv('training_data.csv')

modelling_data = xgboost.DMatrix(data=training_data,
                                 label=labels)

params = {
    'objective':'reg:logistic',
    'eval_metric': 'logloss'
}

bst = xgboost.train(params, modelling_data, num_boost_round=100, evals=[(modelling_data, 'train')])

def supervised_agent(observation, model=bst):
    observation = xgboost.DMatrix(observation)

    return model.predict(observation)

game = SFlappyBirdGame(force_fps=False)
game.run_trial(supervised_agent, verbose=False)