from NN_agent import FlappyBot, NNAgent


game_params = {
        'force_fps': False,
        'display_screen': True
    }

example_bot = FlappyBot(path="trained_agents/test_bot_100_gap.pytorch",
                        flappy_bird_game_params=game_params)

example_bot.run_trial()