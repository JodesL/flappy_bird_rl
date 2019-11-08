from flappy_bird_rl.NN_agent import FlappyBot, NNAgent


game_params = {
        'force_fps': False,
        'display_screen': True
    }

untrained_bot = FlappyBot(NN_agent=NNAgent(),
                          flappy_bird_game_params=game_params)
untrained_bot.run_trial(10)

trained_bot = FlappyBot(path="trained_agents/test_bot_100_gap.pytorch",
                        flappy_bird_game_params=game_params)
trained_bot.run_trial()
