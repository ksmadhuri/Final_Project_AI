from agent import Agent
from game import SnakeGameInterface
import constants as CNST

def train():
    """_summary_
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameInterface()
    while agent.n_games < 150:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.game_play(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.game_rest()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save_model()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            CNST.plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()