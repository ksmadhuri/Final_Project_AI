from snake_game_controller import Agent
from snake_game_logic import SnakeGameInterface
import constants as CNST

def train():
    """Train the Deep Q-Learning agent."""
    plot_scores = []  # List to store scores for plotting
    total_score = 0  # Total score counter
    record = 0  # Record score
    agent = Agent()  # Initialize the agent
    game = SnakeGameInterface()  # Initialize the game environment
    while agent.n_games < 200:
        # Get the old state
        state_old = agent.get_state(game)

        # Get the action to take
        final_move = agent.get_action(state_old)

        # Perform the move and get the new state
        reward, done, score = game.game_play(final_move)
        state_new = agent.get_state(game)

        # Train the agent using short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember the experience
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train the agent using long memory and plot the result
            game.reset_game()
            agent.n_games += 1
            agent.train_long_memory()

            # Update the record score and save the model if a new record is achieved
            if score > record:
                record = score
                agent.model.save_model()

            print('Number of Games', agent.n_games, 'Game Score', score, 'Record Score:', record)
            print("----------------------------")
            # Update the list of scores for plotting and update total score
            plot_scores.append(score)
            total_score += score
            CNST.plot(plot_scores)


if __name__ == '__main__':
    train()