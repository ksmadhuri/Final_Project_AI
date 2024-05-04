import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, Direction, Point
from model import QNetModel, QLearningTrainer
from helper import plot

MAX_HISTORY = 100_000
SAMPLE_SIZE = 1000
LEARNING_RATE = 0.01

class GameAgent:

    def __init__(self):
        self.game_count = 0
        self.randomness = 0  # Initial randomness
        self.discount_factor = 0.9  # Discount rate for future rewards
        self.memory = deque(maxlen=MAX_HISTORY)  # automatic memory management
        self.q_net = QNetModel(11, 256, 3)
        self.q_trainer = QLearningTrainer(self.q_net, learning_rate=LEARNING_RATE, discount_factor=self.discount_factor)

    def extract_state(self, game):
        head = game.snake_position[0]
        left_point = [head[0] - 20, head[1]]
        right_point = [head[0] + 20, head[1]]
        up_point = [head[0], head[1] - 20]
        down_point = [head[0], head[1] + 20]

        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN
        
        # Function to check if the next point is dangerous
        def is_dangerous(point):
            return (game.collision_with_boundaries(point) or 
                game.collision_with_self(point))

        state_vector = [
            # Immediate dangers
            (dir_right and is_dangerous(right_point)) or 
            (dir_left and is_dangerous(left_point)) or 
            (dir_up and is_dangerous(up_point)) or 
            (dir_down and is_dangerous(down_point)),

            # Right-side dangers
            (dir_up and is_dangerous(right_point)) or 
            (dir_down and is_dangerous(left_point)) or 
            (dir_left and is_dangerous(up_point)) or 
            (dir_right and is_dangerous(down_point)),

            # Left-side dangers
            (dir_down and is_dangerous(right_point)) or 
            (dir_up and is_dangerous(left_point)) or 
            (dir_right and is_dangerous(up_point)) or 
            (dir_left and is_dangerous(down_point)),
            
            # Current direction
            dir_left, dir_right, dir_up, dir_down,
            
            # Food location relative to head
            game.food_position[0] < head[0],  # Food on the left
            game.food_position[0] > head[0],  # Food on the right
            game.food_position[1] < head[1],  # Food above
            game.food_position[1] > head[1]   # Food below
            ]

        return np.array(state_vector, dtype=int)

    def store_memory(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def replay_long_memory(self):
        if len(self.memory) > SAMPLE_SIZE:
            mini_batch = random.sample(self.memory, SAMPLE_SIZE)
        else:
            mini_batch = self.memory

        states, actions, rewards, next_states, terminals = zip(*mini_batch)
        self.q_trainer.update(states, actions, rewards, next_states, terminals)

    def replay_recent_memory(self, state, action, reward, next_state, terminal):
        self.q_trainer.update(state, action, reward, next_state, terminal)

    def decide_action(self, state):
        # Exploration-exploitation balance
        self.randomness = 80 - self.game_count
        action_vector = [0, 0, 0]
        if random.randint(0, 200) < self.randomness:
            move_index = random.randint(0, 2)
            action_vector[move_index] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.q_net(state_tensor)
            best_action = torch.argmax(prediction).item()
            action_vector[best_action] = 1

        return action_vector


def run_training_loop():
    score_history = []
    mean_score_history = []
    total_score = 0
    best_score = 0
    agent = GameAgent()
    game = SnakeGame()
    while True:
        current_state = agent.extract_state(game)
        action = agent.decide_action(current_state)
        reward, finished, score = game.play_step(action)
        new_state = agent.extract_state(game)

        agent.replay_recent_memory(current_state, action, reward, new_state, finished)
        agent.store_memory(current_state, action, reward, new_state, finished)

        if finished:
            game.reset()
            agent.game_count += 1
            agent.replay_long_memory()

            if score > best_score:
                best_score = score
                agent.q_net.save_model()

            print('Game', agent.game_count, 'Score', score, 'Record:', best_score)

            score_history.append(score)
            total_score += score
            mean_score = total_score / agent.game_count
            mean_score_history.append(mean_score)
            plot(score_history, mean_score_history)


if __name__ == '__main__':
    run_training_loop()
