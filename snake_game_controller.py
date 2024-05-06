import tensorflow as tf
import random
import numpy as np
from collections import deque
from snake_game_logic import DeepQNNet, DeepQTraining
import constants as CNST

class Agent:
    """Class representing the Agent for the Snake game."""

    def __init__(self):
        """Initialize the Agent."""
        self.n_games = 0
        self.epsilon = CNST.EPSILON
        self.discount_factor = CNST.DISCOUNT_FACTOR
        self.memory = deque(maxlen=CNST.MAXIMUM_MEMORY)
        self.model = DeepQNNet(CNST.INPUT_DIM, CNST.HIDDEN_DIM, CNST.OUTPUT_DIM)
        self.trainer = DeepQTraining(self.model, learning_rate=CNST.LEARNING_RATE, discount_factor=self.discount_factor)


    def get_state(self, game):
        """Get the current state of the game.

        Args:
            game (SnakeGameInterface): The Snake game instance.

        Returns:
            np.array: Array representing the current state.
        """
        head = game.snake[0]
        point_l = CNST.Point(head.x - CNST.MOVEMENT_OFFSET, head.y)
        point_r = CNST.Point(head.x + CNST.MOVEMENT_OFFSET, head.y)
        point_u = CNST.Point(head.x, head.y - CNST.MOVEMENT_OFFSET)
        point_d = CNST.Point(head.x, head.y + CNST.MOVEMENT_OFFSET)
        
        dir_l = game.direction == CNST.snake_direction.LEFT
        dir_r = game.direction == CNST.snake_direction.RIGHT
        dir_u = game.direction == CNST.snake_direction.UP
        dir_d = game.direction == CNST.snake_direction.DOWN
        
        
        state = [
            # Danger straight
            (dir_r and game.collision_chk(point_r)) or 
            (dir_l and game.collision_chk(point_l)) or 
            (dir_u and game.collision_chk(point_u)) or 
            (dir_d and game.collision_chk(point_d)),

            # Danger right
            (dir_u and game.collision_chk(point_r)) or 
            (dir_d and game.collision_chk(point_l)) or 
            (dir_l and game.collision_chk(point_u)) or 
            (dir_r and game.collision_chk(point_d)),

            # Danger left
            (dir_d and game.collision_chk(point_r)) or 
            (dir_u and game.collision_chk(point_l)) or 
            (dir_r and game.collision_chk(point_u)) or 
            (dir_l and game.collision_chk(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Store a game transition in the memory buffer.

        Args:
            state (np.array): Current state.
            action (int): Action taken.
            reward (int): Reward received.
            next_state (np.array): Next state.
            done (bool): Whether the episode is done.
        """
        self.memory.append((state, action, reward, next_state, done)) # popleft if CNST.MAXIMUM_MEMORY is reached

    def train_long_memory(self):
        """Train the model using experiences from memory."""
        if len(self.memory) > CNST.BATCH_SIZE:
            mini_sample = random.sample(self.memory, CNST.BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train the model using a single experience.

        Args:
            state (np.array): Current state.
            action (int): Action taken.
            reward (int): Reward received.
            next_state (np.array): Next state.
            done (bool): Whether the episode is done.
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Choose an action based on the current state.

        Args:
            state (np.array): Current state.

        Returns:
            np.array: Action to take.
        """
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
             # Exploitation: choose the best move based on the model's prediction
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)  # Convert state to tensor and add batch dimension
            prediction = self.model(state_tensor)  # Get model predictions
            move = tf.argmax(prediction[0]).numpy()  # Get the index of the max value
            final_move[move] = 1

        return final_move
