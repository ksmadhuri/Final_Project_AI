import tensorflow as tf
import random
import numpy as np
from collections import deque
from snake_game_logic import DeepQNNet, DeepQTraining
import constants as CNST

class DeepQAgent:
    """Class representing the DeepQAgent for the Snake game."""

    def __init__(self):
        """Initialize the DeepQAgent."""
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
        # Define points in each direction
        left_point = CNST.Point(head.x - CNST.MOVEMENT_OFFSET, head.y)
        right_point = CNST.Point(head.x + CNST.MOVEMENT_OFFSET, head.y)
        up_point = CNST.Point(head.x, head.y - CNST.MOVEMENT_OFFSET)
        down_point = CNST.Point(head.x, head.y + CNST.MOVEMENT_OFFSET)

        # Check for collision in each direction
        danger_straight, danger_right, danger_left = self.get_dangers(game, left_point, right_point, up_point, down_point)

        # Encode move direction
        move_left = game.direction == CNST.snake_direction.LEFT
        move_right = game.direction == CNST.snake_direction.RIGHT
        move_up = game.direction == CNST.snake_direction.UP
        move_down = game.direction == CNST.snake_direction.DOWN

        # Encode food location
        food_left = game.food.x < game.head.x
        food_right = game.food.x > game.head.x
        food_up = game.food.y < game.head.y
        food_down = game.food.y > game.head.y

        # Construct state array
        state = [danger_straight, danger_right, danger_left, move_left, move_right, move_up, move_down, food_left, food_right, food_up, food_down]

        return np.array(state, dtype=int)

    def get_dangers(self, game, left_point, right_point, up_point, down_point):
        """Calculate the dangers in each direction.

        Args:
            game (SnakeGameInterface): The Snake game instance.
            left_point (Point): The point to the left of the snake's head.
            right_point (Point): The point to the right of the snake's head.
            up_point (Point): The point above the snake's head.
            down_point (Point): The point below the snake's head.

        Returns:
            Tuple[bool, bool, bool]: A tuple containing danger flags for straight, right, and left directions.
        """
        danger_straight = (game.direction == CNST.snake_direction.RIGHT and game.collision_chk(right_point)) or \
                        (game.direction == CNST.snake_direction.LEFT and game.collision_chk(left_point)) or \
                        (game.direction == CNST.snake_direction.UP and game.collision_chk(up_point)) or \
                        (game.direction == CNST.snake_direction.DOWN and game.collision_chk(down_point))

        danger_right = (game.direction == CNST.snake_direction.UP and game.collision_chk(right_point)) or \
                    (game.direction == CNST.snake_direction.DOWN and game.collision_chk(left_point)) or \
                    (game.direction == CNST.snake_direction.LEFT and game.collision_chk(up_point)) or \
                    (game.direction == CNST.snake_direction.RIGHT and game.collision_chk(down_point))

        danger_left = (game.direction == CNST.snake_direction.DOWN and game.collision_chk(right_point)) or \
                    (game.direction == CNST.snake_direction.UP and game.collision_chk(left_point)) or \
                    (game.direction == CNST.snake_direction.RIGHT and game.collision_chk(up_point)) or \
                    (game.direction == CNST.snake_direction.LEFT and game.collision_chk(down_point))
                        
        return danger_straight, danger_right, danger_left


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
