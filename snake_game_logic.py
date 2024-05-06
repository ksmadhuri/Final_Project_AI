import pygame
import random
import numpy as np
import tensorflow as tf
import os
import constants as CNST

pygame.init()
font = pygame.font.SysFont('arial', 25)

class SnakeGameInterface:
    """Class representing the interface for the Snake game."""

    def __init__(self, width=CNST.SCREEN_WIDTH, height=CNST.SCREEN_HEIGHT):
        """Initialize the Snake game interface.

        Args:
            width (int, optional): Width of the game window. Defaults to CNST.SCREEN_WIDTH.
            height (int, optional): Height of the game window. Defaults to CNST.SCREEN_HEIGHT.
        """
        self.width = width
        self.height = height
        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset_game()


    def reset_game(self):
        """Reset the game to its initial state."""
        # init game current_state
        self.direction = CNST.snake_direction.RIGHT

        self.head = CNST.Point(self.width/2, self.height/2)
        self.snake = [CNST.Point(self.head.x, self.head.y),
                      CNST.Point(self.head.x, self.head.y)]

        self.score = 0
        self.food = None
        self.create_food()
        self.iterations = 0


    def create_food(self):
        """Create food for the snake."""
        x = random.randint(0, (self.width-CNST.GRID_SIZE )//CNST.GRID_SIZE )*CNST.GRID_SIZE
        y = random.randint(0, (self.height-CNST.GRID_SIZE )//CNST.GRID_SIZE )*CNST.GRID_SIZE
        self.food = CNST.Point(x, y)
        if self.food in self.snake:
            self.create_food()


    def game_play(self, action):
        """Play one step of the game.

        Args:
            action (int): Action to take.

        Returns:
            Tuple[int, bool, int]: Reward, game over status, and current score.
        """
        self.iterations += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self.move_snake(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.collision_chk() or self.iterations > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.create_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self.interface_update()
        self.clock.tick(CNST.SNAKE_SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


    def collision_chk(self, pt=None):
        """Check for collisions.

        Args:
            pt (Point, optional): Point to check for collision. Defaults to None.

        Returns:
            bool: True if collision occurred, False otherwise.
        """
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.width - CNST.GRID_SIZE or pt.x < 0 or pt.y > self.height - CNST.GRID_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def interface_update(self):
        """Update the game interface."""
        self.display.fill(CNST.BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, CNST.WHITE, pygame.Rect(pt.x, pt.y, CNST.GRID_SIZE, CNST.GRID_SIZE))
            #pygame.draw.rect(self.display, CNST.BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, CNST.RED, pygame.Rect(self.food.x, self.food.y, CNST.GRID_SIZE, CNST.GRID_SIZE))

        text = font.render("Score: " + str(self.score), True, CNST.WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def move_snake(self, action):
        """Move the snake based on the action.

        Args:
            action (int): Action to take.
        """
        # [straight, right, left]

        clock_wise = [CNST.snake_direction.RIGHT, CNST.snake_direction.DOWN, CNST.snake_direction.LEFT, CNST.snake_direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == CNST.snake_direction.RIGHT:
            x += CNST.GRID_SIZE
        elif self.direction == CNST.snake_direction.LEFT:
            x -= CNST.GRID_SIZE
        elif self.direction == CNST.snake_direction.DOWN:
            y += CNST.GRID_SIZE
        elif self.direction == CNST.snake_direction.UP:
            y -= CNST.GRID_SIZE

        self.head = CNST.Point(x, y)
        
#model
class DeepQNNet(tf.keras.Model):
    """Deep Q-Network model for the Snake game."""
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the Deep Q-Network model.

        Args:
            input_size (int): Size of the input layer.
            hidden_size (int): Size of the hidden layer.
            output_size (int): Size of the output layer.
        """
        super(DeepQNNet, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=(input_size,))
        self.dense2 = tf.keras.layers.Dense(output_size, activation='relu')

    def call(self, inputs):
        """Forward pass of the model.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            tensor: Output tensor.
        """
        if len(inputs.shape) == 1:  # Means we have only one instance without batch dimension
            inputs = tf.expand_dims(inputs, 0)
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

    def save_model(self, file_name=CNST.MODEL_FILE_NAME):
        """Save the model to a file.

        Args:
            file_name (str, optional): Name of the file to save the model. Defaults to CNST.MODEL_FILE_NAME.
        """
        model_folder_path = CNST.MODEL_FOLDER_PATH
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_path = os.path.join(model_folder_path, file_name)
        self.save_weights(file_path)


class DeepQTraining:
    """Class for training the Deep Q-Network."""
    def __init__(self, model, learning_rate, discount_factor):
        """Initialize the Deep Q-Training.

        Args:
            model (DeepQNNet): Deep Q-Network model.
            learning_rate (float): Learning rate for optimization.
            discount_factor (float): Discount factor for future rewards.
        """
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.discount_factor = discount_factor
        self.loss = tf.keras.losses.MeanSquaredError()

    def train_step(self, current_state, action, reward, next_state, done):
        """Perform one training step.

        Args:
            current_state (tensor): Current state tensor.
            action (tensor): Action tensor.
            reward (tensor): Reward tensor.
            next_state (tensor): Next state tensor.
            done (bool): Whether the episode is done.
        """
        current_state = tf.convert_to_tensor(current_state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        
        if current_state.ndim == 1:
            current_state = tf.expand_dims(current_state, axis=0)
            next_state = tf.expand_dims(next_state, axis=0)
            action = tf.expand_dims(action, axis=0)
            reward = tf.expand_dims(reward, axis=0)
            done = (done, )
            
        with tf.GradientTape() as tape:
            loss = 0
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    max_q_value_next_state = tf.reduce_max(self.model(next_state[idx]), axis=1)
                    Q_new = reward[idx] + self.discount_factor * max_q_value_next_state

                # Compute the Q-value for the chosen action
                num_actions = self.model.dense2.units  # Get the number of units in the output layer
                action_mask = tf.one_hot(action[idx], depth=num_actions)
                Q_chosen_action = tf.reduce_sum(self.model(current_state[idx]) * action_mask, axis=1)

                # Compute the loss for this sample
                sample_loss = self.loss(Q_new, Q_chosen_action)
                loss += sample_loss

        # Get the gradients of the loss with respect to the trainable variables
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradients to update the model parameters
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
