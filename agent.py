import torch
import random
import numpy as np
from collections import deque
from game import DeepQNNet, DeepQTraining
import constants as CNST

class Agent:
    """_summary_
    """

    def __init__(self):
        """_summary_
        """
        self.n_games = 0
        self.epsilon = CNST.EPSILON
        self.gamma = CNST.GAMMA
        self.memory = deque(maxlen=CNST.MAX_MEMORY) # popleft()
        self.model = DeepQNNet(CNST.INPUT_SIZE, CNST.HIDDEN_SIZE, CNST.OUTPUT_SIZE)
        self.trainer = DeepQTraining(self.model, lr=CNST.LR, gamma=self.gamma)


    def get_state(self, game):
        """_summary_

        Args:
            game (_type_): _description_

        Returns:
            _type_: _description_
        """
        head = game.snake[0]
        point_l = CNST.Point(head.x - CNST.OFFSET, head.y)
        point_r = CNST.Point(head.x + CNST.OFFSET, head.y)
        point_u = CNST.Point(head.x, head.y - CNST.OFFSET)
        point_d = CNST.Point(head.x, head.y + CNST.OFFSET)
        
        dir_l = game.direction == CNST.Direction.LEFT
        dir_r = game.direction == CNST.Direction.RIGHT
        dir_u = game.direction == CNST.Direction.UP
        dir_d = game.direction == CNST.Direction.DOWN
        
        
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
        """_summary_

        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            next_state (_type_): _description_
            done (function): _description_
        """
        self.memory.append((state, action, reward, next_state, done)) # popleft if CNST.MAX_MEMORY is reached

    def train_long_memory(self):
        """_summary_
        """
        if len(self.memory) > CNST.BATCH_SIZE:
            mini_sample = random.sample(self.memory, CNST.BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        """_summary_

        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            next_state (_type_): _description_
            done (function): _description_
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """_summary_

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move