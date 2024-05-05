import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import constants as CNST

pygame.init()
font = pygame.font.SysFont('arial', 25)

class SnakeGameInterface:
    """_summary_
    """

    def __init__(self, width=CNST.WIDTH, height=CNST.HEIGHT):
        """_summary_

        Args:
            width (_type_, optional): _description_. Defaults to CNST.WIDTH.
            height (_type_, optional): _description_. Defaults to CNST.HEIGHT.
        """
        self.width = width
        self.height = height
        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset_game()


    def reset_game(self):
        """_summary_
        """
        # init game state
        self.direction = CNST.snake_direction.RIGHT

        self.head = CNST.Point(self.width/2, self.height/2)
        self.snake = [CNST.Point(self.head.x, self.head.y),
                      CNST.Point(self.head.x, self.head.y)]
                      #CNST.Point(self.head.x, self.head.y)]
        #[CNST.Point(self.head.x-CNST.GRID_SIZE, self.head.y),
                      #CNST.Point(self.head.x-(2*CNST.GRID_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.create_food()
        self.iterations = 0


    def create_food(self):
        """_summary_
        """
        x = random.randint(0, (self.width-CNST.GRID_SIZE )//CNST.GRID_SIZE )*CNST.GRID_SIZE
        y = random.randint(0, (self.height-CNST.GRID_SIZE )//CNST.GRID_SIZE )*CNST.GRID_SIZE
        self.food = CNST.Point(x, y)
        if self.food in self.snake:
            self.create_food()


    def game_play(self, action):
        """_summary_

        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
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
        """_summary_

        Args:
            pt (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
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
        """_summary_
        """
        self.display.fill(CNST.BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, CNST.WHITE, pygame.Rect(pt.x, pt.y, CNST.GRID_SIZE, CNST.GRID_SIZE))
            #pygame.draw.rect(self.display, CNST.BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, CNST.RED, pygame.Rect(self.food.x, self.food.y, CNST.GRID_SIZE, CNST.GRID_SIZE))

        text = font.render("Score: " + str(self.score), True, CNST.WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def move_snake(self, action):
        """_summary_

        Args:
            action (_type_): _description_
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
        
# model code     
torch.manual_seed(596)

class DeepQNNet(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save_model(self, file_name=CNST.MODEL_FILE_NAME):
        model_folder_path = CNST.MODEL_FOLDER_PATH
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class DeepQTraining:
    """_summary_
    """
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()