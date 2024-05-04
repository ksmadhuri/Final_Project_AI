import pygame
import random
import time
from enum import Enum
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

# Constants
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GRID_SIZE = 10
SPEED = 10
#DISPLAY_WIDTH = 500
#DISPLAY_HEIGHT = 500

# Snake and food positions
#snake_head = [250, 250]
#snake_position = [[250, 250], [240, 250], [230, 250]]
#food_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]

# Initialize game display
#display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
#clock = pygame.time.Clock()

# Game settings
#score = 0
#prev_button_direction = 1  # Default to moving right
#button_direction = 1

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class SnakeGame:
    
    def __init__(self, width = 500, height = 500):
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        #self.reset()

        self.direction = Direction.RIGHT
        self.snake = [self.width/2, self.height/2]
        self.snake_position = [[self.width/2, self.height/2], [self.width/2, self.height/2], [self.width/2, self.height/2]]
        self.score = 0
        self.food_position = None
        self.generate_new_food_position()
        #self.frame_iteration = 0
        
    def generate_new_food_position(self):
        self.food_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        if self.food_position in self.snake_position:
            self.generate_new_food_position()
    
        
        # Functions for game logic
    def collision_with_boundaries(self):
        if (self.snake[0] >= self.width or self.snake[0] < 0 or
        self.snake[1] >= self.height or self.snake[1] < 0):
            return True
        else:
            return False

    def collision_with_self(self):
        self.snake = self.snake_position[0]
        if self.snake in self.snake_position[1:]:
            return True
        else:
            return False
        

    def play_step(self):
        #self.frame_iteration += 1
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
       
       # 2. move
        self.move(self.direction) # update the head
        self.snake_position.insert(0, self.snake)
        
        # 3. check if game over
        #reward = 0
        game_over = False
        if self.collision_with_boundaries() or self.collision_with_self():
            game_over = True
            #reward = -10
            return game_over, self.score
        
        #4. place new food
        if self.snake == self.food_position:
            self.score +=1
            #reward = 10
            self.generate_new_food_position()
        else:
            self.snake_position.pop()
            
        #5. update ui
        self.update_ui()
        self.clock.tick(SPEED)

        #6
        return game_over, self.score


    def update_ui(self):
        self.display.fill(BLACK)
        
        for position in self.snake_position:
            pygame.draw.rect(self.display, WHITE, pygame.Rect(position[0], position[1], GRID_SIZE, GRID_SIZE))

        # Display food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food_position[0], self.food_position[1], GRID_SIZE, GRID_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
    #clock.tick(10)
    
    
    def move(self,direction):
        
        # [straight, right, left]

        # clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # idx = clock_wise.index(self.direction)

        # if np.array_equal(action, [1, 0, 0]):
        #     new_dir = clock_wise[idx] # no change
        # elif np.array_equal(action, [0, 1, 0]):
        #     next_idx = (idx + 1) % 4
        #     new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        # else: # [0, 0, 1]
        #     next_idx = (idx - 1) % 4
        #     new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        # self.direction = new_dir
        
        x = self.snake[0]
        y = self.snake[1]
        if direction == Direction.RIGHT:
            x += GRID_SIZE
        elif direction == Direction.LEFT:
            x -= GRID_SIZE
        elif direction == Direction.DOWN:
            y += GRID_SIZE
        elif direction == Direction.UP:
            y -= GRID_SIZE
            
        self.snake = [x,y]

if __name__ == '__main__':
    game = SnakeGame()
    
    # game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()