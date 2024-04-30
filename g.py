import pygame
import random
import time

# Initialize Pygame
pygame.init()

# Constants
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GRID_SIZE = 10
DISPLAY_WIDTH = 500
DISPLAY_HEIGHT = 500

# Snake and food positions
snake_head = [250, 250]
snake_position = [[250, 250], [240, 250], [230, 250]]
food_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]

# Initialize game display
display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
clock = pygame.time.Clock()

# Game settings
score = 0
prev_button_direction = 1  # Default to moving right
button_direction = 1

# Functions for game logic
def collision_with_boundaries(snake_head):
    if (snake_head[0] >= DISPLAY_WIDTH or snake_head[0] < 0 or
        snake_head[1] >= DISPLAY_HEIGHT or snake_head[1] < 0):
        return True
    else:
        return False

def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return True
    else:
        return False

def collision_with_food(snake_head, food_position):
    if snake_head == food_position:
        return True
    else:
        return False

#def collision_with_apple(apple_position, score):
    #apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    #score += 1
    #return apple_position, score

def display_final_score(display_text, final_score):
    largeText = pygame.font.Font('freesansbold.ttf', 35)
    TextSurf = largeText.render(display_text, True, BLACK)
    TextRect = TextSurf.get_rect()
    TextRect.center = ((DISPLAY_WIDTH / 2), (DISPLAY_HEIGHT / 2))
    display.blit(TextSurf, TextRect)
    pygame.display.update()
    time.sleep(2)

# Main game loop
running = True
while running:
    display.fill(BLACK)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT and prev_button_direction != 1:
                button_direction = 0
            elif event.key == pygame.K_RIGHT and prev_button_direction != 0:
                button_direction = 1
            elif event.key == pygame.K_UP and prev_button_direction != 2:
                button_direction = 3
            elif event.key == pygame.K_DOWN and prev_button_direction != 3:
                button_direction = 2

    # Change snake's head position based on button direction
    if button_direction == 1:
        snake_head[0] += GRID_SIZE
    elif button_direction == 0:
        snake_head[0] -= GRID_SIZE
    elif button_direction == 2:
        snake_head[1] += GRID_SIZE
    elif button_direction == 3:
        snake_head[1] -= GRID_SIZE

    # Update previous button direction
    prev_button_direction = button_direction

    # Move snake
    snake_position.insert(0, list(snake_head))
    if collision_with_food(snake_head, food_position):
        food_position, score = collision_with_apple(food_position, score)
    else:
        snake_position.pop()

    # Check for collisions
    if collision_with_boundaries(snake_head) or collision_with_self(snake_position):
        display_final_score("Game Over! Final Score: {}".format(score), score)
        running = False

    # Display snake
    for position in snake_position:
        pygame.draw.rect(display, WHITE, pygame.Rect(position[0], position[1], GRID_SIZE, GRID_SIZE))

    # Display food
    pygame.draw.rect(display, RED, pygame.Rect(food_position[0], food_position[1], GRID_SIZE, GRID_SIZE))

    pygame.display.flip()
    clock.tick(10)

pygame.quit()


