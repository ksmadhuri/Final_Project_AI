import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
from enum import Enum
from collections import namedtuple

# Agent parameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001         # can change to 0.001
EPSILON = 1     # can change to 0
GAMMA = 0.9
OFFSET = 20

# model parameters
INPUT_SIZE = 11
HIDDEN_SIZE = 512 # can change 128, 256, 512
OUTPUT_SIZE = 3

MODEL_FILE_NAME = "q_learning_model.pth"
MODEL_FOLDER_PATH = "./models"

# colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLACK = (0,0,0)

# game parameters
GRID_SIZE = 20
SNAKE_SPEED = 60
WIDTH = 640
HEIGHT = 480

# snake parameters
class snake_direction(Enum):
    """Enum for snake directions."""
    RIGHT = 'RIGHT'
    LEFT = 'LEFT'
    UP = 'UP'
    DOWN = 'DOWN'

Point = namedtuple('Point', 'x, y')

# Plotting the progress of the model training
plt.ion()

def plot(scores):
    """Plot the training progress."""
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    sns.set_style("whitegrid") 
    
    plt.plot(scores, label='Scores', color='blue')
    
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    plt.text(len(scores)-1, round(scores[-1], 4), str(round(scores[-1], 4)))

    plt.legend()

    plt.ylim(ymin=0)
    plt.show(block=False)
    plt.pause(0.1)
