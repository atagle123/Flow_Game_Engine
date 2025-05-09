import os
from omegaconf import OmegaConf
from tqdm import tqdm
from src.utils.dataset import Maze_Dataset
import wandb
from typing import Dict
from src.models.flow import FlowLearner

filepath = "logs/data/data_maze.npz"


dataset = Maze_Dataset(filepath)


sample = dataset.sample_jax(1, keys=None)
print(sample)