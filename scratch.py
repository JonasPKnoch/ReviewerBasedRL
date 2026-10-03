import torch
from torch.utils.data import DataLoader
from torch import nn
from tm_task_generators import contains_one_task_generator
from tm_interface import TMAgent, TMInterface, TMClass, sample_logits
from reviewer_rl import *
from reviewer_rl_training import train_reviewer, TrainConfig

print(f"Using Cuda: {torch.cuda.is_available()}")

tm_class = TMClass(embed_state_dim=4, max_move=1)
tasks = contains_one_task_generator(8, 512, tm_class)
agent = TMAgent([8, 8, 8], tm_class).to('cuda')
interface = TMInterface(tm_class)
tape, correct_output  = tasks

print("Creating population...")
train_population = create_population(agent, 1_000_000)
print("Scoring population...")
train_scores = score_population_compiled(train_population, tasks, interface, slice_size=1000)
print("Making dataset...")
train_dataset = ReviewerDataset(train_population, train_scores)
print("Saving dataset...")
torch.save(train_dataset, "training_data/1mSmall.pt")
print("Done :)")
