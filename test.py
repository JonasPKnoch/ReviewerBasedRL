import torch
from torch import nn
from tm_task_generators import contains_one_task_generator
from tm_interface import TMAgent, TMInterface, TMClass
from reviewer_rl import fast_rollout, fast_rollout, create_population, score_function

tasks = contains_one_task_generator(10, 64)
agent = TMAgent()
interface = TMInterface()

pop = create_population(agent, 1000)

score_function(tasks, agent, interface)