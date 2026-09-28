

import torch
from torch import nn
from tm_task_generators import contains_one_task_generator
from tm_interface import TMAgent, TMInterface, TMClass
from reviewer_rl import create_population, score_population, score_population_streams, ReviewerDataset, Reviewer
from reviewer_rl_training import train_reviewer



tasks = contains_one_task_generator(10, 256)
agent = TMAgent().to('cuda')
interface = TMInterface()


train_population = create_population(agent, 50)
train_scores = score_population(train_population, tasks, interface)


train_dataset = ReviewerDataset(train_population, train_scores)


test_population = create_population(agent, 50)
test_scores = score_population(test_population, tasks, interface)


test_dataset = ReviewerDataset(test_population, test_scores)


reviewer = Reviewer(agent.parameters_vector()).to('cuda')


reviewer(test_population[5].parameters_vector().to('cuda'))

results = train_reviewer(reviewer, train_dataset, test_dataset)


