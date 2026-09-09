import torch
from torch import nn
from time import time
import random
import numpy as np

from ReviewerNeuralComputing.base_world_state import BaseWorldState
from ReviewerNeuralComputing.base_agent_state import BaseAgentState
from ReviewerNeuralComputing.base_reviewer import BaseReviewer

def train(reviewer: BaseReviewer, training_data: list[tuple[BaseWorldState, BaseAgentState, int]], epochs = 200, batch_size = 64, lr = 0.0001, log_every = 5):
    current_loss = 0
    all_losses = []
    reviewer.train()
    optimizer = torch.optim.Adam(reviewer.parameters(), lr=lr)
    criterion = nn.BCELoss()

    start_time = time()
    print(f"training on data set with n = {len(training_data)}")

    for epoch in range(epochs):
        indices = list(range(len(training_data)))
        random.shuffle(indices)
        batch_count = len(training_data) // batch_size
        batch_indices = np.array_split(indices, batch_count)

        for batch in batch_indices:
            batch_loss = 0
            for index in batch:
                world, agent, target_output = training_data[index]
                output = reviewer(world, agent)
                loss = criterion(output, torch.tensor([target_output], dtype=torch.float32))
                batch_loss += loss / len(batch)

            batch_loss.backward()

            nn.utils.clip_grad_norm_(reviewer.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()
                

            current_loss += batch_loss.item()/len(batch)

        all_losses.append(current_loss/batch_count)
        if epoch % log_every == 0:
            print(f"{epoch}/{epochs}: average batch loss = {all_losses[-1]}")
        current_loss = 0

    return all_losses