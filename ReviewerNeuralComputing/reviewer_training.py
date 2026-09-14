import torch
from torch import nn
from time import time
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob

from base_world_state import BaseWorldState
from base_agent_state import BaseAgentState
from base_reviewer import BaseReviewer

class ReviewerDataset(Dataset):
    def __init__(self, dir="data", prefix="training_samples", shard_size=1000):
        self.shard_paths = glob.glob(f"{dir}/{prefix}_*.pt")
        self.shard_size = shard_size
        self._cache = {}
        self.shard_count = len(self.shard_paths)

    def __len__(self):
        return self.shard_count * self.shard_size

    def __getitem__(self, idx):
        shard_i, offset = divmod(idx, self.shard_size)

        return self.load_shard(shard_i)[offset]

    def clear_cache(self):
        self._cache = {}

    def load_shard(self, shard_index: int):
        if shard_index not in self._cache:
            self._cache = {shard_index: torch.load(self.shard_paths[shard_index], weights_only=False)}
        return self._cache[shard_index]


def train(reviewer: BaseReviewer, training_data: ReviewerDataset, epochs = 200, batch_size = 64, lr = 0.001, log_every = 5):
    current_loss = 0
    all_losses = []
    reviewer.train()
    optimizer = torch.optim.Adam(reviewer.parameters(), lr=lr)
    criterion = nn.BCELoss()

    start_time = time()
    print(f"training on data set with n = {len(training_data)}")

    for epoch in range(epochs):
        shard_index = random.randint(0, training_data.shard_count-1)
        indices = list(range(len(training_data.shard_size)))
        random.shuffle(indices)
        batch_count = len(training_data) // batch_size
        batch_indices = np.array_split(indices, batch_count)

        for batch in batch_indices:
            batch_loss = 0
            for index in batch:
                agent, score = training_data.load_shard(shard_index)
                output = reviewer(agent)
                loss = criterion(output, torch.tensor([score], dtype=torch.float32))
                batch_loss += loss / len(batch)
            print(f"{batch_loss}")
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

def train(reviewer: BaseReviewer, training_data: ReviewerDataset, epochs = 200, batch_size=64, batch_count=100, batch_source_files=4, lr = 0.001, log_every = 5):
    current_loss = 0
    all_losses = []
    reviewer.train()
    optimizer = torch.optim.Adam(reviewer.parameters(), lr=lr)
    criterion = nn.BCELoss()

    start_time = time()
    print(f"training on data set with n = {len(training_data)}")

    for epoch in range(epochs):
        for batch_index in range(batch_count):
            #This is weird and janky, but I am doing it as a hacky way to avoid loading too much data at once. Could of course be imrpoved but hopefully this works :P
            shard_indices = list(range(training_data.shard_count))
            random.shuffle(shard_indices)
            shard_indices = shard_indices[:batch_source_files]

            sample_indices = [random.choice(shard_indices)*training_data.shard_size + random.randint(0, training_data.shard_size) for _ in range(batch_size)]

            batch_loss = 0
            for sample_index in sample_indices:
                agent, score = training_data[sample_index]
                output = reviewer(agent)
                loss = criterion(output, torch.tensor([score], dtype=torch.float32))
                batch_loss += loss/batch_size

            batch_loss.backward()
            print(f"Batch loss: {batch_loss} {batch_index}/{batch_count}")

            nn.utils.clip_grad_norm_(reviewer.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()
                
            training_data.clear_cache()
            current_loss += batch_loss.item()/batch_count

        all_losses.append(current_loss/training_data.batch_size)
        if epoch % log_every == 0:
            print(f"{epoch}/{epochs}: average batch loss = {all_losses[-1]}")
        current_loss = 0

    return all_losses