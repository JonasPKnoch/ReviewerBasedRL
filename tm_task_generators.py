import torch
from torch import nn
from reviewer_rl import TaskGenerator
from tm_interface import TMWorldState, TMClass

def contains_one_task_generator(max_size: int, batch_size: int, tm_class: TMClass = TMClass()) -> TMWorldState:
    symbols = torch.randint(0, tm_class.symbol_count, (batch_size, max_size), device='cuda') #Creat a complete grid of random 0-symbol_count values
    zero_tapes = torch.rand((batch_size,), device='cuda') > 0.5
    symbols[zero_tapes] = 0

    sizes = torch.randint(2, max_size-2, (batch_size,), device='cuda').unsqueeze(1) 
    indices = torch.arange(max_size, device='cuda').unsqueeze(0) 
    symbols[indices > sizes] = tm_class.symbol_count
    symbols[:,0] = tm_class.symbol_count

    tapes = nn.functional.one_hot(symbols, tm_class.symbol_count + 1)
    contains_one = torch.any(symbols == 1, dim=1).to(torch.int)

    return (tapes, contains_one)

_contains_one_task_generator: TaskGenerator = contains_one_task_generator #Just for type checking