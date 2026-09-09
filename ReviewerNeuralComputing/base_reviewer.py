import torch
from torch import nn

from base_world_state import BaseWorldState
from base_agent_state import BaseAgentState

class BaseReviewer(nn.Module):
    def __init__(self, *args, **kwargs):
            super(BaseReviewer, self).__init__()
        
    def forward(self, world_state: BaseWorldState, agent_state: BaseAgentState) -> torch.Tensor:
        pass