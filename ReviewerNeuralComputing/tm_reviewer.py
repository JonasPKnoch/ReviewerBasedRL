import torch
from torch import nn

from base_reviewer import BaseReviewer
from tm_tape import TMTape
from tm_agent import TMAgentBrain, TMAgent

class TMReviewer(BaseReviewer):
    def __init__(self, 
                 brain_template: TMAgentBrain, 
                 mlp_shape = [1024, 512, 256, 128, 64]):
        super(TMReviewer, self).__init__()
        self.mlp_shape = mlp_shape
        self.debug_agent = None

        self.params_dim = 0
        with torch.no_grad(): #Do I need this?
            self.params_dim = sum([p.numel() for p in brain_template.parameters()])

        self.mlp = nn.Sequential()

        self.mlp.append(nn.LayerNorm((self.params_dim)))
        self.mlp.append(nn.Linear(self.params_dim, mlp_shape[0]))
        self.mlp.append(nn.ReLU())
        for i in range(1, len(mlp_shape)):
            self.mlp.append(nn.Linear(mlp_shape[i-1], mlp_shape[i]))
            self.mlp.append(nn.ReLU())

        self.mlp.append(nn.Linear(mlp_shape[-1], 1))
        self.mlp.append(nn.Sigmoid())

    def forward(self, agent_state: TMAgent) -> torch.Tensor:
        params_list = []
        for p in agent_state.brain.parameters():
            params_list.append(p.detach().flatten())
        params_input = torch.concat(params_list)

        output = self.mlp(params_input)

        return output