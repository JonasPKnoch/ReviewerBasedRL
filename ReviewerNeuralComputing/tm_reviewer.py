import torch
from torch import nn

from ReviewerNeuralComputing.base_reviewer import BaseReviewer
from ReviewerNeuralComputing.tm_tape import TMTape
from ReviewerNeuralComputing.tm_agent import TMAgentBrain, TMAgent

class TMReviewer(BaseReviewer):
    def __init__(self, 
                 brain_template: TMAgentBrain, 
                 brain_encoder_dim = 64, brain_encoder_layers = 3, 
                 rnn_hidden_dim = 16, rnn_layers = 3, 
                 head_dim = 32, head_layers = 4):
        super(TMReviewer, self).__init__()
        self.brain_template = brain_template
        self.brain_encoder_dim = brain_encoder_dim
        self.brain_encoder_layers  = brain_encoder_layers
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layers = rnn_layers
        self.head_dim = head_dim
        self.head_layers  = head_layers
        self.debug_agent = None

        params_dim = 0
        with torch.no_grad(): #Do I need this?
            params_dim = sum([p.numel() for p in brain_template.parameters()])

        self.brain_encoder = nn.Sequential()

        self.brain_encoder.append(nn.LayerNorm((params_dim)))
        self.brain_encoder.append(nn.Linear(params_dim, brain_encoder_dim))
        self.brain_encoder.append(nn.ReLU())
        for _ in range(brain_encoder_layers):
            self.brain_encoder.append(nn.Linear(brain_encoder_dim, brain_encoder_dim))
            self.brain_encoder.append(nn.ReLU())

        self.brain_to_rnn = nn.Linear(brain_encoder_dim, rnn_hidden_dim*rnn_layers)

        self.rnn = nn.RNN(brain_template.symbol_count, rnn_hidden_dim, rnn_layers)

        self.rnn_to_head = nn.Linear(rnn_hidden_dim, int(head_dim/2))
        self.brain_to_head = nn.Linear(brain_encoder_dim, int(head_dim/2))

        self.head = nn.Sequential()
        for _ in range(head_layers):
                    self.head.append(nn.Linear(head_dim, head_dim))
                    self.head.append(nn.ReLU())
        self.head.append(nn.Linear(head_dim, 1))
        self.head.append(nn.Sigmoid())

    def forward(self, tape: TMTape, agent_state: TMAgent) -> torch.Tensor:
        params_list = []
        for p in agent_state.brain.parameters():
            params_list.append(p.detach().flatten())
        params_input = torch.concat(params_list)


        brain_embedding = self.brain_encoder(params_input)

        brain_rnn_input = self.brain_to_rnn(brain_embedding).reshape(self.rnn_layers, self.rnn_hidden_dim)
        tape_input = nn.functional.one_hot(torch.tensor(tape.tape_arr), tape.symbol_count).to(torch.float32)
        _, rnn_output = self.rnn(tape_input, brain_rnn_input)

        head_input = torch.concat([self.rnn_to_head(rnn_output[0]), self.brain_to_head(brain_embedding)])

        head_output = self.head(head_input)

        return head_output