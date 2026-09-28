from reviewer_rl import AgentWorldInterface, Agent, Reward, Terminal
import torch
from torch import nn, Tensor
from typing import NamedTuple
from torch.distributions import Categorical
from jaxtyping import Float, Int, Bool

class TMClass(NamedTuple):
    symbol_count: int = 2
    embed_state_dim: int = 8
    max_move: int = 3

type TMAgentState = tuple[Float[Tensor, "batch embed_state"], Int[Tensor, "batch"], Bool[Tensor, "batch"]]
type TMAgentObs = Float[Tensor, "batch symbol_count+1"]
type TMAgentLogits = tuple[Float[Tensor, "batch embed_state"], Float[Tensor, "batch symbol_count"], Float[Tensor, "batch max_move*2"], Float[Tensor, "batch 3"]]
type TMWorldState = tuple[Float[Tensor, "batch max_world_size symbol_count+1"], Int[Tensor, "batch"]]

def sample_logits(logits: Float[Tensor, "batch cats"]) -> Int[Tensor, "batch"]:
    norm_logits = logits - logits.logsumexp(dim=1, keepdim=True)
    probs = torch.exp(norm_logits)
    cumprobs = torch.cumsum(probs, dim=1)

    sample_locations = torch.rand((logits.shape[0]))
    samples = torch.zeros((logits.shape[0]), dtype=torch.int)

    for i in range(probs.shape[1]-1, -1, -1):
        samples[sample_locations < cumprobs[:,i]] = i #There might be a better way to do this, but honestly, fuck broadcasting :P

    return samples


class TMInterface(AgentWorldInterface):
    def __init__(self, tm_class: TMClass = TMClass()) -> None:
       self.tm_class = tm_class

    def get_obs(self, agent_state: TMAgentState, world_state: TMWorldState) -> TMAgentObs:
        embed_state, position, halted = agent_state
        tape, correct_output = world_state
        obs = tape[torch.arange(tape.shape[0]), torch.clamp(position, 0, tape.shape[1]-1)].to(torch.float)

        return obs


    def take_action(self, agent_logits: TMAgentLogits, agent_state: TMAgentState, world_state: TMWorldState) -> tuple[TMAgentState, TMWorldState, Reward, Terminal]:
        embed_output, write_logits, move_logits, halt_logits = agent_logits
        embed_state, position, halted = agent_state
        tape, correct_output  = world_state

        write_symbol = sample_logits(write_logits)
        move_index = sample_logits(move_logits)
        halt = sample_logits(halt_logits)

        new_tape = tape.detach()
        write = torch.zeros((self.tm_class.symbol_count + 1), dtype=torch.long)
        write[write_symbol] = 1
        new_tape[torch.clamp(position, 0, tape.shape[1]-1)] = write

        move = move_index - self.tm_class.max_move
        move += move > 1
        new_position = position + move

        reward = halt == (correct_output + 1)
        reward = reward.to(torch.float)
        reward.masked_fill(halted, 0) #Esnure that any halted output gives no reward


        new_halted = torch.logical_or(halted, (halt > 0))

        new_agent_state = (embed_output, new_position, new_halted)
        new_world_state = (new_tape, correct_output)

        return (new_agent_state, new_world_state, reward, new_halted)


    def initial_agent_state(self, batch_size: int) -> TMAgentState:
        return (
            torch.zeros((batch_size, self.tm_class.embed_state_dim), dtype=torch.float), 
            torch.zeros((batch_size,), dtype=torch.int),
            torch.zeros((batch_size,), dtype=torch.bool))

class TMAgent(Agent):
    def __init__(self, mlp_shape: list[int] = [16, 16, 16], tm_class: TMClass = TMClass()):
        super(TMAgent, self).__init__()
        if mlp_shape[0]%2 == 1:
                raise ValueError("Hidden dimension must be divisible by 2")
        self.tm_class = tm_class

        self.obs_input = nn.Linear(tm_class.symbol_count + 1, int(mlp_shape[0]/2))
        self.embed_state_input = nn.Linear(tm_class.embed_state_dim, int(mlp_shape[0]/2))

        self.mlp = nn.Sequential()
        self.mlp.append(nn.ReLU())
        for i in range(1, len(mlp_shape)):
            self.mlp.append(nn.Linear(mlp_shape[i-1], mlp_shape[i]))
            self.mlp.append(nn.ReLU())

        self.embed_state_head = nn.Linear(mlp_shape[-1], tm_class.embed_state_dim)
        self.write_head = nn.Sequential(nn.Linear(mlp_shape[-1], tm_class.symbol_count), nn.LogSoftmax(dim=1))
        self.move_head = nn.Sequential(nn.Linear(mlp_shape[-1], tm_class.max_move*2), nn.LogSoftmax(dim=1))
        self.halt_head = nn.Sequential(nn.Linear(mlp_shape[-1], 3), nn.LogSoftmax(dim=1))

    def forward(self, agent_state: TMAgentState, agent_obs: TMAgentObs) -> TMAgentLogits:
        embed_state, position, halted = agent_state

        state_project = self.embed_state_input(embed_state)
        read_project = self.obs_input(agent_obs)

        input = torch.concat([read_project, state_project], dim=1)
        mlp_output = self.mlp(input)

        embed_state_output = self.embed_state_head(mlp_output)
        write_logits = self.write_head(mlp_output)
        move_logits = self.move_head(mlp_output)
        halt_logits = self.halt_head(mlp_output)

        return (embed_state_output, write_logits, move_logits, halt_logits) 