from reviewer_rl import AgentWorldInterface, Agent, Reward, Terminal, MirrorReviewer, MirrorReviewerInputParams, PAgentScores
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
    logits = logits.float()
    noise = torch.empty_like(logits).exponential_().log()   # -Gumbel
    return (logits - noise).argmax(-1)


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
        embed_state, position, terminal = agent_state
        tape, correct_output  = world_state

        batch_size: int = tape.shape[0]

        write_symbol = sample_logits(write_logits)
        move_index = sample_logits(move_logits)
        halt = sample_logits(halt_logits)

        idx = torch.arange(batch_size, device=tape.device)
        write_one_hot = torch.nn.functional.one_hot(write_symbol, tape.shape[-1]).to(tape.dtype)
        tape[idx, torch.clamp(position, 0, tape.shape[1]-1)] = write_one_hot

        move = move_index - self.tm_class.max_move
        move = move + (move > 1)
        position = position + move

        reward = halt == (correct_output + 1)
        reward = reward.to(torch.float)
        reward = torch.where(terminal, 0., reward) #Ensure that any halted output gives no reward

        new_terminal = torch.logical_or(terminal, (halt > 0))

        new_agent_state = (embed_output, position, new_terminal)
        new_world_state = (tape, correct_output)

        return (new_agent_state, new_world_state, reward, new_terminal)


    def initial_agent_state(self, batch_size: int) -> TMAgentState:
        return (
            torch.zeros((batch_size, self.tm_class.embed_state_dim), dtype=torch.float, device='cuda'), 
            torch.ones((batch_size,), dtype=torch.int, device='cuda'),
            torch.zeros((batch_size,), dtype=torch.bool, device='cuda'))

    def print_state(self, agent_state: TMAgentState, world_state: TMWorldState, reward: Reward, batch: int = 0) -> None:
        embed_state, position, halted = agent_state
        tape, correct_output  = world_state
        position = position[batch]
        halted = halted[batch]
        tape = tape[batch]
        reward = reward[batch]
        
        tape_arr = [f" {int(torch.argmax(el))} " for el in tape]

        clamped_position = torch.clamp(position, 0, len(tape)-1)

        agent_str = f"({tape_arr[clamped_position][1]})"
        tape_arr[clamped_position] = agent_str

        tape_str = "|".join(tape_arr)

        print(f"{tape_str}      REWARD: {reward} HALTED: {halted}")


class TMAgent(Agent):
    def __init__(self, mlp_shape: list[int] = [16, 16, 16], tm_class: TMClass = TMClass()):
        super(TMAgent, self).__init__()
        if mlp_shape[0]%2 == 1:
                raise ValueError("Hidden dimension must be divisible by 2")
        self.tm_class = tm_class
        self.mlp_shape = mlp_shape

        self.obs_input = nn.Linear(tm_class.symbol_count + 1, int(mlp_shape[0]/2))
        self.embed_state_input = nn.Linear(tm_class.embed_state_dim, int(mlp_shape[0]/2))

        self.mlp = nn.Sequential()
        self.mlp.append(nn.ReLU())
        for i in range(1, len(mlp_shape)):
            self.mlp.append(nn.Linear(mlp_shape[i-1], mlp_shape[i]))
            self.mlp.append(nn.ReLU())

        self.embed_state_head = nn.Linear(mlp_shape[-1], tm_class.embed_state_dim)
        self.write_head = nn.Sequential(nn.Linear(mlp_shape[-1], tm_class.symbol_count), nn.LogSoftmax(dim=-1))
        self.move_head = nn.Sequential(nn.Linear(mlp_shape[-1], tm_class.max_move*2), nn.LogSoftmax(dim=-1))
        self.halt_head = nn.Sequential(nn.Linear(mlp_shape[-1], 3), nn.LogSoftmax(dim=-1))

    def forward(self, agent_state: TMAgentState, agent_obs: TMAgentObs) -> TMAgentLogits:
        embed_state, position, halted = agent_state

        state_project = self.embed_state_input(embed_state)
        read_project = self.obs_input(agent_obs)

        input = torch.concat([read_project, state_project], dim=-1)
        mlp_output = self.mlp(input)

        embed_state_output = self.embed_state_head(mlp_output)
        write_logits = self.write_head(mlp_output)
        move_logits = self.move_head(mlp_output)
        halt_logits = self.halt_head(mlp_output)

        return (embed_state_output, write_logits, move_logits, halt_logits) 

class TMAgentReviewer(MirrorReviewer):
    def __init__(self, agent: TMAgent, width_factor: int = 2):
        super(TMAgentReviewer, self).__init__(agent, width_factor)
        print(agent.embed_state_input)
        self.agent: TMAgent = self.agent
        self.agent.write_head.pop(1)
        self.agent.move_head.pop(1)
        self.agent.halt_head.pop(1)


        self.obs_input_encode = nn.Linear(
            agent.param_count(), 
            agent.obs_input.in_features * width_factor)
        self.embed_state_input_encode = nn.Linear(
            agent.param_count(), 
            agent.embed_state_input.in_features * width_factor)

        self.output_decode = nn.Linear(
            (agent.tm_class.embed_state_dim + 
             agent.tm_class.symbol_count +
             agent.tm_class.max_move*2 +
             3)*width_factor,
             1
        )

    def forward(self, params: MirrorReviewerInputParams) -> PAgentScores:
        self.input_params.update(params)
        all_params = params["all_params"]

        obs_encode = self.obs_input_encode(all_params)
        embed_encode = self.embed_state_input_encode(all_params)

        print(embed_encode.shape)
        print(self.agent.embed_state_input)
        embed_state_output, write_logits, move_logits, halt_logits = self.agent((embed_encode, None, None), obs_encode)

        output = torch.cat([embed_state_output, write_logits, move_logits, halt_logits], dim=-1)

        prediction = self.output_decode(output)

        return prediction
