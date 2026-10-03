import torch
import copy
import math
from torch import Tensor, nn
from torch.nn import Module
from torch.utils.data import Dataset
from torch.func import stack_module_state, functional_call, vmap
from jaxtyping import Float, Int, Bool
from typing import Protocol, Callable, Self, ParamSpec, TypeVar, Any

DEVICE = 'cuda'

type AgentState = tuple[Float[Tensor, "batch ..."] | Int[Tensor, "batch ..."] | Bool[Tensor, "batch ..."], ...]
type AgentObs = Float[Tensor, "batch ..."] | Int[Tensor, "batch ..."] | Bool[Tensor, "batch ..."]
type AgentLogits = tuple[Float[Tensor, "batch ..."], ...]
type WorldState = tuple[Float[Tensor, "batch ..."] | Int[Tensor, "batch ..."] | Bool[Tensor, "batch ..."], ...]
type Reward = Float[Tensor, "batch"]
type Terminal = Bool[Tensor, "batch"]
type AgentScore = Float[Tensor, ""]
type AgentParameters = Float[Tensor, "param"]
type AgentParametersBatch = Float[Tensor, "batch param"]
type ReviewerScore = Float[Tensor, "batch"]

type PRewards = Float[Tensor, "agent batch"]
type PWorldStates = Float[Tensor, "agent batch ..."]
type PAgentScores = Float["Tensor", "agent"]

class AgentWorldInterface(Protocol):
    def get_obs(self, agent_state: AgentState, world_state: WorldState) -> AgentObs:
        ...

    def take_action(self, agent_logits: AgentLogits, agent_state: AgentState, world_state: WorldState) -> tuple[AgentState, WorldState, Reward, Terminal]:
        ...

    def initial_agent_state(self, batch_size) -> AgentState:
        ...

    def print_state(self, agent_state: AgentState, world_state: WorldState, reward: Reward, batch: int =0) -> None:
        ...

type AgentCallable = Callable[[AgentState, AgentObs], AgentLogits]

class Agent(Module):
    def forward(self, agent_state: AgentState, agent_obs: AgentObs) -> AgentLogits:
        ...

    def mutate(self, scale=0.1) -> Self:
        new_agent = copy.deepcopy(self)
        for p in new_agent.parameters():
            std = scale*math.sqrt(6.0/sum(p.data.shape)) #Basically normal Xavier but hopefully works for all vectors
            delta = torch.normal(0.0, std, size = p.data.shape, device='cuda')
            p.data += delta
        return new_agent

    def parameters_vector(self) -> AgentParameters:
        params_list = []
        for p in self.parameters():
            params_list.append(p.detach().flatten())
        return torch.concat(params_list) 

@torch.inference_mode()
def fast_rollout(agent: AgentCallable, initial_world_state: WorldState, interface: AgentWorldInterface, max_depth=20) -> Reward:
    world_state = copy.deepcopy(initial_world_state)
    agent_state = interface.initial_agent_state(initial_world_state[0].shape[0])
    sum_reward = 0
    
    for i in range(max_depth):
        obs = interface.get_obs(agent_state, world_state)
        logits = agent(agent_state, obs)
        agent_state, world_state, reward, terminal = interface.take_action(logits, agent_state, world_state)
        sum_reward += reward

    return sum_reward

compiled_rollout = torch.compile(fast_rollout, mode="reduce-overhead")

def verbose_rollout(agent: AgentCallable, initial_world_state: WorldState, interface: AgentWorldInterface, max_depth=100) -> Reward:
    world_state = copy.deepcopy(initial_world_state)
    agent_state = interface.initial_agent_state(initial_world_state[0].shape[0])
    sum_reward = 0
    
    for i in range(max_depth):
        obs = interface.get_obs(agent_state, world_state)
        logits = agent(agent_state, obs)
        agent_state, world_state, reward, terminal = interface.take_action(logits, agent_state, world_state)
        interface.print_state(agent_state, world_state, reward)
        sum_reward += reward

    return sum_reward

type TaskGenerator = Callable[[int, int, Any], WorldState]

def score_function(tasks: WorldState, agent: AgentCallable, interface: AgentWorldInterface) -> AgentScore:
    reward = fast_rollout(agent, tasks, interface)
    mean = reward.mean(dim=0)

    return mean

def compiled_score_function(tasks: WorldState, agent: AgentCallable, interface: AgentWorldInterface) -> AgentScore:
    reward = compiled_rollout(agent, tasks, interface)
    mean = reward.mean(dim=0)
    
    return mean

def create_population(parent: Agent, population_size: int, scale: float=0.1) -> list[Agent]:
    return [parent.mutate(scale) for _ in range(population_size)]

def create_population_streams(parent: Agent, population_size: int, scale: float=0.1) -> list[Agent]:
    streams = [torch.cuda.Stream() for _ in range(population_size)]
    agent_dict: dict[int, Agent] = {}

    for i in range(population_size):
        with torch.cuda.stream(streams[i]):
            agent_dict[i] = parent.mutate()

    torch.cuda.synchronize()
    
    return list(agent_dict.values())

def score_population(pop: list[Agent], tasks: WorldState, interface: AgentWorldInterface) -> list[AgentScore]:
    return [score_function(tasks, agent, interface) for agent in pop]

def score_population_streams(pop: list[Agent], tasks: WorldState, interface: AgentWorldInterface) -> list[AgentScore]:
    streams = [torch.cuda.Stream() for _ in pop]
    scores_dict: dict[int, AgentScore] = {}

    for i, agent in enumerate(pop):
        with torch.cuda.stream(streams[i]):
            scores_dict[i] = score_function(tasks, agent, interface)

    torch.cuda.synchronize()

    return list(scores_dict.values())

def score_population_compiled(pop: list[Agent], tasks: WorldState, interface: AgentWorldInterface, slice_size: int = 1000) -> list[AgentScore]:
    base_agent: Module = copy.deepcopy(pop[0]).cuda().eval().requires_grad_(False)
    scores_dict: dict[int, AgentScore] = {}
    streams = [torch.cuda.Stream() for _ in range(len(pop)//slice_size)]

    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            for j in range(slice_size):
                index = i*slice_size + j
                base_agent.load_state_dict(pop[index].state_dict())
                scores_dict[index] = compiled_score_function(tasks, base_agent, interface).clone()

    return list(scores_dict.values())

def transform_agent_params(agents: list[Agent],
                           mean: Float[Tensor, "params"] | None = None, std: Float[Tensor, "params"] | None = None
                           ) -> tuple[AgentParametersBatch, Float[Tensor, "params"], Float[Tensor, "params"]]:
    params = torch.stack([agent.parameters_vector() for agent in agents])    
    if mean == None:
        mean = params.mean(dim=0)
    if std == None:
        std = params.std(dim=0)

    params = params - mean
    params = params/std

    return params, mean, std

def transform_scores(agent_scores: list[AgentScore], 
                     mean: Float[Tensor, ""] | None = None, std: Float[Tensor, ""] | None = None
                     )-> tuple[ReviewerScore, Float[Tensor, ""], Float[Tensor, ""]]:
    scores = torch.stack(agent_scores)
    if mean == None:
        mean = scores.mean()
    if std == None:
        std = scores.std()

    scores = scores - mean
    scores = scores/std

    return scores, mean, std

class ReviewerDataset(Dataset):
    def __init__(self, agents: list[Agent], agent_scores: list[AgentScore], 
                param_mean: Float[Tensor, "params"] | None = None, param_std: Float[Tensor, "params"] | None = None,
                score_mean: Float[Tensor, ""] | None = None, score_std: Float[Tensor, ""] | None = None,
                ) -> None:
        agent_params, mean, std  = transform_agent_params(agents, param_mean, param_std)
        self.agent_params: AgentParametersBatch = agent_params
        self.param_mean: Float[Tensor, "params"] = mean
        self.param_std: Float[Tensor, "params"] = std

        agent_scores, mean, std = transform_scores(agent_scores, score_mean, score_std)
        self.agent_scores: ReviewerScore = agent_scores
        self.score_mean: Float[Tensor, ""] = mean
        self.score_std:Float[Tensor, ""] = std

    def __len__(self):
        return len(self.agent_params)

    def __getitem__(self, index) -> tuple[AgentParameters, AgentScore]:
        return self.agent_params[index], self.agent_scores[index]

class Reviewer(Module):
    def __init__(self, paramaters_template: AgentParameters, mlp_shape: list[int] = [256, 128, 64]):
        super(Reviewer, self).__init__()
        self.input_dim = paramaters_template.shape[0]

        self.mlp = nn.Sequential()
        
        self.mlp.append(nn.LayerNorm((self.input_dim)))
        self.mlp.append(nn.Linear(self.input_dim, mlp_shape[0]))
        self.mlp.append(nn.LayerNorm((mlp_shape[0])))
        self.mlp.append(nn.ReLU())
        for i in range(1, len(mlp_shape)):
            self.mlp.append(nn.Linear(mlp_shape[i-1], mlp_shape[i]))
            self.mlp.append(nn.LayerNorm((mlp_shape[i])))
            self.mlp.append(nn.ReLU())

        self.mlp.append(nn.Linear(mlp_shape[-1], 1))

    def forward(self, params: AgentParametersBatch) -> ReviewerScore:
        return self.mlp(params)