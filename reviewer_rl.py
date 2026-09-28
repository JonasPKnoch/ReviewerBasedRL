import torch
import copy
import math
from torch import Tensor, nn
from torch.nn import Module
from torch.utils.data import Dataset
from torch.func import stack_module_state, functional_call, vmap
from jaxtyping import Float, Int, Bool
from typing import Protocol, Callable, Self, ParamSpec, TypeVar, Any

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

type AgentCallable = Callable[[AgentState, AgentObs], AgentLogits]

class Agent(Module):
    def forward(self, agent_state: AgentState, agent_obs: AgentObs) -> AgentLogits:
        ...

    def mutate(self, scale=0.1) -> Self:
        new_agent = copy.deepcopy(self)
        for p in new_agent.parameters():
            std = scale*math.sqrt(6.0/sum(p.data.shape)) #Basically normal Xavier but hopefully works for all vectors
            delta = torch.normal(0.0, std, size = p.data.shape)
            p.data += delta
        return new_agent

    def parameters_vector(self) -> AgentParameters:
        params_list = []
        for p in self.parameters():
            params_list.append(p.detach().flatten())
        return torch.concat(params_list) 


def fast_rollout(agent: AgentCallable, initial_world_state: WorldState, interface: AgentWorldInterface, max_depth=100) -> Reward:
    initial_agent_state = interface.initial_agent_state(initial_world_state[0].shape[0])
    obs = interface.get_obs(initial_agent_state, initial_world_state)
    logits = agent(initial_agent_state, obs)
    agent_state, world_state, reward, terminal = interface.take_action(logits, initial_agent_state, initial_world_state)
    sum_reward = reward
    
    for i in range(max_depth):
        obs = interface.get_obs(agent_state, world_state)
        logits = agent(agent_state, obs)
        agent_state, world_state, reward, terminal = interface.take_action(logits, agent_state, world_state)

        sum_reward += reward

    return sum_reward


type TaskGenerator = Callable[[int, int, Any], WorldState]

def score_function(tasks: WorldState, agent: AgentCallable, interface: AgentWorldInterface) -> AgentScore:
    reward = fast_rollout(agent, tasks, interface)
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

class ReviewerDataset(Dataset):
    def __init__(self, agents: list[Agent], agent_scores: list[AgentScore]) -> None:
        self.agent_params: list[AgentParameters] = [agent.parameters_vector() for agent in agents]
        self.agent_scores: list[AgentScore] = agent_scores

    def __len__(self):
        return len(self.agent_params)

    def __getitem__(self, index) -> tuple[AgentParameters, AgentScore]:
        return self.agent_params[index], self.agent_scores[index]
    
class Reviewer(Module):
    def __init__(self, paramaters_template: AgentParameters, mlp_shape: list[int] = [512, 256, 128, 64]):
        super(Reviewer, self).__init__()
        self.input_dim = paramaters_template.shape[0]

        self.mlp = nn.Sequential()
        
        self.mlp.append(nn.LayerNorm((self.input_dim)))
        self.mlp.append(nn.Linear(self.input_dim, mlp_shape[0]))
        self.mlp.append(nn.ReLU())
        for i in range(1, len(mlp_shape)):
            self.mlp.append(nn.Linear(mlp_shape[i-1], mlp_shape[i]))
            self.mlp.append(nn.ReLU())

        self.mlp.append(nn.Linear(mlp_shape[-1], 1))
        self.mlp.append(nn.Sigmoid())

    def forward(self, params: AgentParametersBatch) -> ReviewerScore:
        return self.mlp(params)