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

type MirrorReviewerInputParams = dict[str, Float[Tensor, "batch param"]]

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

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameters_vector(self) -> AgentParameters:
        params_list = []
        for p in self.parameters():
            params_list.append(p.detach().flatten())
        return torch.concat(params_list) 

    def parameters_dict(self) -> MirrorReviewerInputParams:
        params_list = []
        params_dict = {}
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                module_params = []
                for p in m.parameters():
                    module_params.append(p.detach().flatten())
                p_v = torch.cat(module_params) 
                params_dict[n] = p_v
                params_list.append(p_v)

        all_params = torch.cat(params_list) 
        params_dict["all_params"] = all_params
        return params_dict
        


@torch.inference_mode()
def fast_rollout(agent: AgentCallable, initial_world_state: WorldState, interface: AgentWorldInterface, max_depth=20) -> Reward:
    world_state = copy.deepcopy(initial_world_state)
    agent_state = interface.initial_agent_state(initial_world_state[0].shape[0])
    sum_reward = torch.zeros((initial_world_state[0].shape[0]), device=initial_world_state[0].device)
    
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
    
    sum_reward = torch.zeros((initial_world_state[0].shape[0]), device=initial_world_state[0].device)
    
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

class FlatReviewerDataset(Dataset):
    def __init__(self, agents: list[Agent], agent_scores: list[AgentScore], 
                param_mean: Float[Tensor, "params"] | None = None, param_std: Float[Tensor, "params"] | None = None,
                score_mean: Float[Tensor, ""] | None = None, score_std: Float[Tensor, ""] | None = None,
                ) -> None:
        agent_params, mean, std  = transform_agent_params(agents, param_mean, param_std)
        self.agent_params: AgentParametersBatch = agent_params
        self.param_mean: Float[Tensor, "params"] = mean
        self.param_std: Float[Tensor, "params"] = std

        scores, mean, std = transform_scores(agent_scores, score_mean, score_std)
        self.agent_scores: ReviewerScore = scores
        self.score_mean: Float[Tensor, ""] = mean
        self.score_std:Float[Tensor, ""] = std

    def __len__(self):
        return len(self.agent_params)

    def __getitem__(self, index) -> tuple[AgentParameters, AgentScore]:
        return self.agent_params[index], self.agent_scores[index]

class MirrorReviewerDataset(Dataset):
    def __init__(self, agents: list[Agent], agent_scores: list[AgentScore], 
                param_mean: Float[Tensor, "params"] | None = None, param_std: Float[Tensor, "params"] | None = None,
                score_mean: Float[Tensor, ""] | None = None, score_std: Float[Tensor, ""] | None = None,
                ) -> None:
        agent_params, mean, std  = transform_agent_params(agents, param_mean, param_std)
        self.agent_params: AgentParametersBatch = agent_params
        self.param_mean: Float[Tensor, "params"] = mean
        self.param_std: Float[Tensor, "params"] = std

        scores, mean, std = transform_scores(agent_scores, score_mean, score_std)
        self.agent_scores: ReviewerScore = scores
        self.score_mean: Float[Tensor, ""] = mean
        self.score_std:Float[Tensor, ""] = std

    def __len__(self):
        return len(self.agent_params)

    def __getitem__(self, index) -> tuple[AgentParameters, AgentScore]:
        return self.agent_params[index], self.agent_scores[index]

class FlatReviewer(Module):
    def __init__(self, paramaters_template: AgentParameters, mlp_shape: list[int] = [256, 128, 64]):
        super(FlatReviewer, self).__init__()
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

class MirrorReviewer(Module):
    def __init__(self, mirroring: Agent, width_factor: int = 2):
        super(MirrorReviewer, self).__init__()
        self.input_params: MirrorReviewerInputParams = {}
        self.width_factor = width_factor
        self.agent: Agent = copy.deepcopy(mirroring)

        module_names: dict[Module, str] = {n: m for m, n in self.agent.named_modules()}
        self.recursive_replace(self.agent, module_names)

    def forward(self, params: MirrorReviewerInputParams) -> PAgentScores:
        raise NotImplemented("Reviewer must implement forward")


    def recursive_replace(self, module: Module, module_names: dict[Module, str]) -> None:
        for n, _ in module.named_parameters(recurse=False):
            raise ValueError(f"Non-linear module {module_names[module]} has parameter {n}!")
        if isinstance(module, nn.Sequential):
            self.replace_sequential(module, module_names)
        else:
            self.replace_module(module, module_names)

    def replace_sequential(self, module: nn.Sequential, module_names: dict[Module, str]) -> None:
        for i in range(len(module)):
            child = module[i]
            if isinstance(child, nn.Linear):
                module[i] = MirrorReviewerLinear(child, module_names[child], self.input_params, self.width_factor).cuda()
            else:
                self.recursive_replace(child, module_names)

    def replace_module(self, module: Module, module_names: dict[Module, str]) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, MirrorReviewerLinear(child, module_names[child], self.input_params, self.width_factor).cuda())
            else:
                self.recursive_replace(child, module_names)

class MirrorReviewerLinear(Module):
    def __init__(self, mirroring: nn.Linear, module_name: str, input_params: MirrorReviewerInputParams, width_factor: int, hidden_size: int = 16, mlp_layer_count: int = 2):
        super(MirrorReviewerLinear, self).__init__()
        self.module_name = module_name
        self.input_params = input_params

        param_count = sum(p.numel() for p in mirroring.parameters())
        self.bilinear = nn.Bilinear(mirroring.in_features*width_factor, param_count, hidden_size)

        self.hidden_input_norm = nn.LayerNorm([mirroring.in_features*width_factor])
        self.param_input_norm = nn.LayerNorm([param_count])

        self.mlp = nn.Sequential()
        for i in range(1, mlp_layer_count):
            self.mlp.append(nn.Linear(hidden_size, hidden_size))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(hidden_size, mirroring.out_features*width_factor))

    def forward(self, hidden: Any) -> Any:
        params = self.input_params[self.module_name]

        hidden_normalized = self.hidden_input_norm(hidden)
        params_normalized = self.param_input_norm(params)

        x = self.bilinear(hidden_normalized, params_normalized)
        x = self.mlp(x)

        return x

