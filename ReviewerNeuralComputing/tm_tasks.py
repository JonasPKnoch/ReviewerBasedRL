import numpy as np
import torch

from tm_agent import TMAgent, TMAgentBrain
from tm_tape import TMTape
from world_model_runner import WorldModelRunner

SYMBOL_COUNT = 2

class TMTask:
    def __init__(self, tape_arr: np.ndarray, target_output: int, name: str):
        self.tape_arr = tape_arr
        self.target_output = target_output
        self.name = name

def display_tasks(tasks):
    for task in tasks:
        print(f"{"".join([str(e) for e in task.tape])} OUT: {str(task.target)}, NAME: {task.name}")

def generate_contains_one_tasks(num_tasks: int, min_len: int = 2, max_len: int = 8,
                                 seed: int = 0) -> list[TMTask]:
    """Target = 1 if the tape contains at least one '1' symbol, else 0."""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(num_tasks):
        length = rng.integers(min_len, max_len + 1)

        if rng.random() > 0.5:
            tape = rng.integers(0, SYMBOL_COUNT, size=length)
        else:
            tape = np.zeros(length, dtype=int)
        target = int(np.any(tape == 1))
        tasks.append(TMTask(tape, target, f"contains_one_{i}"))
    return tasks


def generate_parity_tasks(num_tasks: int, min_len: int = 2, max_len: int = 8,
                           seed: int = 0) -> list[TMTask]:
    """Target = 1 if the number of '1' symbols on the tape is odd, else 0."""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(num_tasks):
        length = rng.integers(min_len, max_len + 1)
        tape = rng.integers(0, SYMBOL_COUNT, size=length)
        target = int(np.sum(tape) % 2)
        tasks.append(TMTask(tape, target, f"parity_{i}"))
    return tasks


def generate_first_equals_last_tasks(num_tasks: int, min_len: int = 2, max_len: int = 8,
                                      seed: int = 0) -> list[TMTask]:
    """Target = 1 if the first and last symbols on the tape match, else 0."""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(num_tasks):
        length = rng.integers(min_len, max_len + 1)
        tape = rng.integers(0, SYMBOL_COUNT, size=length)
        target = int(tape[0] == tape[-1])
        tasks.append(TMTask(tape, target, f"first_eq_last_{i}"))
    return tasks

def generate_palindrome_tasks(num_tasks: int, min_len: int = 2, max_len: int = 8,
                                      seed: int = 0) -> list[TMTask]:
    """Target = 1 if the tape is a palindrome, else 0."""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(num_tasks):
        length = rng.integers(min_len, max_len + 1)
        is_palindrome = rng.integers(0, 2)
        tape = rng.integers(0, SYMBOL_COUNT, size=length)

        if is_palindrome:
            for i in range(length//2):
                tape[-i - 1] = tape[i]
        else:
            random_target = np.int64(1)
            for i in range(length//2):
                if tape[-i - 1] != tape[i]:
                    random_target = np.int64(0)
                    break
            is_palindrome = random_target

        tasks.append(TMTask(tape, is_palindrome, f"palindrome_{i}"))
    return tasks

def generate_identical_seq_tasks(num_tasks: int, min_len: int = 2, max_len: int = 8,
                                      seed: int = 0) -> list[TMTask]:
    """Target = 1 if the first half of the tape is identical to the last, else 0."""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(num_tasks):
        half_length = rng.integers(min_len//2, max_len//2 + 1)
        is_identical = rng.integers(0, 2)
        tape = rng.integers(0, SYMBOL_COUNT, size=half_length*2)

        if is_identical:
            for i in range(half_length):
                tape[half_length + i] = tape[i]
        else:
            random_target = np.int64(1)
            for i in range(half_length):
                if tape[half_length + i] != tape[i]:
                    random_target = np.int64(0)
                    break
            is_identical = random_target

        tasks.append(TMTask(tape, is_identical, f"identical_seq_{i}"))
    return tasks

def generate_1bit_sorted_tasks(num_tasks: int, min_len: int = 2, max_len: int = 8,
                                      seed: int = 0) -> list[TMTask]:
    """Target = 1 if the tape is sorted (in 1 bit), else 0."""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(num_tasks):
        length = rng.integers(min_len, max_len + 1)
        is_sorted = rng.integers(0, 2)
        tape = rng.integers(0, SYMBOL_COUNT, size=length)
        
        sorted_tape = np.sort(tape)
        if np.array_equal(tape, sorted_tape):
            is_sorted = np.int64(1)
        elif  is_sorted:
            tape = sorted_tape

        tasks.append(TMTask(tape, is_sorted, f"sorted_1bit_{i}"))
    return tasks

def to_binary_arr(seq, width):
    binary_str = ""
    for el in seq:
        binary_str += np.binary_repr(el, width)
    
    return np.array([int(c) for c in binary_str], dtype=np.int64)

def generate_2bit_sorted_tasks(num_tasks: int, min_len: int = 2, max_len: int = 8,
                                      seed: int = 0) -> list[TMTask]:
    """Target = 1 if the tape is sorted (in 2 bits), else 0."""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(num_tasks):
        sequence_length = rng.integers(min_len//2, max_len//2 + 1)
        tape_length = sequence_length*2
        is_sorted = rng.integers(0, 2)

        num_seq = rng.integers(0, 4, size=sequence_length)
        sorted_seq = np.sort(num_seq)
        if np.array_equal(num_seq, sorted_seq):
            is_sorted = np.int64(1)
        elif  is_sorted:
            num_seq = sorted_seq
        
        tape = to_binary_arr(num_seq, 2)
        
        tasks.append(TMTask(tape, is_sorted, f"sorted_2bit_{i}"))
    return tasks

def evaluate_agents(agents: list[TMAgent], tasks: list[TMTask]):
    scored_agents = []
    for agent in agents:
        correct = 0
        for task in tasks:
            tape = TMTape(task.tape_arr)
            runner = WorldModelRunner(tape, agent)
            final_outout = runner.rollout()
            if int(final_outout) == task.target_output:
                correct += 1

        score = float(correct)/float(len(tasks))
        scored_agents.append((agent, score))
        print(f"Agent scored {len(scored_agents)}/{len(agents)}")

    scored_agents.sort(key=lambda e: -e[1])
    return scored_agents

def generate_reviewer_training_data(tasks: list[TMTask], agent_count: int, symbol_count, embed_state_dim
                                    ) -> list[tuple[TMTape, TMAgent, int]]:
    tapes = []
    for task in tasks:
        tape = TMTape(task.tape_arr)
        tapes.append(tape)
    
    agents = []
    for _ in range(agent_count):
        agent = TMAgent(TMAgentBrain(symbol_count, embed_state_dim, 16, 3), torch.zeros((embed_state_dim)), 0, symbol_count)
        agents.append(agent)

    training_samples = []
    for i in range(len(tasks)):
        task = tasks[i]
        initial_tape = tapes[i]
        for agent in agents:
            runner = WorldModelRunner(initial_tape, agent)
            output = runner.rollout()
            correct = 1 if int(output) == task.target_output else 0
            training_samples.append((tape, agent, correct))

    return training_samples

