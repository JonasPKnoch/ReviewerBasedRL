import torch
from torch import nn
from typing import Self

from ReviewerNeuralComputing.base_agent_state import BaseAgentState
from ReviewerNeuralComputing.tm_tape import TMTape

class TMAgentBrain(nn.Module):
    def __init__(self, symbol_count, embed_state_dim, hidden_dim: int = 16, hidden_layers: int = 3):
        super(TMAgentBrain, self).__init__()
        if hidden_dim%2 == 1:
            raise ValueError("Hidden dimension must be divisible by 2")

        self.symbol_count = symbol_count
        self.embed_state_dim = embed_state_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        self.read_input = nn.Linear(symbol_count + 1, int(hidden_dim/2))
        self.embed_state_input = nn.Linear(embed_state_dim, int(hidden_dim/2))

        self.mlp = nn.Sequential()
        for _ in range(hidden_layers):
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(nn.ReLU())

        self.write_head = nn.Sequential(nn.Linear(hidden_dim, symbol_count), nn.Softmax())
        self.embed_state_head = nn.Linear(hidden_dim, embed_state_dim)
        self.move_head = nn.Sequential(nn.Linear(hidden_dim, 2), nn.Softmax())
        self.halt_head = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Softmax())

    def forward(self, read_embed: torch.tensor, embed_state: torch.tensor):
        read_project = self.read_input(read_embed)
        state_project = self.embed_state_input(embed_state)

        input = torch.concat([read_project, state_project])
        hidden_embed = self.mlp(input)

        write_logits = self.write_head(hidden_embed)
        embed_state_output = self.embed_state_head(hidden_embed)
        move_logits = self.move_head(hidden_embed)
        halt_logits = self.halt_head(hidden_embed)

        return (write_logits, embed_state_output, move_logits, halt_logits) 


class TMAgent(BaseAgentState):
    def __init__(self, brain: TMAgentBrain, embed_state: torch.tensor, position: int = 0, symbol_count: int = 0):
        super().__init__()

        self.brain = brain
        self.embed_state = embed_state
        self.position = position
        self.symbol_count = symbol_count

        self.halt = False
        self.halt_output = None
        self.verbose = False


    def take_action(self, tape: TMTape) -> tuple[TMTape, Self]:
        if self.halt:
            raise RuntimeError("Terminal TM cannot take actions")
        
        read_symbol = self.symbol_count
        in_range = False
        if self.position in range(0, len(tape.tape_arr)):
            in_range = True
            read_symbol = tape.tape_arr[self.position]

        read_embed = nn.functional.one_hot(torch.tensor(read_symbol), self.symbol_count + 1).to(torch.float)

        write_logits, embed_state_output, move_logits, halt_logits = self.brain(read_embed, self.embed_state)

        write_symbol = torch.argmax(write_logits)
        move_output = torch.argmax(move_logits)
        halt_output = torch.argmax(halt_logits)

        new_position = self.position + 1 if move_output == 1 else -1

        new_tape = TMTape(tape.tape_arr.copy())
        if in_range:
            new_tape.tape_arr[self.position] = write_symbol

        new_tm = TMAgent(self.brain, embed_state_output, new_position, self.symbol_count)
        if halt_output != 0:
            new_tm.halt = True
            new_tm.halt_output = halt_output
        new_tm.verbose = self.verbose

        if self.verbose:
            tape_print_list = [f"| {el} " for el in tape.tape_arr]
            tape_print_list.append("|")

            action_str = ">" if move_output == 1 else "<"
            if halt_output != 0:
                action_str = "X" if halt_output == 1 else "☺"

            if in_range:
                tape_print_list[self.position] = f"|{read_symbol}{action_str}{write_symbol}"
            elif self.position < 0:
                tape_print_list[0] = f"{action_str} {tape.tape_arr[0]} "
            else:
                tape_print_list[-1] = action_str
                

            tape_str = "".join(tape_print_list)

            state_in_str = ",".join([f"{n: .3f}" for n in self.embed_state])
            state_out_str = ",".join([f"{n: .3f}" for n in self.embed_state])
            debug_str = f"R{read_symbol} W{write_symbol} M{move_output} H{halt_output} P{self.position} IN{state_in_str} OUT{state_out_str}"

            print(f"{tape_str}      {debug_str}")

        return (new_tape, new_tm)

    def is_terminal(self):
        return self.halt

    def terminal_output(self):
        if not self.is_terminal():
            raise RuntimeError("Cannot access terminal output in non-halted TM")
        
        return self.halt_output