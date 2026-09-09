from typing import Optional
from base_agent_state import BaseAgentState
from base_world_state import BaseWorldState

class WorldModelRunner:
    def __init__(self, initial_world: BaseWorldState, initial_agent: BaseAgentState, timeout=100):
        self.current_world = initial_world
        self.current_agent = initial_agent
        self.is_terminal = False
        self.terminal_value = 0.0
        self.timeout = timeout

        self.world_states = [initial_world]
        self.agent_states = [initial_agent]

    def take_action(self) -> tuple[BaseWorldState, BaseAgentState]:
        if self.is_terminal:
            raise RuntimeError("Agent/Runner is terminal and cannot take actions")

        next_world, next_agent = self.current_agent.take_action(self.current_world)
        self.world_states.append(next_world)
        self.agent_states.append(next_agent)
        self.current_world = next_world
        self.current_agent = next_agent

        if next_agent.is_terminal():
            self.is_terminal = True
            self.terminal_value = next_agent.terminal_output()

        if len(self.world_states) >= self.timeout:
            self.is_terminal = True
            self.terminal_value = -1.0

        return (next_world, next_agent)

    def rollout(self) -> float:
        while not self.is_terminal:
            self.take_action()

        return self.terminal_value

        