from typing import Self
from base_world_state import BaseWorldState

class BaseAgentState:
    def take_action(self, world_state: BaseWorldState) -> tuple[BaseWorldState, Self]:
        pass

    def is_terminal(self) -> bool:
        pass

    def terminal_output(self) -> float:
        pass
