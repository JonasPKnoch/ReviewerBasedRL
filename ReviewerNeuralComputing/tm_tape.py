import numpy as np

from base_world_state import BaseWorldState

class TMTape(BaseWorldState):
    def __init__(self, tape_arr: np.ndarray, symbol_count: int = 2):
        super().__init__()

        self.tape_arr = tape_arr
        self.symbol_count = symbol_count