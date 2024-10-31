import numpy as np
from copy import copy

class lqr:
    def __init__(self,A,B,Q,R) -> None:
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.P = copy(self.Q)

    def solver(self):
        pass
        # calc K
        # calc P