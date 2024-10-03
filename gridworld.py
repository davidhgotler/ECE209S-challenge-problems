import numpy as np

label_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

class gridworld:
    def __init__(self,size:tuple[int],agent_start:tuple[int],obst_loc:list[tuple[int]],icecream_loc:list[tuple[int]],hazard_loc:list[tuple[int]]) -> None:
        '''
        `size` = size of the grid-world in X,Y
        `agent_start` = (x,y) coordinate for the starting position of the agent
        `obst_loc` = list of (x,y) coordinates for obstacles
        `icecream_loc` = list of (x,y) coordinates for ice cream shops
        `hazard_loc` = list of (x,y) coordinates for hazards
        '''
        if len(size) == 2:
            self.x_max = size[0] - 1
            self.y_max = size[1] - 1
            self.size = size[0]*size[1]
        else:
            raise Exception("size should be of lenth 2")
        self.make_grid()
        if len(agent_start) == 2:
            if (agent_start[0] <= self.x_max and agent_start[0] >= 0) and (agent_start[1] <= self.y_max and agent_start[1] >= 0):
                pass
                # initialize agent/state class
    
    def make_grid(self):
        x = list(range(0,self.x_max + 1))
        y = list(range(0,self.y_max + 1))
        
        states = {}
        for i,label in enumerate(label_str[0:self.size]):
            states[label] = (i%self.x_max, -(-i//self.y_max)) # (i mod x_max, ceil(i/y_max))

        self.states = states

    def get_state_from_coords(self,coord):
        for key,val in self.states.items():
            if val == coord:
                return key