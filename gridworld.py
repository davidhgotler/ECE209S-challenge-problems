from dataclasses import dataclass
import numpy as np

label_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

class gridworld:
    def __init__(self,size:tuple[int],agent_start:tuple[int],destinations:list[tuple[int]],obstacles:list[tuple[int]],hazards:list[tuple[int]],destination_type='ice cream shop') -> None:
        '''
        `size` = size of the grid-world in X,Y\\
        `agent_start` = (x,y) coordinate for the starting position of the agent\\
        `destinations` = list of (x,y) coordinates for destinations of type `destination_type`\\
        `obstacles` = list of (x,y) coordinates for obstacles\\
        `hazards` = list of (x,y) coordinates for hazards\\
        `destination_type` = str or int label for type of destination, ex: "ice cream shop"
        '''
        if len(size) == 2:
            self.x_max = size[0]
            self.y_max = size[1]
            self.num_states = size[0]*size[1]
        else:
            raise Exception("size should be integer valued tuple lenth 2")
        self.make_grid(destinations,obstacles,hazards,destination_type)
        if len(agent_start) == 2:
            # Check start coord to be in bounds
            if agent_start in self.state_coords:
                self.agent = agent(self.coord2state(agent_start))

        else:
            raise Exception("agent_start should be integer valued tuple of length 2")
    
    def make_grid(self,destinations,obstacles,hazards,destination_type):
        self.states = []
        self.state_labels = []
        self.state_coords = []
        for i,label in enumerate(label_str[0:self.num_states]):
            x = i%self.x_max
            y = i//self.x_max
            self.state_labels.append(label)
            self.state_coords.append((x,y))
            if (x,y) in destinations:
                self.states.append(destination(label,x,y,destination_type))
            elif (x,y) in obstacles:
                self.states.append(obstacle(label,x,y))
            elif (x,y) in hazards:
                self.states.append(hazard(label,x,y))
            else:
                self.states.append(state(label,x,y))

    def coord2state(self,coord):
        for state in self.states:
            if coord == (state.x,state.y):
                return state
    def label2state(self,label):
        for state in self.states:
            if label == state.label:
                return state

@dataclass
class state:
    label:str
    x:int
    y:int

@dataclass
class destination(state):
    destination_type:str = 'ice cream shop'

class obstacle(state):
    def __init__(self,label,x,y):
        super().__init__(label,x,y)
        
class hazard(state):
    def __init__(self,label,x,y):
        super().__init__(label,x,y)

class agent:
    def __init__(self,s:state,history:list[state]=[]) -> None:
        self.s = s
        self.history = history

        # Direction vectors for actions
        self.RIGHT = (1,0)
        self.UP = (0,1)
        self.LEFT = (-1,0)
        self.DOWN = (0,-1)
        self.STAY = (0,0)

    def update_state(self,s_new):
        self.history.append(self.s)
        self.s = s_new

