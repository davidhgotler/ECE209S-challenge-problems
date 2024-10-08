from dataclasses import dataclass
import sys
import numpy as np

label_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Direction vectors for actions
STAY = np.array((0,0))
RIGHT = np.array((1,0))
LEFT = np.array((-1,0))
UP = np.array((0,1))
DOWN = np.array((0,-1))
actions = [STAY,RIGHT,LEFT,UP,DOWN]

@dataclass
class state:
    label:str
    coord:np.ndarray

@dataclass
class destination(state):
    destination_type:str = 'ice cream shop'

@dataclass
class obstacle(state):
    pass
        
@dataclass
class hazard(state):
    pass

class agent:
    def __init__(self,s:state,history:list[state]=[]) -> None:
        self.s = s
        self.history = history

    def update_state(self,s_new):
        self.history.append(self.s)
        self.s = s_new


class gridworld:
    def __init__(self,size:tuple[int],agent_start:np.ndarray,destinations:list[tuple[int]],obstacles:list[tuple[int]],hazards:list[tuple[int]],p_e = 0.2,destination_type='ice cream shop') -> None:
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
        # Initialize/calc grid states
        self.make_grid(destinations,obstacles,hazards,destination_type)
        self.p_e = p_e # prob of error
        # transition probabilities
        self.calc_p_matrix()
        self.calc_o_matrix()

        if len(agent_start) == 2:
            # Check start coord to be in bounds
            s_start = self.coord2state(agent_start)
            if s_start and s_start is not obstacle:
                self.agent = agent(s_start)
            else:
                raise Exception("could not find valid state from agent_start")
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
            self.state_coords.append(np.array((x,y)))
            if (x,y) in destinations:
                self.states.append(destination(label,np.array((x,y)),destination_type))
            elif (x,y) in obstacles:
                self.states.append(obstacle(label,np.array((x,y))))
            elif (x,y) in hazards:
                self.states.append(hazard(label,np.array((x,y))))
            else:
                self.states.append(state(label,np.array((x,y))))
        self.state_coords = np.array(self.state_coords)

    def coord2state(self,coord):
        for state in self.states:
            if np.all(coord==state.coord):
                return state
        return
        
    def label2state(self,label):
        for state in self.states:
            if label == state.label:
                return state
        return
    
    # def update_action_set(self):
    #     self.action_set = []
    #     for action in actions:
    #         s_new = self.coord2state((self.state.x+action[0],self.state.y+action[1]))
    #         if s_new is obstacle:
    #             continue
    #         if s_new in self.states:
    #             self.action.append(action)

   def calc_p_matrix(self):
        self.p_matrix = np.zeros((len(actions),self.num_states,self.num_states))
        for i,a_t in enumerate(actions):
            for j,s_t in enumerate(self.states):
              #print('s_t: ', s_t)
              if type(s_t) is not obstacle:
                #print('not obstacle')
                s_calc = self.coord2state(s_t.coord + a_t)
                if not s_calc or (type(s_calc) is obstacle):
                    self.p_matrix[i,j,j] += 1 - self.p_e - self.p_e/(len(actions)-1)
                    self.p_matrix[:,j,j] += self.p_e/(len(actions)-1)
                for k,s_tp1 in enumerate(self.states):
                    if type(s_tp1) is not obstacle:
                    # limit motion to one action away
                      if np.any(np.all((s_tp1.coord - s_t.coord)==actions,axis=1)):
                          # intended direction
                            if s_calc == s_tp1:
                                self.p_matrix[i,j,k] += 1 - self.p_e
                            else:
                                self.p_matrix[i,j,k] += self.p_e/(len(actions)-1)

                      # states that are more than 1 action away
                      else:
                          self.p_matrix[i,j,k] += 0 # redundant
    
    def calc_o_matrix(self):
        # calc d for each destination at each state
        dest_coords = np.array([d.coord for d in self.states if type(d) is destination])
        d_arr = np.array([np.linalg.norm(d-self.state_coords,axis=1) for d in dest_coords])
        # returns an array of dimension (No. of destinations, No. of states)
        # calc h, ceil(h), floor(h) at each state
        # h = 2/(sum: 1/d_i) = 2 * product: d_i / sum: d_i
        if d_arr.shape[0]>1:
            h_arr = 2*np.multiply.reduce(d_arr,axis=0)/np.sum(d_arr,axis=0)
        else:
            h_arr = d_arr

        #h_arr is a 1D matrix with size as the number of states
        h_up_arr = np.ceil(h_arr).astype(int)
        h_dn_arr = np.floor(h_arr).astype(int)
        h_max = np.max(h_up_arr)
        h_vals = np.arange(h_max+1)

        # calc prob(ceil(h),floor(h)|s_t)
        p_up_arr = 1 - (h_up_arr - h_arr)
        p_dn_arr = 1 - (h_arr - h_dn_arr)
        p_h_arr = np.zeros((h_max+1,self.num_states))
        for h_i in h_vals:
            for s,(h_up,h_dn,p_up,p_dn) in enumerate(zip(h_up_arr,h_dn_arr,p_up_arr,p_dn_arr)):
                if h_up == h_i and h_dn == h_i:
                    p_h_arr[h_i,s] += (p_up + p_dn)/2
                elif h_up == h_i:
                    p_h_arr[h_i,s] += p_up
                elif h_dn == h_i:
                    p_h_arr[h_i,s] += p_dn
        #p_h_arr is the probability matrix for each output value with dimensions (no. of possible h values, no. of states)
        # calc prob(h_i|s_t,a_t) = P_h,s_t,a_t = P_h,s_t+1 . P_s_t+1,s_t,a_t
        self.p_o_matrix = np.dot(p_h_arr,self.p_matrix.swapaxes(1,2))

