from dataclasses import dataclass
from random import Random
from datetime import datetime
from copy import copy,deepcopy
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
class obstacle(state):
    pass

@dataclass
class reward(state):
    reward:int

@dataclass
class destination(reward):
    destination_name:str = 'ice cream shop'

@dataclass
class hazard(reward):
    pass

class agent:
    def __init__(self,s_start:state,history:list[state]=[]) -> None:
        self.state = s_start
        self.history = history

    def save_state(self,s_new):
        self.history.append(self.state)
        self.state = s_new

class gridworld:
    def __init__(self,size:tuple[int],n_agensts:int,agents_start:list[np.ndarray],destinations:list[tuple[int]],obstacles:list[tuple[int]],hazards:list[tuple[int]],p_e = 0.2,dest_rewards=None,haz_rewards=None,dest_names='ice cream shop',gamma = 0.8,epsilon = 0.01) -> None:
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
        self.n_agents = n_agensts
        
        if len(destinations) > 0:
            if type(dest_rewards) is list:
                if len(dest_rewards)!= len(destinations):
                    if len(dest_rewards)!=1:
                        raise Exception("destination rewards should be given as one integer value, or a list of same size as destinations list.")
            if dest_rewards is None:
                print('No destination reward value given, defaulting to 1')
                dest_rewards = 1
        
        if len(hazards) > 0:
            if type(haz_rewards) is list:
                if len(haz_rewards)!= len(hazards):
                    if len(haz_rewards)!=1:
                        raise Exception("hazard rewards should be given as one integer value, or a list of same size as hazards list.")
            if haz_rewards is None:
                print('No hazard reward value given, defaulting to -1')
                haz_rewards = -1

        # Initialize/calc grid states
        self.get_states(destinations,obstacles,hazards,dest_rewards,haz_rewards,dest_names)

        if len(agent_start) == 2:
            # Check start coord to be in bounds
            s_start = self.coord2state(agent_start)
            if s_start and s_start is not obstacle:
                self.agent = agent(s_start)
            else:
                raise Exception("could not find valid state from agent_start")
        else:
            raise Exception("agent_start should be integer valued tuple of length 2")
        
        self.p_e = p_e # prob of error
        self.gamma = gamma # discount factor   
        self.epsilon = epsilon # convergence threshold
         
        # transition probabilities
        self.calc_p_matrix()
        self.calc_o_matrix()
        self.calc_r_matrix()
        self.value_iteration()
        self.extract_policy()
        self.update_o()

    def get_states(self,destinations,obstacles,hazards,dest_rewards,haz_rewards,dest_names):
        grid_states = []
        state_labels = []
        state_coords = []
        for i,label in enumerate(label_str[0:self.num_states]):
            x = i%self.x_max
            y = i//self.x_max
            state_labels.append(label)
            state_coords.append(np.array((x,y)))
            if (x,y) in destinations:
                d = destinations.index((x,y))
                if type(dest_names) is list:
                    if len(dest_names)>1:
                        dest_name = dest_names[d]
                    else:
                        dest_name = dest_names[0]
                else:
                    dest_name = dest_names

                if type(dest_rewards) is list:
                    if len(dest_rewards) > 1:   
                        dest_reward = dest_rewards[d]
                    else:
                        dest_reward = dest_rewards[0]
                else:
                    dest_reward =  dest_rewards

                grid_states.append(destination(label,np.array((x,y)),dest_reward,dest_name))
            elif (x,y) in obstacles:
                grid_states.append(obstacle(label,np.array((x,y))))
            elif (x,y) in hazards:
                h = hazards.index((x,y))
                if type(haz_rewards) is list:
                    if len(haz_rewards) > 1:
                        haz_reward = haz_rewards[h]
                    else:
                        haz_rewards = haz_rewards[0]
                else:
                    haz_reward = haz_rewards

                grid_states.append(hazard(label,np.array((x,y)),haz_reward))
            else:
                grid_states.append(state(label,np.array((x,y))))
        
        # Compute combinations of all agent states
        indices = []
        for k in range(self.n_agents,0,-1):
            indices.insert(0,[[i]*(self.n_agents**k) for i in range(self.num_states)])
        print(indices)
        i0 = indices[0]
        states = [[grid_states[i]] for i in i0]
        for ii in indices[1:]:
            for i in i0:
                for j in ii:
                    states[i].append([grid_states[j]])
        print(states)

        self.state_labels = state_labels
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
                else:
                    continue #skip obstacle states bc prob 0 of being in state
    
    def calc_r_matrix(self):
        self.r_matrix = np.zeros((len(actions),self.num_states,self.num_states))
        for i,a_t in enumerate(actions):
            for j,s_t in enumerate(self.states):
                s_calc = self.coord2state(s_t.coord + a_t)
                for k,s_tp1 in enumerate(self.states):
                    if type(s_tp1) is destination:
                        # Check intended direction
                        if s_calc == s_tp1:
                            self.r_matrix[i,j,k] = s_tp1.reward
                        else:
                            self.r_matrix[i,j,k] = 0
                    elif type(s_tp1) is hazard:
                        if self.p_matrix[i,j,k] > 0:
                            self.r_matrix[i,j,k] = s_tp1.reward
                        else:
                            self.r_matrix[i,j,k] = 0
                    else:
                        self.r_matrix[i,j,k] = 0

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
            h_arr = d_arr[0]
            # original code: h_arr = d_arr
            # reported error when create world with one destination
            # d_arr was originally [[...]], should be [...]

        #h_arr is a 1D matrix with size as the number of states
        h_up_arr = np.ceil(h_arr).astype(int)
        h_dn_arr = np.floor(h_arr).astype(int)
        o_max = np.max(h_up_arr)
        self.o_vals = np.arange(o_max+1)

        # calc prob(ceil(h),floor(h)|s_t)
        p_up_arr = 1 - (h_up_arr - h_arr)
        p_dn_arr = 1 - (h_arr - h_dn_arr)

        #p_o_arr is the probability matrix for each output value with dimensions (no. of possible o values, no. of states)
        self.p_o_arr = np.zeros((o_max+1,self.num_states))
        for o in self.o_vals:
            for s,(h_up,h_dn,p_up,p_dn) in enumerate(zip(h_up_arr,h_dn_arr,p_up_arr,p_dn_arr)):
                if h_up == o and h_dn == o:
                    self.p_o_arr[o,s] += (p_up + p_dn)/2
                elif h_up == o:
                    self.p_o_arr[o,s] += p_up
                elif h_dn == o:
                    self.p_o_arr[o,s] += p_dn
    
    def update_o(self):
        # probabilistically update o given current state based on output proabilities
        RNG = Random()
        RNG.seed(datetime.timestamp(datetime.now()))
        s = self.states.index(self.agent.state)
        self.o = RNG.choices(self.o_vals,weights=self.p_o_arr[:,s])[0]
    
    def update_state(self,action):
        # probabilistically update state given an input, based on transition probabilities
        RNG = Random()
        RNG.seed(datetime.timestamp(datetime.now()))
        # Get index of current state and action
        s = self.states.index(self.agent.state)
        a = np.where(np.all(actions==action,axis=1))[0][0]
        # Make probabilistic choice of states given the transition probabilities
        new_state = RNG.choices(self.states,weights=self.p_matrix[a,s,:])[0]
        # Save new state
        self.agent.save_state(new_state)
        # Save new sample of o at new state
        self.update_o()
        
    def value_iteration(self):
        self.calc_p_matrix()
        self.calc_r_matrix()
        # states = np.arange(self.num_states)
        # Initialize value function
        value_map = np.zeros((len(actions),self.num_states))
        action_indices = np.arange(len(actions))
        self.value_map = np.zeros(self.num_states)
        delta = 9999
        while delta > self.epsilon:
            delta = 0
            # Save current optimal
            v = self.value_map.copy()
            # Compute value function for all actions
            for s in range(self.num_states):
                for a in range(len(actions)):
                    # if type(self.states[s]) is not obstacle:
                    value_map[a,s] = sum(self.p_matrix[a, s, s_next] * (self.r_matrix[a, s, s_next] + self.gamma * value_map[a,s_next]) for s_next in range(self.num_states)) # value function
            # get optimal value function
            self.value_map = np.max(value_map,axis=0)
            self.policy_map = np.argmax(value_map,axis=0)
            delta = np.max(np.abs(self.value_map - v)) # amount in which value function has changed


    def extract_policy(self):
        self.value_iteration()
        policy = np.zeros(self.num_states,dtype=int)
        for s in range(self.num_states):
            # find the max a in actions that maximize the function given by lambda
            # if type(self.states[s]) is not obstacle:
            policy[s] = max(np.arange(len(actions)), key=lambda a: sum(self.p_matrix[a, s, s_next] * (self.r_matrix[a, s, s_next] + self.gamma * self.value_map[s_next]) for s_next in (range(self.num_states))))
        self.policy_map = policy
    
    def print_policy_map(self):
        policy_sign_map = np.zeros((self.x_max,self.y_max),dtype = str)
        action_signs = ['o','→','←','↑','↓']
        indices = np.zeros((self.x_max,self.y_max))
        for i in range(self.x_max):
            for j in range(self.y_max):
                policy_sign_map[i][self.y_max - 1 - j] = action_signs[self.policy_map[i+j*self.x_max]]
                indices[i,self.y_max - 1 - j] = i+j*self.x_max
        print(policy_sign_map.T)
    
    def check_if_arrive_at_dest(self):
        # if arrived, change the rewards of the destination to 0
        if type(self.agent.state) is destination:
            self.agent.state.reward = 0
            self.calc_r_matrix()
            self.value_iteration()
            self.extract_policy()
            return True
        else:
            return False
        


