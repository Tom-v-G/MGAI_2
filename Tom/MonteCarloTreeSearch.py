from captureAgents import CaptureAgent
import random, time, util, math
from game import Directions
from capture import GameState
import game

class MDP():
    def __init__(self, GameState, gamma=1):
        self.GameState = GameState
        self.gamma = gamma

    def get_actions(self, agentIndex):
        return self.GameState.getLegalActions(agentIndex)
    
    def current_state():
        raise NotImplementedError
    
    def apply(self, action):
        next_state = self.GameState.generateSuccesor(agentIndex, action)
        reward = 0 #needs to be calculated with heuristics
        return (next_state, reward)
    
    def is_leaf(self, state):
        raise NotImplementedError
    
    def take_action(self, action):
        raise NotImplementedError

class MCTS():
    def __init__(self, MDP, Q_function, strategy=None, root=None) -> None:
        self.MDP = MDP
        self.Q_function = Q_function
        self.strategy = strategy
        self.root = root

    def create_root(self, state):
        return TreeNode(self.MDP, parent_node=None, state=self.MDP.current_state())

    def choose(self, state):
        if self.strategy is None: #if no strategy is provided simulation uses random actions
            return random.choice(self.MDP.get_actions(state))
        return self.strategy(state)

    def simulate(self, node, max_depth):
        '''
        Simulates a node until leaf state or max_depth has been reached. Returns the reward.
        '''

        state = node.state
        sum_reward = 0
        depth = 0

        while (not self.MDP.is_leaf(state)) and (depth < max_depth):
            action = self.choose(state)
            (next_state, reward) = self.MDP.apply(state, action)
            sum_reward += pow(self.MDP.gamma, depth) * reward
            depth +=1 

            state = next_state

        return sum_reward

    
    def MCTS(self, agentIndex, max_time=1):
        
        # Initialise root
        if self.root is None:
            self.root = self.create_root()

        start_time, current_time = time.time(), time.time()
        
        while current_time < start_time + max_time:
            
            current_node = self.root.select()
            if not self.MDP.is_leaf(current_node):
                child_node = current_node.expand()
                value = self.simulate(child_node)
                child_node.backpropagate(value)
            current_time = time.time()
        
        return self.root

class TreeNode():
  def __init__(self, MDP, parent_node, state, reward=0, action=None) -> None:
    self.MDP = MDP
    self.parent = parent_node
    self.state = state
    self.timesvisited = 0
    self.children = {} # dictionary of actions to nodes
    self.expanded = False
    self.reward = reward
    self.action = action 

    def is_expanded(self):
        '''
        A node is fully expanded when all child nodes for each possible action in a state have been made.
        '''
        if len(self.MDP.get_actions(self.state)) == len(self.children):
            return True
        return False

    def select(self, C=1):
        '''
        Selects a node that is not fully expanded using UCT algorithm
        '''
        if self.MDP.is_leaf(self) or not self.is_expanded() :
            return self
        
        uct_list = []
        max_uct_val = 0
        for child_node in self.children.values(): #loop through nodes
            uct = child_node.reward + (C * math.sqrt( math.log(self.timesvisited) / child_node.timesvisited))
            if uct > max_uct_val:
                max_uct_val = uct
                uct_list = [child_node]
            elif uct == max_uct_val:
                uct_list.append(child_node)
        best_child = random.choice(uct_list)
        return best_child.select()
            
    def expand(self):
        '''
        expands the current node
        '''
        if self.MDP.is_leaf(self.state):
            return self
        # select and action that has not been taken yet
        possible_actions = self.MDP.get_actions(self.state) - self.children.keys()
        action = random.choice(list(possible_actions))
        
        next_state = self.MDP.take_action(action)
        self.children[action] = TreeNode(self.MDP, self, next_state, action=action)
        return self.children[action]

    
    def backpropagate(self, value):
        current_node = self
        while current_node is not None:
            current_node.timesvisited += 1
            current_node.reward = (current_node.reward + value) / (current_node.reward + 1)
            current_node = self.parent
        
    


            