from captureAgents import CaptureAgent
import distanceCalculator
import random as rn, time, util, sys
from game import Directions
import game
from util import nearestPoint

from collections import deque
import math

# Iterative Deepening Agent with defensive and aggresive strategies

agent_instances = {}

def createTeam(firstIndex, secondIndex, isRed):
    global agent_instances
    bots = [DefensiveMinimax(firstIndex), AggresiveMinimax(secondIndex)]
    agent_instances[firstIndex] = bots[0]
    agent_instances[secondIndex] = bots[1]
    return bots

class MinimaxAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.generate_map_center(gameState)

    def minimax(self, Node, gameState, curr_player_idx, depth=0, searchDepth=5, start_time=0, max_time=0.8):
        '''
        Recursive building of search tree up to depth searchDepth or until the time limit is reached
        '''

        # End recursive call
        if gameState.isOver() or depth >= searchDepth:
            return self.evaluate_state(gameState)
        
        actions = gameState.getLegalActions(curr_player_idx)

        for action in actions : # iterate over available actions
            next_gameState = gameState.generateSuccessor(curr_player_idx, action)
            child_Node = Node.expand(action, not Node.is_red)
            # recursive minimax call
            child_Node.set_value(self.minimax(child_Node, 
                                                next_gameState, 
                                                (curr_player_idx +1) % 4,
                                                depth+1,
                                                searchDepth,
                                                start_time,
                                                max_time))
            # break if time limit is reached
            if time.time() - start_time  >= max_time:
                return Node.children[max(actions, key=lambda a: greedy_score(Node, a, self.red))].value
            
        return Node.children[max(actions, key=lambda a: greedy_score(Node, a, self.red))].value

    
    def chooseAction(self, gameState, soft_time_limit=0.9):
        '''
        Runs iterative deepening loop until the time limit is reached
        '''
        start_time = time.time()
        
        self.update_turn_state(gameState)
        
        # Generate search tree
        minimax_tree = Node(self.red)
        
        self.agent_POIs = [PositionalPOI(self.map_center) if gameState.isOnRedTeam(i) != self.red else agent_instances[i].get_POI() for i in range(4)]
        self.optimal_positions = self.get_agent_positions(gameState, 0)
        actions = gameState.getLegalActions(self.index)

        # starts ID with a maxDepth of 5
        maxDepth = 5
        while time.time() - start_time < soft_time_limit:
            self.minimax(minimax_tree, gameState, self.index, 0, maxDepth, start_time, soft_time_limit)
            maxDepth += 1

        #print(f'Deepened until {maxDepth}, took {time.time() - start_time} seconds.')
        return max(actions, key=lambda a: greedy_score(minimax_tree, a, self.red))  
    
    
    def get_agent_positions(self, state, t):
        return {(i, state.data.agentStates[i].getPosition()): t for i in range(4)}

    def generate_map_center(self, state):
        self.map_center = (state.data.layout.width // 2, state.data.layout.height // 2)
        try:
            self.getMazeDistance(self.start, self.map_center)
            not_found_valid_position = False
        except:
            not_found_valid_position = True
        d = 1
        while not_found_valid_position:
            self.map_center = (state.data.layout.width // 2 + rn.randint(-d, d), state.data.layout.height // 2 + rn.randint(-d, d))
            try:
                self.getMazeDistance(self.start, self.map_center)
                not_found_valid_position = False
            except:
                not_found_valid_position = True
            d += 1

    def update_turn_state(self, gameState):
        raise NotImplementedError()

    def get_POI(self):
        raise NotImplementedError()

    def evaluate_state(self, gameState):
        # Score is positive for red and negative for blue
        score = 10 * gameState.getScore()
        # General heuristic definition thats negative for blue agents
        heuristic = lambda foo, coeff: sum([coeff * (1 if gameState.isOnRedTeam(i) else -1) * foo(gameState, i) for i in range(4)])

        dist =  heuristic(lambda state, i: -self.getMazeDistance(state.getAgentPosition(i), self.agent_POIs[i].get_position(gameState)), 1/40)
        
        carrying = heuristic(lambda state, i: state.getAgentState(i).numCarrying, 1)

        return score + dist + carrying


class AggresiveMinimax(MinimaxAgent):
    hover_time = 5

    def  __init__(self, index, timeForComputing=0.1):
        super().__init__(index, timeForComputing)
        self.food_target = None
        self.hover_delta = AggresiveMinimax.hover_time
    
    def update_turn_state(self, gameState):
        self_state = gameState.getAgentState(self.index)
        if self.food_target is None and self_state.numCarrying == 0:
            self.food_target = rn.choice(self.getFood(gameState).asList())
            return
        
        if self.food_target is not None and self_state.numCarrying != 0:
            if self.hover_delta == 0:
                self.food_target = None
                return
            self.hover_delta -= 1

    def get_POI(self):
        if self.food_target is not None:
            return PositionalPOI(self.food_target)
        return PositionalPOI(self.map_center)


class DefensiveMinimax(MinimaxAgent):
    def  __init__(self, index, timeForComputing=0.1):
        super().__init__(index, timeForComputing)
        self.target_id = None

    def update_turn_state(self, gameState):
        enemyids = gameState.getBlueTeamIndices() if self.red else gameState.getRedTeamIndices()
        enemyids = [id for id in enemyids if gameState.getAgentState(id).isPacman]
        if not enemyids:
            self.target_id = None
            return
        self.target_id = min(enemyids, key=lambda id: self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(id)))

    def get_POI(self):
        if self.target_id is None:
            return PositionalPOI(self.map_center)
        return AgentPOI(self.target_id)


class Node:
    def __init__(s, is_red, parent=None):
        s.is_red = is_red
        s.parent = parent
        s.diff_score = 0
        s.sims = 0
        s.children = {}
        s.value = 0
    
    def __getitem__(s, key):
        if key not in s.children:
            return None
        return s.children[key]
    
    def expand(s, a, is_red):
        try: 
            node = s.children[a]
        except:
            node = Node(is_red, s)
            s.children[a] = node
        return node

    def set_value(s, val):
        s.value = val


class POI:
    def get_position(self, gameState):
        raise NotImplementedError()

class PositionalPOI(POI):
    def __init__(self, position):
        self.position = position
        
    def get_position(self, gameState):
        return self.position

class AgentPOI(POI):
    def __init__(self, id):
        self.id = id
        
    def get_position(self, gameState):
        return gameState.getAgentPosition(self.id)


def greedy_score(parent, a, is_red):
    if a not in parent.children:
        return -float('inf')
    return parent[a].value if is_red else -parent[a].value

def epsilon_choice(actions, epsilon, node, is_red):
    if rn.random() <= epsilon:
        return rn.choice(actions)
    return max(actions, key=lambda a: greedy_score(node, a, is_red))

def get_tree_string(tree):
    queue = deque()
    queue.append((0, 'none', None, tree))
    curr_level = queue[0][0]
    res = []
    while queue:
        new_level, action, parent, node = queue.popleft()
        if new_level != curr_level:
            res.append('\n')
            curr_level = new_level
        else:
            res.append('    ')
        res.append(f'{action}: score: {node.diff_score}, abs score: {node.abs_score}, UCT {UCT(parent, action, node.is_red)}')
        
        for a in node.children:
            queue.append((curr_level + 1, a, node, node[a]))
    
    return ''.join(res)
