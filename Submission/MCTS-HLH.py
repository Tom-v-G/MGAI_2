from captureAgents import CaptureAgent
import distanceCalculator
import random as rn, time, util, sys
from game import Directions
import game
from util import nearestPoint

from collections import deque
import math

# MCTS agent with reward model, Qvalue dampening factor gamma and two MCTS strategies, competitive and single
# plus aggresive / defensive strategies for individuals

agent_instances = {}

def createTeam(firstIndex, secondIndex, isRed):
    global agent_instances
    bots = [AggresiveMCTSAgent(firstIndex), DefensiveMCTSAgent(secondIndex)]
    agent_instances[firstIndex] = bots[0]
    agent_instances[secondIndex] = bots[1]
    return bots

class MCTSAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.generate_map_center(gameState)

    def chooseAction(self, gameState, searchDepth=40, soft_time_limit=0.95, gamma=0.9):

        start_time = time.time()
        
        self.update_turn_state(gameState)
        
        mc_trees = [Node(self.red), Node(self.red)]
        def update_curr_player(single_player):
            nonlocal curr_player_index
            if single_player:
                curr_player_index = curr_player_index
            else:
                curr_player_index = (curr_player_index + 1) % 4
        get_player_red_team = lambda: curr_player_index in var_gameState.redTeam
        
        self.agent_POIs = [PositionalPOI(self.map_center) if gameState.isOnRedTeam(i) != self.red else agent_instances[i].get_POI() for i in range(4)]
        n_runs = 0
        mode = 0
        playouts = [[], []]
        self.optimal_positions = self.get_agent_positions(gameState, 0)
        while time.time() - start_time < soft_time_limit:
            curr_player_index = self.index
            var_gameState = gameState.deepCopy()
            prev_state_score = self.evaluate_state(var_gameState)
            curr_node = mc_trees[mode]
            is_on_tree = True
            rewards = deque([0])
            gamma_acc = 1
            for t in range(1, searchDepth + 1 if not mode else searchDepth // 4 + 1):
                was_on_tree = is_on_tree
                actions = var_gameState.getLegalActions(curr_player_index)
                diff, actions = self.cull_subopt_positions(var_gameState, t, curr_player_index, actions)
                if is_on_tree:
                    a = rn.choice(actions)
                    prev_node = curr_node
                    curr_node = prev_node[a]
                    if curr_node is None:
                        curr_node = prev_node.expand(a, get_player_red_team())
                        is_on_tree = False
                else:
                    a = rn.choice(actions)
                
                var_gameState = var_gameState.generateSuccessor(curr_player_index, a)
                self.update_optimal_positions(var_gameState, t)
                curr_state_score = self.evaluate_state(var_gameState)
                if was_on_tree:
                    rewards.append(curr_state_score - prev_state_score)
                else:
                    gamma_acc *= gamma
                    rewards[-1] += gamma_acc * (curr_state_score - prev_state_score)
                update_curr_player(mode)
                prev_state_score = curr_state_score
                
                if var_gameState.isOver():
                    break
            playouts[mode].append(curr_node.backprop(rewards, 0, gamma))
            n_runs += 1
            mode = (mode + 1) % 2

        actions = gameState.getLegalActions(self.index)
        stds = [get_std(playouts[0]), get_std(playouts[1])]
        max_diffs = [get_max_diff(mc_trees[0], actions), get_max_diff(mc_trees[1], actions)]
        
        if safe_div(max_diffs[1], stds[1]) >= 2 * safe_div(max_diffs[0], stds[0]):
            chosen_mode = 1 # Individual play
        else:
            chosen_mode = 0 # Competitive play
        
        return max(actions, key=lambda a: greedy_score(mc_trees[chosen_mode], a, self.red))
    
    def get_agent_positions(self, state, t):
        return {(i, state.data.agentStates[i].getPosition()): t for i in range(4)}
    
    def cull_subopt_positions(self, state, t, curr_player_index, actions):
        diffs = [self.get_position_delay(state.generateSuccessor(curr_player_index, a), t) for a in actions]
        min_diff = min(diffs)
        res = [a for a, diff in zip(actions, diffs) if diff == min_diff]
        return min_diff, res
    
    def get_position_delay(self, state, t):
        curr_positions = self.get_agent_positions(state, t)
        _sum = 0
        for key in curr_positions:
            if key in self.optimal_positions:
                _sum += max(0, curr_positions[key] - self.optimal_positions[key])
        return _sum
    
    def update_optimal_positions(self, state, t):
        curr_positions = self.get_agent_positions(state, t)
        for key in curr_positions:
            if key in self.optimal_positions:
                self.optimal_positions[key] = min(self.optimal_positions[key], curr_positions[key])
            else:
                self.optimal_positions[key] = curr_positions[key]

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
        
        scared = heuristic(lambda state, i: state.getAgentState(i).scaredTimer, -1)

        return score + dist + carrying


class AggresiveMCTSAgent(MCTSAgent):
    hover_time = 5

    def  __init__(self, index, timeForComputing=0.1):
        super().__init__(index, timeForComputing)
        self.food_target = None
        self.hover_delta = AggresiveMCTSAgent.hover_time
    
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


class DefensiveMCTSAgent(MCTSAgent):
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
    
    def __getitem__(s, key):
        if key not in s.children:
            return None
        return s.children[key]
    
    def expand(s, a, is_red):
        node = Node(is_red, s)
        s.children[a] = node
        return node
    
    def backprop(s, rewards, acc, gamma):
        reward = rewards.pop() + acc
        s.diff_score += reward
        s.sims += 1
        if s.parent:
            return s.parent.backprop(rewards, gamma * reward, gamma)
        return reward


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

def get_std(l):
    _mean = sum(l) / len(l)
    return math.sqrt(sum([(e - _mean) ** 2 for e in l]) / (len(l) - 1))

def get_max_diff(mc_tree, actions):
    scores = [greedy_score(mc_tree, a, 0) for a in actions if a in mc_tree.children]
    return max(scores) - min(scores)

def safe_div(a, b):
    if b != 0:
        return a / b
    if a == 0:
        return 0
    return (1 if a > 0 else -1) * float('inf')

def UCT(parent, a, is_red, c=1.414):
    if parent is not None:
        N = parent.sims + 1
    else:
        return 'N/A'
    
    if parent[a] is None:
        return c * math.sqrt(math.log(N) / 1)
    
    node = parent[a]
    w = node.diff_score if is_red else -node.diff_score
    n = node.sims + 1
    
    return w / n + c * math.sqrt(math.log(N) / n)

def greedy_score(parent, a, is_red):
    if a not in parent.children:
        return -float('inf')
    return parent[a].diff_score if is_red else -parent[a].diff_score

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
