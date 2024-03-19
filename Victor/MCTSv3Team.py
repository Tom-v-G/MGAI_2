from captureAgents import CaptureAgent
import distanceCalculator
import random as rn, time, util, sys
from game import Directions
import game
from util import nearestPoint

from collections import deque
import math

# MCTS agent which does not visit positions which have already been visited at a previous timestep in any other playout

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MCTSAgent', second = 'MCTSAgent'):

    return [eval(first)(firstIndex), eval(second)(secondIndex)]

class MCTSAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.generate_map_center(gameState)

    def chooseAction(self, gameState, searchDepth=40, soft_time_limit=0.55, ignore_other_players=False):

        start_time = time.time()
        
        mc_tree = Node(self.red)
        def update_curr_player():
            # unclear what the actual rule is, but not very relevant for the simulation
            nonlocal curr_player_index
            if ignore_other_players:
                curr_player_index = curr_player_index
            else:
                curr_player_index = (curr_player_index + 1) % 4
        get_player_red_team = lambda: curr_player_index in var_gameState.redTeam
        
        n_runs = 0
        self.optimal_positions = self.get_agent_positions(gameState, 0)
        while time.time() - start_time < soft_time_limit:
            curr_player_index = self.index
            var_gameState = gameState.deepCopy()
            curr_node = mc_tree
            is_on_tree = True
            for t in range(1, searchDepth + 1):
                actions = var_gameState.getLegalActions(curr_player_index)
                diff, actions = self.cull_subopt_positions(var_gameState, t, curr_player_index, actions)
                if is_on_tree:
                    a = max(actions, key=lambda a: UCT(curr_node, a, get_player_red_team()))
                    # a = rn.choice(var_gameState.getLegalActions(curr_player_index))
                    prev_node = curr_node
                    curr_node = prev_node[a]
                    if curr_node is None:
                        curr_node = prev_node.expand(a, get_player_red_team())
                        is_on_tree = False
                else:
                    a = rn.choice(actions)
                
                var_gameState = var_gameState.generateSuccessor(curr_player_index, a)
                self.update_optimal_positions(var_gameState, t)
                update_curr_player()
                if var_gameState.isOver():
                    break
            curr_node.backprop(self.evaluate_state(var_gameState) - self.evaluate_state(gameState))
            n_runs += 1

        print('n_runs', n_runs)
        # print(get_tree_string(mc_tree))
        actions = gameState.getLegalActions(self.index)
        return max(actions, key=lambda a: greedy_score(mc_tree, a, self.red))
    
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

    def evaluate_state(self, gameState):
        heuristic = lambda foo, coeff: sum([coeff * (1 if gameState.isOnRedTeam(i) else -1) * foo(gameState, i) for i in range(4)])
        
        return 100 * gameState.getScore() + heuristic(lambda state, i: -self.getMazeDistance(state.getAgentPosition(i), self.map_center), 1e-6) + heuristic(lambda state, i: state.getAgentState(i).numCarrying, 1)


class Node:
    def __init__(s, is_red, parent=None):
        s.is_red = is_red
        s.parent = parent
        s.diff_score = 0
        s.n_sims = 0
        s.children = {}
    
    def __getitem__(s, key):
        if key not in s.children:
            return None
        return s.children[key]
    
    def expand(s, a, is_red):
        node = Node(is_red, s)
        s.children[a] = node
        return node
    
    def backprop(s, score):
        s.diff_score += score
        s.n_sims += 1
        if s.parent:
            s.parent.backprop(score)

def UCT(parent, a, is_red, c=1.414):
    if parent is not None:
        N = parent.n_sims + 1
    else:
        return 'N/A'
    
    if parent[a] is None:
        return c * math.sqrt(math.log(N) / 1)
    
    node = parent[a]
    w = node.diff_score if is_red else -node.diff_score
    n = node.n_sims + 1
    
    return w / n + c * math.sqrt(math.log(N) / n)

def greedy_score(parent, a, is_red):
    if a not in parent.children:
        return -1e6
    return parent[a].diff_score if is_red else -parent[a].diff_score

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
