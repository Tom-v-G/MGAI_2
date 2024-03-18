from captureAgents import CaptureAgent
import distanceCalculator
import random as rn, time, util, sys
from game import Directions
import game
from util import nearestPoint

from collections import deque
import math

# MCTS agent with reward model and Qvalue dampening factor gamma

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MCTSAgent', second = 'MCTSAgent'):

    return [eval(first)(firstIndex), eval(second)(secondIndex)]

class MCTSAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState, searchDepth=40, soft_time_limit=0.95, ignore_other_players=False, gamma=0.6, epsilon=0.1):

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
        while time.time() - start_time < soft_time_limit:
            curr_player_index = self.index
            var_gameState = gameState.deepCopy()
            prev_state_score = self.evaluate_state(var_gameState)
            curr_node = mc_tree
            is_on_tree = True
            rewards = deque([0])
            gamma_acc = 1
            for _ in range(searchDepth):
                was_on_tree = is_on_tree
                if is_on_tree:
                    actions = var_gameState.getLegalActions(curr_player_index)
                    # a = max(actions, key=lambda a: UCT(curr_node, a, get_player_red_team()))
                    # a = epsilon_choice(actions, epsilon, curr_node, get_player_red_team())
                    a = rn.choice(var_gameState.getLegalActions(curr_player_index))
                    prev_node = curr_node
                    curr_node = prev_node[a]
                    if curr_node is None:
                        curr_node = prev_node.expand(a, get_player_red_team())
                        is_on_tree = False
                else:
                    a = rn.choice(var_gameState.getLegalActions(curr_player_index))
                
                var_gameState = var_gameState.generateSuccessorNoCopy(curr_player_index, a) # may have to CHANGE BACK
                curr_state_score = self.evaluate_state(var_gameState)
                if was_on_tree:
                    rewards.append(curr_state_score - prev_state_score)
                else:
                    gamma_acc *= gamma
                    rewards[-1] += gamma_acc * (curr_state_score - prev_state_score)
                update_curr_player()
                prev_state_score = curr_state_score
                
                if var_gameState.isOver():
                    break
            curr_node.backprop(rewards, 0, gamma)
            n_runs += 1

        print('n_runs', n_runs)
        # print(get_tree_string(mc_tree))
        actions = gameState.getLegalActions(self.index)
        return max(actions, key=lambda a: greedy_score(mc_tree, a, self.red))

    def evaluate_state(self, gameState):
        heuristic = lambda foo, coeff: sum([coeff * (1 if gameState.isOnRedTeam(i) else -1) * foo(gameState, i) for i in range(4)])
        map_center = (gameState.data.layout.width // 2, gameState.data.layout.height // 2)
        
        return 1e10 * gameState.getScore() + heuristic(lambda state, i: -self.getMazeDistance(state.getAgentPosition(i), map_center), 1e-40) + heuristic(lambda state, i: state.getAgentState(i).numCarrying, 1)


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
            s.parent.backprop(rewards, gamma * reward, gamma)

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
