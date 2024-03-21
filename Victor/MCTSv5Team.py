from captureAgents import CaptureAgent
import distanceCalculator
import random as rn, time, util, sys
from game import Directions
import game
from util import nearestPoint

from collections import deque
import math

# MCTS agent with reward model, Qvalue dampening factor gamma and two MCTS strategies, competitive and single

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MCTSAgent', second = 'DEFMCTSAgent'):

    return [eval(first)(firstIndex), eval(second)(secondIndex)]

class MCTSAgent(CaptureAgent):
    
    targetFood = None
    
    def  __init__(self, index, timeForComputing=0.1):
        super().__init__(index, timeForComputing)
        self.defensive = False
        self.delta = 4
        self.prevNumCarry = 0
    
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        self.numAgents = gameState.getNumAgents()
        CaptureAgent.registerInitialState(self, gameState)
        self.generate_map_center(gameState)

    def chooseAction(self, gameState, searchDepth=80, soft_time_limit=0.55, gamma=0.60):
        print(f'Defensive: {self.defensive}')
        start_time = time.time()
        
        mc_trees = [Node(self.red), Node(self.red)]
        def update_curr_player(single_player):
            nonlocal curr_player_index
            if single_player:
                curr_player_index = curr_player_index
            else:
                curr_player_index = (curr_player_index + 1) % 4
        get_player_red_team = lambda: curr_player_index in var_gameState.redTeam
        
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

        print('n_runs', n_runs)
        actions = gameState.getLegalActions(self.index)
        stds = [get_std(playouts[0]), get_std(playouts[1])]
        max_diffs = [get_max_diff(mc_trees[0], actions), get_max_diff(mc_trees[1], actions)]
        
        if safe_div(max_diffs[1], stds[1]) >= 2 * safe_div(max_diffs[0], stds[0]):
            chosen_mode = 1 # Individual play
            print('Choosing individual strategy')
        else:
            chosen_mode = 0 # Competitive play
            print('Choosing competitive strategy')
        
        print(max(actions, key=lambda a: greedy_score(mc_trees[chosen_mode], a, self.red)))
        #input()
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

    def evaluate_state(self, gameState):
        # Score is positive for red and negative for blue
        score = 1e6 * gameState.getScore()
        # General heuristic definition thats negative for blue agents
        heuristic = lambda foo, coeff: sum([coeff * (1 if gameState.isOnRedTeam(i) else -1) * foo(gameState, i) for i in range(4)])
        
        redTeam = gameState.getRedTeamIndices()
        blueTeam = gameState.getBlueTeamIndices()
        # Check if it is team turn or enemy turn
        if gameState.data._agentMoved is not None:
            currentMove = ((gameState.data._agentMoved + 1) % self.numAgents) 
        else: currentMove = 0
        
        #print(f'Current move for {currentMove}')
 
        teamturn = currentMove in redTeam if self.red else currentMove in blueTeam
        
        # Own team heuristics
        if teamturn:
            #print('teamturn')
            # defensive agent heuristics
            # assumes 1 defensive and 1 offensive agent on team
            if (self.defensive and currentMove == self.index) or (not self.defensive and currentMove != self.index):
                #print('using defensive heuristic')
                if self.red: 
                    coef = 1e-6
                    enemyidxs = blueTeam
                else: 
                    coef = -1e-6
                    enemyidxs = redTeam
                enemy_pos_list = []
                for agent in enemyidxs:
                    if gameState.getAgentState(agent).isPacman:
                        enemy_pos_list.append(agent)
                if enemy_pos_list:
                    # minimize distance to the closest enemy agent in the list
                    dist_list = [self.getMazeDistance(gameState.getAgentPosition(currentMove), gameState.getAgentPosition(i)) for i in enemy_pos_list]
                    dist = coef * -min(dist_list)
                else:
                    dist = coef * -self.getMazeDistance(gameState.getAgentPosition(currentMove), self.map_center)
                carrying = heuristic(lambda state, i: state.getAgentState(i).numCarrying, 1)
            
            # offensive agent heuristics
            else: 
                #print('using offensive heuristic')
                goForFood = False
                coef = 1e-6 if self.red else -1e-6
                if gameState.getAgentState(self.index).numCarrying == 0:
                    if MCTSAgent.targetFood is None:
                        print('Choosing Food')
                        if self.red: food = gameState.getBlueFood()
                        else : food = gameState.getRedFood()
                        pos_list = []
                        for x, col in enumerate(food):
                            for y, row in enumerate(col):
                                if food[x][y] : pos_list.append(tuple([x , y]))
                        print(f'Possible Choices {pos_list}')
                        MCTSAgent.targetFood = rn.choice(pos_list)
                        print(f'chose {MCTSAgent.targetFood}')
                    goForFood = True
                elif self.prevNumCarry == 0:
                    self.prevNumCarry = gameState.getAgentState(self.index).numCarrying
                    goForFood = True
                    self.delta = 4
                elif self.delta > 0:
                    self.delta -= 1
                    goForFood = True
                if goForFood:
                    dist = coef * -self.getMazeDistance(gameState.getAgentPosition(currentMove), MCTSAgent.targetFood)
                else : 
                    dist = coef * -self.getMazeDistance(gameState.getAgentPosition(currentMove), self.map_center)
                carrying = heuristic(lambda state, i: state.getAgentState(i).numCarrying, 1)
        
        # Enemy Team heuristics (simple)
        else:
            #print('enemy turn')
            # if not gameState.getAgentState(currentMove).isPacman: #ghost uses defensive heuristic
            #     if self.red: 
            #         coef = -1e-6
            #         enemyidxs = redTeam
            #     else: 
            #         coef = 1e-6
            #         enemyidxs = blueTeam
            #     enemy_pos_list = []
            #     for agent in enemyidxs:
            #         if gameState.getAgentState(agent).isPacman:
            #             enemy_pos_list.append(agent)
            #     if enemy_pos_list:
            #         # minimize distance to the closest enemy agent in the list
            #         dist_list = [self.getMazeDistance(gameState.getAgentPosition(currentMove), gameState.getAgentPosition(enemy_pos_list[i])) for i in enemy_pos_list]
            #         dist = coef * -min(dist_list)
            #     else:
            #         dist = coef * -self.getMazeDistance(gameState.getAgentPosition(currentMove), self.map_center)
            #     carrying = heuristic(lambda state, i: state.getAgentState(i).numCarrying, 1)               
            # else :
                dist =  heuristic(lambda state, i: -self.getMazeDistance(state.getAgentPosition(i), self.map_center), 1e-6)
                carrying = heuristic(lambda state, i: state.getAgentState(i).numCarrying, 1)
        #print(f'{score} + {dist} + {carrying}')
        #input()
        return score + dist + carrying


class DEFMCTSAgent(MCTSAgent):
    def __init__(self, index, timeForComputing=0.1):
        super().__init__(index, timeForComputing)
        self.defensive = True

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
