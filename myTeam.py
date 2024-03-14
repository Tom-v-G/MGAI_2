# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """


  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)





class MCTSAgent(CaptureAgent):
  """
  A basic MCTS agent 
  """

  def __init__(self, index, timeForComputing=0.1):
    super().__init__(index, timeForComputing)  
    self.searchTree = TreeNode()

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    raise NotImplementedError 
    
  def chooseAction(self, gameState):

    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    raise NotImplementedError
  
  def select(self):
    raise NotImplementedError
  
  def MCTS(self, root, max_time) -> None:
    '''
    Performs Monte Carlo Tree Search update on the current gamestate
    '''

    '''
    Pseudocode:
    create root Node v with gameState s
    while budget > 0:
      v_l = Treepolicy(v_0)
      delta = DefaultPolicy(s(v_l))
      Backup(v_l, Delta)
    return a(BestChild(v_0, 0))
    '''
    start = time.time()
    root = TreeNode()
    while(time.time() - start <= max_time): #budget
      selected_node = self.select(root)
      child_node = self.expand(selected_node)
      value = self.simulate(child_node)
      self.backpropagate(selected_node, child_node, value)

    raise NotImplementedError


class TreeNode():
  '''
  Search tree node for use in MCTS algorithm
  '''

  def __init__(self) -> None:
    self.timesvisited = 0
    self.children = []
    self.expanded = False

  def check_childNode(self, action) -> bool:
    raise NotImplementedError
  
  def expand_childNode(self, action) -> None:
    raise NotImplementedError
  
  def isLeaf(self) -> bool:
    raise NotImplementedError
  
  def value(self) -> int:
    raise NotImplementedError
  
  def get_timesvisited(self):
    return self.timesvisited
  
  def set_timesvisited(self, n: int) -> None:
    self.timesvisited = n

  def bestChild(self, c: float):
    '''
    Returns the best action to take in this node. 
    Nodes are ranked according to their UCB value
    '''

    max_val = -1
    for child in self.children:
      fd

    raise NotImplementedError

  

