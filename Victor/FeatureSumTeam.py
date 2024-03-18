from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'AggresiveFeatureSumAgent', second = 'DefensiveFeatureSumAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class FeatureSumAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    
    self.ref_positions = gameState.getBlueFood().asList() + gameState.getRedFood().asList()

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    
    # relevant features:
    # if self is Pacman:
    #   self stored food (V)
    #   closest food distance (V)
    #   self stored food * distance to self team side (V)
    #   closest opposing ghost distance
    #   closest opposing scared ghost distance
    #   (closest capsule distance) * (self not yet powered / opposing team not scared)
    #
    # if self is Ghost:
    #   (closest opposing Pacman) * (1 for self is not scared else -1)

    myState = successor.getAgentState(self.index)

    if myState.isPacman:
        self.getPacmanFeatures(features, successor, myState)
    else:
        self.getGhostFeatures(features, successor, myState)
    
    return features
    
  def getPacmanFeatures(self, features, successor, myState):
    myPos = myState.getPosition()
    features['self_stored'] = myState.numCarrying
    
    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['closest_food_dist'] = minDistance
    
    half_max_dist = max([self.getMazeDistance(w, self.start) for w in self.ref_positions if successor.isRed(w) == successor.isOnRedTeam(self.index)])
    features['self_score_potential'] = myState.numCarrying * (self.getMazeDistance(myPos, self.start) / half_max_dist)

  def getGhostFeatures(self, features, successor, myState):
    myPos = myState.getPosition()
    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    enemy_pacmans = [a for a in enemies if a.isPacman and a.getPosition() != None]
    if len(enemy_pacmans) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemy_pacmans]
      features['nearest_enemy_pacman_dist'] = min(dists) * (1 if not myState.scaredTimer else -1)


  def getWeights(self, gameState, action):
    return {'self_stored': 2, 'closest_food_dist': -0.1, 'self_score_potential': -1, 'nearest_enemy_pacman_dist': -1}


class AggresiveFeatureSumAgent(FeatureSumAgent):
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    
    self.getPacmanFeatures(features, successor, myState)
    
    return features


class DefensiveFeatureSumAgent(FeatureSumAgent):
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    
    self.getGhostFeatures(features, successor, myState)
    
    return features
