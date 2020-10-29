# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        "*** YOUR CODE HERE ***"
        if successorGameState.isLose(): return -999999999
        evaluation = 0.0
        foodList = newFood.asList()
        foodDistance = []
        ghostDistance = []
        if foodList:
            for i in foodList: foodDistance.append(manhattanDistance(newPos,i))
            evaluation -= 0.9 * min(foodDistance)
            evaluation -= (0.1 * sum(foodDistance) / len(foodDistance))
        for i in successorGameState.getGhostPositions(): ghostDistance.append(manhattanDistance(newPos,i))
        if min(ghostDistance) < 3: evaluation -= 99999
        evaluation -= 0.3 * min(ghostDistance)
        if newPos in currentGameState.getFood().asList(): evaluation += 999
        for i in newScaredTimes: evaluation += i
        evaluation += 1.2 * (successorGameState.getScore() - currentGameState.getScore())
        return evaluation

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def Minimax(gameState, bound, depth, num_agents):
            best_move = Directions.STOP
            if bound * num_agents == depth or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(gameState), best_move
            turn = depth % num_agents
            value = 0
            if not turn: value = -99999999999
            if turn: value = 99999999999
            for move in gameState.getLegalActions(turn):
                nxt_val, nxt_move = Minimax(gameState.generateSuccessor(turn, move),bound,depth+1,num_agents)
                if not turn and value < nxt_val:
                    value, best_move = nxt_val, move
                if turn and value > nxt_val:
                    value, best_move = nxt_val, move
            return value, best_move
        res = Minimax(gameState, self.depth, 0, gameState.getNumAgents())
        return res[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def AlphaBeta(gameState, bound, depth, num_agents, alpha, beta):
            best_move = " "
            if bound * num_agents == depth or gameState.isWin() or gameState.isLose(): return self.evaluationFunction(gameState), best_move
            turn = depth % num_agents
            value = 0.0
            if not turn: value = -9999999999.0
            else: value = 9999999999.0
            for move in gameState.getLegalActions(turn):
                nxt_val, nxt_move = AlphaBeta(gameState.generateSuccessor(turn, move), bound, depth + 1, num_agents, alpha, beta)
                if not turn:
                    if value < nxt_val : value, best_move = nxt_val, move
                    if value >= beta: return value, best_move
                    alpha = max(alpha, value)
                else:
                    if value > nxt_val: value, best_move = nxt_val, move
                    if value <= alpha: return value, best_move
                    beta = min(beta, value)
            return value, best_move
        res = AlphaBeta(gameState,self.depth,0, gameState.getNumAgents(), -999999999999.0, 999999999999.0)
        return res[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def Expectimax(gameState, depth, bound, num_agents):
            best_move = Directions.STOP
            if bound * num_agents == depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), best_move
            turn = depth % num_agents
            value = 0.0
            if not turn: value = -9999999999.0
            else: value = 0.0
            for move in gameState.getLegalActions(turn):
                nxt_val, nxt_move = Expectimax(gameState.generateSuccessor(turn, move), depth + 1, bound, num_agents)
                if not turn and value < nxt_val: value, best_move = nxt_val, move
                elif turn != 0:
                    value += nxt_val / float(len(gameState.getLegalActions(turn)))
            return value, best_move
        res = Expectimax(gameState, 0, self.depth, gameState.getNumAgents())
        return res[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentPosition = currentGameState.getPacmanPosition()
    currentGhost = currentGameState.getGhostStates()
    foodList = currentGameState.getFood().asList()
    if currentGameState.isWin(): return 999999999.9
    if currentGameState.isLose(): return -999999999.9
    evaluation = 0.0
    ghostDistance = []
    for i in currentGhost: ghostDistance.append((manhattanDistance(i.getPosition(),currentPosition)))
    foodDistance = []
    for i in foodList: foodDistance.append(manhattanDistance(i,currentPosition))
    for i in foodDistance:
        if i < 3: evaluation -= i
        elif i < 9: evaluation -= 0.5 * i
        else: evaluation -= 0.1 * i
    for i in ghostDistance:
        if i < 3: evaluation -= 19 * i
        else: evaluation -= 10 * i
    evaluation += 1.4 * currentGameState.getScore()
    evaluation += -12 * len(foodList)
    evaluation += -20 * len(currentGameState.getCapsules())
    return evaluation
# Abbreviation
better = betterEvaluationFunction
