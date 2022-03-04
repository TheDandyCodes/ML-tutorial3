from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
from subprocess import ABOVE_NORMAL_PRIORITY_CLASS
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters

abierta = []
cerrada = [((0,0), 0, 0, 0),((0,0), 0, 0, 0)]

class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."
    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
        
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST

class BasicAgentAA(BustersAgent): #############################INTERESA#############################

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState): #############################INTERESA#############################
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())#
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())#
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)  
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())#
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())#
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances) #
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood()) 
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood()) 
        # Map walls
        print("Map:")
        print(gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())
        
        
    def chooseAction(self, gameState):
        global abierta
        global f_abierta
        global cerrada

        self.countActions = self.countActions + 1
        move = Directions.STOP
        legal = gameState.getLegalActions(0) #Legal position from the pacman

        nearestGhost = min(d for d in gameState.data.ghostDistances if d is not None) #Objective
        pacmanPosition = gameState.getPacmanPosition() #Tuple
        

        #Walls araound pacman
        pacmanWallUp = gameState.getWalls()[gameState.getPacmanPosition()[0]][gameState.getPacmanPosition()[1]+1] #If there is a wall in current pacman's position
        pacmanWallDown = gameState.getWalls()[gameState.getPacmanPosition()[0]][gameState.getPacmanPosition()[1]-1]
        pacmanWallLeft = gameState.getWalls()[gameState.getPacmanPosition()[0]-1][gameState.getPacmanPosition()[1]]
        pacmanWallRight = gameState.getWalls()[gameState.getPacmanPosition()[0]+1][gameState.getPacmanPosition()[1]]


        
        #Expande un estado si no hay un muro en la posición contigua
        if not pacmanWallUp:
            h = gameState.data.ghostDistances.index(nearestGhost) #Manhattan distance from next node
            lastStates = cerrada[::-1][0:4]
            var = [i for i in lastStates if i[0] == (pacmanPosition[0], pacmanPosition[1]+1)]
            if len(var) != 0: g = cerrada[-1][1]+3
            else: g = cerrada[-1][1]+1
            f = g+h
            newState = ((pacmanPosition[0],pacmanPosition[1]+1), g, h, f) 
            abierta.append(newState)
            f_abierta.append(f)
        if not pacmanWallDown:
            h = gameState.data.ghostDistances.index(nearestGhost)
            lastStates = cerrada[::-1][0:4]
            var = [i for i in lastStates if i[0] == (pacmanPosition[0],pacmanPosition[1]-1)]
            if len(var) != 0: g = cerrada[-1][1]+3
            else: g = cerrada[-1][1]+1
            f = g+h
            abierta.append(((pacmanPosition[0],pacmanPosition[1]-1), g, h, f))
            f_abierta.append(f)
        if not pacmanWallLeft:
            h = gameState.data.ghostDistances.index(nearestGhost)
            lastStates = cerrada[::-1][0:4]
            var = [i for i in cerrada if i in lastStates if i[0] == (pacmanPosition[0]-1,pacmanPosition[1])]
            if len(var) != 0: g = cerrada[-1][1]+3
            else: g = cerrada[-1][1]+1
            f = g+h
            abierta.append(((pacmanPosition[0]-1,pacmanPosition[1]), g, h, f))
            f_abierta.append(f)
        if not pacmanWallRight:
            h = gameState.data.ghostDistances.index(nearestGhost)
            lastStates = cerrada[::-1][0:4]
            var = [i for i in lastStates if i[0] == (pacmanPosition[0]+1,pacmanPosition[1])]#var = [i for i in cerrada if i in lastStates]
            if len(var) != 0: g = cerrada[-1][1]+3
            else: g = cerrada[-1][1]+1
            f = g+h
            abierta.append(((pacmanPosition[0]+1,pacmanPosition[1]), g, h, f))
            f_abierta.append(f)

        abiertaMinF = [i for i in abierta if i[3] == min(abierta, key = lambda t: t[-1])[3]]
        sort = sorted(abiertaMinF, key = lambda g: g[1])
        nextState = sort[0]

        if nextState[0][0] > pacmanPosition[0]:
            movement = 1
        elif nextState[0][0] < pacmanPosition[0]:   
            movement = 0
        elif nextState[0][1] > pacmanPosition[1]:
            movement = 2
        elif nextState[0][1] < pacmanPosition[1]:
            movement = 3
    
    
        if   ( movement == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( movement == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( movement == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( movement == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH

        abierta.clear()
        f_abierta.clear()
        cerrada.append(nextState)

        return move

    def printLineData(self, gameState):
        import numpy as np
        new_line = []
        pacmanXPosition = gameState.getPacmanPosition()[0]
        pacmanYPosition = gameState.getPacmanPosition()[1]
        pacmanDirection = gameState.data.agentStates[0].getDirection()
        livingGhosts = gameState.getLivingGhosts()
        new_info = [pacmanXPosition, pacmanYPosition, pacmanDirection]+livingGhosts
        ghostPositions = gameState.getGhostPositions()
        ghostDistances = gameState.data.ghostDistances

        for x in ghostPositions:
            for i in x:
                new_info.append(i)  
        
        new_info = new_info+ghostDistances
        new_line.append(new_info)

        with open('LineData.csv', 'a') as file:
            np.savetxt(file, new_line, delimiter=',', fmt='%s')
        with open('LineData.txt', 'a') as file2: 
            np.savetxt(file2, new_line, delimiter=',', fmt='%s')
        
        print(new_info)