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
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
import os

prevPacmanPosition = (0,0)

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

    def printLineData(self, gameState):
        import numpy as np
        relation = "\n@relation all-data-pacman"
        atribute1 = "\n@attribute pacmanXpos NUMERIC"
        atribute2 = "\n@attribute pacmanYpos NUMERIC"
        atribute3 = "\n@attribute pacmanDirec {West, East, North, South}"
        #numberGhosts = "@attribute"
        atribute4 = "\n@attribute LivingGhost1 {False, True}"
        atribute5 = "\n@attribute LivingGhost2 {False, True}"
        atribute6 = "\n@attribute LivingGhost3 {False, True}"
        atribute7 = "\n@attribute LivingGhost4 {False, True}"
        atribute9 = "\n@attribute ghost1XPos NUMERIC"
        atribute10 = "\n@attribute ghost1YPos NUMERIC"
        atribute11 = "\n@attribute ghost2XPos NUMERIC"
        atribute12 = "\n@attribute ghost2YPos NUMERIC"
        atribute13 = "\n@attribute ghost3XPos NUMERIC"
        atribute14 = "\n@attribute ghost3YPos NUMERIC"
        atribute15 = "\n@attribute ghost4XPos NUMERIC"
        atribute16 = "\n@attribute ghost4YPos NUMERIC"
        atribute17 = "\n@attribute ghost5XPos NUMERIC"
        atribute18 = "\n@attribute ghost5YPos NUMERIC"
        atribute19 = "\n@attribute ghost1Dist NUMERIC"
        atribute20 = "\n@attribute ghost2Dist NUMERIC"
        atribute21 = "\n@attribute ghost3Dist NUMERIC"
        atribute22 = "\n@attribute ghost4Dist NUMERIC"
        atribute23 = "\n@attribute ghost5Dist NUMERIC"
        clase = "\n@attribute action {West, East, North, South}"
        instance = [relation, atribute1, atribute2, atribute3, atribute4, atribute5, atribute6, atribute7, atribute9, atribute10, atribute11, atribute12, atribute13, atribute14, atribute15, atribute16, atribute17, atribute18, atribute19, atribute20, atribute21, atribute22, atribute23, clase, ""]

        if not os.path.isfile("weka-pacman/all-data-pacman.arff"):
            with open('weka-pacman/all-data-pacman.arff', 'w') as file:
                for i in instance:
                    file.write(i)
                file.write("\n@data")
                file.write("\n")

        new_line = []
        pacmanXPosition = gameState.getPacmanPosition()[0]
        pacmanYPosition = gameState.getPacmanPosition()[1]
        pacmanDirection = gameState.data.agentStates[0].getDirection()
        livingGhosts = gameState.getLivingGhosts()[1:]
        new_info = [pacmanXPosition, pacmanYPosition, pacmanDirection]+livingGhosts
        ghostPositions = gameState.getGhostPositions()
        ghostDistances = gameState.data.ghostDistances[:] #Copy the list, there'll be changes, so change the assigment
        
        for i in range(len(ghostDistances)):
            if ghostDistances[i] == None: ghostDistances[i] = 0

        takenAction = KeyboardAgent.getAction(self, gameState)

        for x in ghostPositions:
            for i in x:
                new_info.append(i)  
        
        new_info = new_info+ghostDistances
        new_info.append(takenAction)
        new_line.append(new_info)

        with open('weka-pacman/all-data-pacman.arff','a') as file:
            np.savetxt(file, new_line, delimiter=',', fmt='%s')
        
        print(new_info)

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

    def printInfo(self, gameState): 
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)  
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)]) ##THIS
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances) ##THIS
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood()) ##THIS
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood()) ##THIS
        # Map walls
        print("Map:")
        print(gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())
        
        
    def chooseAction(self, gameState): 
        global prevPacmanPosition

        self.countActions = self.countActions + 1
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        nearestGhost = min(d for d in gameState.data.ghostDistances if d is not None) #Distance - problem whit None type
        nearestGhostPosition = gameState.getGhostPositions()[gameState.data.ghostDistances.index(nearestGhost)] #With index from neares Ghost in list ghostDistances, we get position of the ghost with list getghostPositions
        pacmanPosition = gameState.getPacmanPosition() #Tuple
        pacmanCurrentDirection = 0
        if gameState.data.agentStates[0].getDirection() == "North": pacmanCurrentDirection = 2
        elif gameState.data.agentStates[0].getDirection() == "South": pacmanCurrentDirection = 3
        elif gameState.data.agentStates[0].getDirection() == "West": pacmanCurrentDirection = 0
        elif gameState.data.agentStates[0].getDirection() == "East": pacmanCurrentDirection = 1

        dx = nearestGhostPosition[0]-pacmanPosition[0] #Distance x axis between Ghost and Pacman (horizontal movement)
        dy = nearestGhostPosition[1]-pacmanPosition[1] #Distance y axis between Ghost and Pacman (vertical movement)

        #Walls araound pacman
        pacmanWallUp = gameState.getWalls()[gameState.getPacmanPosition()[0]][gameState.getPacmanPosition()[1]+1] #If there is a wall in current pacman's position
        pacmanWallDown = gameState.getWalls()[gameState.getPacmanPosition()[0]][gameState.getPacmanPosition()[1]-1]
        pacmanWallLeft = gameState.getWalls()[gameState.getPacmanPosition()[0]-1][gameState.getPacmanPosition()[1]]
        pacmanWallRight = gameState.getWalls()[gameState.getPacmanPosition()[0]+1][gameState.getPacmanPosition()[1]]
        wallsAroundPacman = [pacmanWallLeft, pacmanWallRight,  pacmanWallUp, pacmanWallDown]

        movement = random.randint(0,3)
        if dy < 0 and not pacmanWallDown and pacmanPosition[1]-1 is not prevPacmanPosition[1]:
            movement = 3
        elif dy > 0 and not pacmanWallUp and pacmanPosition[1]+1 is not prevPacmanPosition[1]:
            movement = 2
        elif dx < 0 and not pacmanWallLeft and pacmanPosition[0]-1 is not prevPacmanPosition[0]:
            movement = 0
        elif dx > 0 and not pacmanWallRight and pacmanPosition[0]+1 is not prevPacmanPosition[0]:
            movement = 1
        
        elif pacmanWallUp and pacmanWallDown and not pacmanWallLeft and not pacmanWallRight: movement = pacmanCurrentDirection
        elif pacmanWallRight and pacmanWallLeft and not pacmanWallUp and not pacmanWallDown: movement = pacmanCurrentDirection
        elif pacmanWallLeft and pacmanWallUp and not pacmanWallRight and not pacmanWallDown:
            if pacmanCurrentDirection == 2: movement = 1 
            else: movement = 3
        elif pacmanWallLeft and pacmanWallDown and not pacmanWallRight and not pacmanWallUp:
            if pacmanCurrentDirection == 3: movement = 1
            else: movement = 2
        elif pacmanWallRight and pacmanWallUp and not pacmanWallLeft and not pacmanWallDown:  
            if pacmanCurrentDirection == 2: movement = 0
            else: movement = 3
        elif pacmanWallRight and pacmanWallDown and not pacmanWallLeft and not pacmanWallUp:
            if pacmanCurrentDirection == 3: movement = 0
            else: movement = 2
        elif wallsAroundPacman.count(False) == 1:
            wall = wallsAroundPacman.index(False)
            movement = wall
        elif wallsAroundPacman.count(True) == 1:
            wall = wallsAroundPacman.index(True)
            if pacmanCurrentDirection is not wall: movement = pacmanCurrentDirection
                
    
        if   ( movement == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( movement == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( movement == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( movement == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH

        prevPacmanPosition = pacmanPosition
        

        return move


    def printLineData(self, gameState):
        import numpy as np
        relation = "\n@relation all-data-pacman"
        atribute2 = "\n@attribute pacmanXpos NUMERIC"
        atribute3 = "\n@attribute pacmanYpos NUMERIC"
        atribute1 = "\n@attribute pacmanDirec {West, East, North, South, Stop}"
        atribute4 = "\n@attribute LivingGhost1 {False, True}"
        atribute5 = "\n@attribute LivingGhost2 {False, True}"
        atribute6 = "\n@attribute LivingGhost3 {False, True}"
        atribute7 = "\n@attribute LivingGhost4 {False, True}"
        atribute9 = "\n@attribute ghost1XPos NUMERIC"
        atribute10 = "\n@attribute ghost1YPos NUMERIC"
        atribute11 = "\n@attribute ghost2XPos NUMERIC"
        atribute12 = "\n@attribute ghost2YPos NUMERIC"
        atribute13 = "\n@attribute ghost3XPos NUMERIC"
        atribute14 = "\n@attribute ghost3YPos NUMERIC"
        atribute15 = "\n@attribute ghost4XPos NUMERIC"
        atribute16 = "\n@attribute ghost4YPos NUMERIC"
        atribute17 = "\n@attribute ghost1Dist NUMERIC"
        atribute18 = "\n@attribute ghost2Dist NUMERIC"
        atribute19 = "\n@attribute ghost3Dist NUMERIC"
        atribute20 = "\n@attribute ghost4Dist NUMERIC"
        clase = "\n@attribute action {West, East, North, South}"
        instance = [relation, atribute1, atribute2, atribute3, atribute4, atribute5, atribute6, atribute7, atribute9, atribute10, atribute11, atribute12, atribute13, atribute14, atribute15, atribute16, atribute17, atribute18, atribute19, atribute20, clase]

        if not os.path.isfile("weka-pacman/all-data-pacman.arff"):
            with open('weka-pacman/all-data-pacman.arff', 'w') as file:
                for i in instance:
                    file.write(i)
                file.write("\n@data")
                file.write("\n")

        new_line = []
        pacmanXPosition = gameState.getPacmanPosition()[0]
        pacmanYPosition = gameState.getPacmanPosition()[1]
        pacmanDirection = gameState.data.agentStates[0].getDirection()
        livingGhosts = gameState.getLivingGhosts()[1:]
        new_info = [pacmanDirection, pacmanXPosition, pacmanYPosition]+livingGhosts
        ghostPositions = gameState.getGhostPositions()
        ghostDistances = gameState.data.ghostDistances[:] #Copy the list, there'll be changes, so change the assigment
        
        for i in range(len(ghostDistances)):
            if ghostDistances[i] == None: ghostDistances[i] = 0

        takenAction = BustersAgent.getAction(self, gameState)

        for x in ghostPositions:
            for i in x:
                new_info.append(i)  
        
        new_info = new_info+ghostDistances
        new_info.append(takenAction)
        new_line.append(new_info)

        with open('weka-pacman/all-data-pacman.arff','a') as file:
            np.savetxt(file, new_line, delimiter=',', fmt='%s')
        
        print(new_info)

    def printFilterData1(self, gameState):
        import numpy as np
        relation = "\n@relation filter-data-pacman-manual1"
        atribute2 = "\n@attribute pacmanXpos NUMERIC"
        atribute3 = "\n@attribute pacmanYpos NUMERIC"
        atribute1 = "\n@attribute pacmanDirec {West, East, North, South, Stop}"
        atribute4 = "\n@attribute ghost1XPos NUMERIC"
        atribute5 = "\n@attribute ghost1YPos NUMERIC"
        atribute6 = "\n@attribute ghost2XPos NUMERIC"
        atribute7 = "\n@attribute ghost2YPos NUMERIC"
        atribute8 = "\n@attribute ghost3XPos NUMERIC"
        atribute9 = "\n@attribute ghost3YPos NUMERIC"
        atribute10 = "\n@attribute ghost4XPos NUMERIC"
        atribute11 = "\n@attribute ghost4YPos NUMERIC"
        clase = "\n@attribute action {West, East, North, South}"
        instance = [relation, atribute1, atribute2, atribute3, atribute4, atribute5, atribute6, atribute7, atribute8, atribute9, atribute10, atribute11, clase]

        if not os.path.isfile("weka-pacman/filter-data-pacman-manual1.arff"):
            with open('weka-pacman/filter-data-pacman-manual1.arff', 'w') as file:
                for i in instance:
                    file.write(i)
                file.write("\n@data")
                file.write("\n")

        new_line = []
        pacmanXPosition = gameState.getPacmanPosition()[0]
        pacmanYPosition = gameState.getPacmanPosition()[1]
        pacmanDirection = gameState.data.agentStates[0].getDirection()
        new_info = [pacmanDirection, pacmanXPosition, pacmanYPosition]
        ghostPositions = gameState.getGhostPositions()

        takenAction = BustersAgent.getAction(self, gameState)

        for x in ghostPositions:
            for i in x:
                new_info.append(i)  
        
        new_info.append(takenAction)
        new_line.append(new_info)

        with open('weka-pacman/filter-data-pacman-manual1.arff','a') as file:
            np.savetxt(file, new_line, delimiter=',', fmt='%s')
        
        print("filter 1:", new_info)

    def printFilterData2(self, gameState):
        import numpy as np
        relation = "\n@relation filter-data-pacman-manual2"
        atribute1 = "\n@attribute LivingGhost1 {False, True}"
        atribute2 = "\n@attribute LivingGhost2 {False, True}"
        atribute3 = "\n@attribute LivingGhost3 {False, True}"
        atribute4 = "\n@attribute LivingGhost4 {False, True}"
        atribute5 = "\n@attribute ghost1Dist NUMERIC"
        atribute6 = "\n@attribute ghost2Dist NUMERIC"
        atribute7 = "\n@attribute ghost3Dist NUMERIC"
        atribute8 = "\n@attribute ghost4Dist NUMERIC"
        clase = "\n@attribute action {West, East, North, South}"
        instance = [relation, atribute1, atribute2, atribute3, atribute4, atribute5, atribute6, atribute7, atribute8, clase]

        if not os.path.isfile("weka-pacman/filter-data-pacman-manual2.arff"):
            with open('weka-pacman/filter-data-pacman-manual2.arff', 'w') as file:
                for i in instance:
                    file.write(i)
                file.write("\n@data")
                file.write("\n")

        new_line = []
        livingGhosts = gameState.getLivingGhosts()[1:]
        new_info = livingGhosts[:]
        ghostDistances = gameState.data.ghostDistances[:] #Copy the list, there'll be changes, so change the assigment
        
        for i in range(len(ghostDistances)):
            if ghostDistances[i] == None: ghostDistances[i] = 0

        takenAction = BustersAgent.getAction(self, gameState)
        
        new_info = new_info+ghostDistances
        new_info.append(takenAction)
        new_line.append(new_info)

        with open('weka-pacman/filter-data-pacman-manual2.arff','a') as file:
            np.savetxt(file, new_line, delimiter=',', fmt='%s')
        
        print("filter 2: ", new_info)