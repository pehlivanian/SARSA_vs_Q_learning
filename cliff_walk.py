import abc
import functools
import sys
import numpy as np
import sys
import matplotlib.pyplot as plt
from fsm import CliffWalkingStateMachine, SARSA_decorator, Q_learner_decorator

'''
    Setup for cliff-walking exercise. The grid is of size (nrows, ncols), with allowable
    actions: up, down, left, right.

    The setup requires specification of the following

        ACTIONS         =  ['UP', 'DOWN', 'LEFT', 'RIGHT' ]
        VISIBLE_STATES  =  ['START', 'INTERIOR', 'OOB', 'EDGE', 'CLIFF', 'DESTINATION' ]
        INITIAL_STATE   =  'START'
        TERMINAL_STATES =  ['CLIFF', 'DESTINATION'] 
        REWARDS         =  [-1, -1, -1, -1, -100, -1]

    The boundary conditions should also be setup

        _isEdgeCoordinate(self, x, y)
        _isCliffCoordinate(self, x, y)
        _isStartCoordinate(self, x, y)
        _isDestinationCoordinate(self, x, y)
        _isOOBCoordinate(self, x, y)

  A CliffWalk class with below parameters provides a typical example of a walk on a 8x10 grid,
  starting at (0,0), with destination (10,-1), and repelling walls along the lines {x = 0}, {y = 8},
  and { x = 10 }.

  Change the parameters below to modify the grid size, rewards, solving parameters for SARSA,
  Q-learning solvers.

  Output in '{solver type}_Q_hat.png'

'''    
    
nrows                   = 8
ncols                   = 10
nact                    = 4

nepisodes               = 100000
epsilon                 = 0.25
alpha                   = 0.1
gamma                   = 0.95

reward_normal           = -1
reward_cliff            = -100
reward_destination      = -1


class CliffWalk(CliffWalkingStateMachine):

    START_COORDINATES = (0, -1)
    DESTINATION_COORDINATES = (ncols-1, -1)

    @functools.lru_cache(maxsize=512)
    def _isEdgeCoordinate(self, x, y):
        return y == 0

    @functools.lru_cache(maxsize=512)
    def _isCliffCoordinate(self, x, y):
        return y < 0 and not self._isDestinationCoordinate(x, y)

    @functools.lru_cache(maxsize=512)
    def _isStartCoordinate(self, x, y):
        return x == self.START_COORDINATES[0] and y == self.START_COORDINATES[1]

    @functools.lru_cache(maxsize=512)    
    def _isDestinationCoordinate(self, x, y):
        return x == self.DESTINATION_COORDINATES[0] and y == self.DESTINATION_COORDINATES[1]

    @functools.lru_cache(maxsize=512)
    def _isOOBCoordinate(self, x, y):
        return x <= -1 or x >= ncols or y >= nrows and \
            not self._isStartCoordinate(x, y) and not self._isDestinationCoordinate(x, y)
    
    def _goToStart(self):
        self.setState('START')
        self.set_coordinates(self.START_COORDINATES)
        return self.START_COORDINATES


# Use Q_learning
cw = Q_learner_decorator( CliffWalk( nrows, ncols), alpha, gamma, epsilon)

for n in range(nepisodes+1):
    if (n % 1000 == 0):
        print('episode # ', n)

    cw._goToStart()

    action = cw.explore_exploit()

    while not cw.isTerminalState():
        new_action = cw.move(action)
        action = new_action

cw.plot_Q('Q_learner_Q_hat.png')
