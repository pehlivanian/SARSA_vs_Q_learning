import abc
from datetime import datetime
import functools
from pytz import timezone
import enum
import numpy as np
import logging
from tornado.log import app_log

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')
app_log.setLevel(logging.INFO)

DEBUG_FLAG = False

def random_action():
    # a = 0 ~ top/north
    # a = 1 ~ right/east
    # a = 2 ~ bottom/south
    # a = 3 ~ left/west
    nact = 4
    a = np.random.randint(nact)
    return a

class Event(object):
    def __init__(self, name='', payload=0):
        self.name = name
        self.payload = payload

class StateMachine(object):
    def __init__(self, states, events, transitionFunctions, initialState, terminalStates=[]):
        self._states = states
        self._events = events
        self._transitionFunctions = transitionFunctions

        self._eventMap  = enum.Enum('eventMap',  {event:ind for ind,event in enumerate(events)} )
        self._stateMap  = enum.Enum('stateMap',  {state:ind for ind,state in enumerate(states)} )

        self._terminalStates = terminalStates

        self._initialStates = [initialState]
        
        self.setState(initialState)

    def getNumStates(self):
        return len(self._states)

    def getNumEvents(self):
        return len(self._events)
    
    def setState(self, state):
        self._state = state
                    
    def assertState(self, state):
        return self._state == state

    def assertNotState(self, state):
        return not assertState( state)

    def isInitialState(self):
        return self.getState() in self._initialStates
    
    def isTerminalState(self):
        return self.getState() in self._terminalStates

    def getState(self):
        return self._state

    def registerTransitionFunctions(self, transitionFunctions):
        self._transitionFunctions = transitionFunctions

    def demux(self, event):
        return self._demuxMap[event]

    def processEvent( self, event, *a, **k):
        fn = self._transitionFunctions[self._eventMap[event.name].value][self._stateMap[self.getState()].value]
        return fn(event.payload, *a, **k)

class CliffWalkingStateMachine(StateMachine):

    MOVE_EVENT      =  'MOVE'
    VISIBLE_STATES  =  ['START', 'INTERIOR', 'OOB', 'EDGE', 'CLIFF', 'DESTINATION' ]
    INITIAL_STATE   =  'START'
    TERMINAL_STATES =  ['CLIFF', 'DESTINATION'] 
    REWARDS         =  [-1, -1, -1, -1, -100, -1]
    
    class MoveEvent(Event):
        # By convention:
        # payload = 0 ~ up
        # payload = 1 ~ right
        # payload = 2 ~ down
        # payload = 3 ~ left
        def __init__(self, payload):
            super( CliffWalkingStateMachine.MoveEvent, self).__init__(
                name=CliffWalkingStateMachine.MOVE_EVENT,
                payload=payload
            )
    
    def __init__(self,
                 nrows,
                 ncols,
                 states=VISIBLE_STATES,
                 rewards=REWARDS,
                 initialState=INITIAL_STATE,
                 terminalStates=TERMINAL_STATES
    ):
        tfn = self._transitionFunctions()
        self._initialState = initialState
        self._terminalStates = terminalStates
        self._visibleStates = states
        self._rewards = rewards

        self._rewardMap = enum.Enum('rewardMap', {state:reward for state,reward in zip(states, rewards) } )
        
        super(CliffWalkingStateMachine, self).__init__(states=self._visibleStates,
                                                       events=self._events(),
                                                       transitionFunctions=tfn,
                                                       initialState=self._initialState,
                                                       terminalStates=self._terminalStates,
        )
        
        self._nrows = nrows
        self._ncols = ncols

        self._x, self._y = self._goToStart()

    def show_transition(fn):
        @functools.wraps(fn)
        def wrapper(self, a, *args, **kwargs):
            if DEBUG_FLAG:
                x_pvs,y_pvs,state_pvs = self._x, self._y, self.getState()
            r = fn(self, a, *args, **kwargs)
            if DEBUG_FLAG:
                x,y,state = self._x, self._y, self.getState()
                app_log.info('ACTION: {}'.format(a))
                app_log.info('STATE TRANSITION: {0:<16} --> {1:<16}'.format(state_pvs,state ) )
                app_log.info('COORD TRANSITION: ({0:<2},{1:<2})          --> ({2:<2},{3:<2})'.format(x_pvs,y_pvs,x,y))
            return r
        return wrapper

    def _set_x(self, x):
        self._x = x

    def _set_y(self, y):
        self._y = y

    def set_coordinates(self, coords):
        self._x = coords[0]
        self._y = coords[1]
        
    def getNumRows(self):
        return self._nrows

    def getNumCols(self):
        return self._ncols

    def getNumActions(self):
        return 4

    def getNumEvents(self):
        return self.getNumActions()

    def getReward(self):
        return self._rewardMap[self.getState()].value
    
    @abc.abstractmethod
    def _isEdgeCoordinate(self, x, y):
        pass

    @abc.abstractmethod
    def _isCliffCoordinate(self, x, y):
        pass

    @abc.abstractmethod
    def _isStartCoordinate(self, x, y):
        pass

    @abc.abstractmethod
    def _isDestinationCoordinate(self, x, y):
        pass

    @functools.lru_cache(maxsize=512)
    def _isInteriorCoordinate(self, x, y):
        return not self._isEdgeCoordinate(x, y) and \
            not self._isCliffCoordinate(x, y) and \
            not self._isStartCoordinate(x, y) and \
            not self._isDestinationCoordinate(x, y) and \
            not self._isOOBCoordinate(x, y)
    
    @abc.abstractmethod
    def _isOOBCoordinate(self, x, y):
        pass

    def isEdge(self):
        return self._isEdgeCoordinate( self._x, self._y)

    def isCliff(self):
        return self._isCliffCoordinate( self._x, self._y)

    def isStart(self):
        return self._isStartCoordinate( self._x, self._y)

    def isDestination(self):
        return self._isDestinationCoordinate( self._x, self._y )

    def isOOB(self):
        return self._isOOBCoordinate(self._x, self._y)
    
    def isInterior(self):
        return self._isInteriorCoordinate(self._x, self._y)

    def _adjustCoordinates(self, action):
        if action == 0:
            self._y += 1
        elif action == 1:
            self._x += 1
        elif action == 2:
            self._y -= 1
        elif action == 3:
            self._x -= 1
    
    def _events(self):
        return [ self.MOVE_EVENT ]

    def _transitionFunctions(self):
        nop = lambda *a, **k: None
        mit = self._moveInterior
        med = self._moveEdge
        mst = self._moveStart

        TFn = [
            # START  INTERIOR OOB   EDGE   CLIFF  DESTINATION
            [ mst,   mit,     nop,  med,    nop,   nop        ] # MOVE
            ]

        del mit
        del med
        del mst

        return TFn
    
    def _coordinates(self):
        return (self._x, self._y)
    
    def processEvent(self, action, *a, **k):
        event = self.MoveEvent(payload=action)
        super(CliffWalkingStateMachine, self).processEvent(event, *a, **k)

    @show_transition
    def _moveInterior(self, a):

        self.assertState('INTERIOR')

        # Possibility that we hit a wall
        x_pvs,y_pvs,state_pvs = self._x, self._y, self.getState()
        
        self._adjustCoordinates(a)

        if self.isCliff():
            self.setState('CLIFF')
        elif self.isEdge():
            self.setState('EDGE')
        elif self.isOOB():
            self._x, self._y = x_pvs,y_pvs
            self.setState(state_pvs)
        elif self.isDestination():
            self.setState('DESTINATION')

    @show_transition
    def _moveEdge(self, a):

        self.assertState('EDGE')

        # Possibility that we hit a wall
        x_pvs,y_pvs,state_pvs = self._x, self._y, self.getState()
        
        self._adjustCoordinates(a)

        if self.isCliff():
            self.setState('CLIFF')
        elif self.isOOB():
            self._x, self._y = x_pvs,y_pvs
            self.setState(state_pvs)
        elif self.isDestination():
            self.setState('DESTINATION')
        elif not self.isEdge():
            self.setState('INTERIOR')            

    @show_transition
    def _moveStart(self, a):

        self.setState('START')
        
        if a == 0:
            self._adjustCoordinates(a)
            self.setState('EDGE')

class learner_decorator(object):
    def __init__(self, decorated, alpha, gamma, epsilon=.1):
        self._fsm = decorated

        self._Q = np.zeros((self._fsm.getNumCols(),
                            self._fsm.getNumRows(),
                            self._fsm.getNumEvents()), dtype=np.float)

        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon

    @property
    def _x(self):
        return self._fsm._x

    @property
    def _y(self):
        return self._fsm._y

    def _goToStart(self):
        return self._fsm._goToStart()

    def isInitialState(self):
        return self._fsm.isInitialState()

    def isTerminalState(self):
        return self._fsm.isTerminalState()

    def getState(self):
        return self._fsm.getState()
        
    def explore_exploit(self):
        if self._fsm.isStart():
            a = 0
            return a

        eps = np.random.uniform()

        if (eps < self._epsilon):
            a = random_action()
        else:
            a = self.exploit()
        return a

    def exploit(self):
        if self._fsm.isStart():
            a = 0
            return a

        x,y = self._x, self._y
        if x == self._fsm.DESTINATION_COORDINATES[0] and \
           y == 1 + self._fsm.DESTINATION_COORDINATES[1]:
            a = 2
            return a

        if self._fsm.isOOB():
            print('OOB!')
            
        if self._fsm.isDestination():
            print('DESTINATION!')

        if self._fsm.isInterior() or self._fsm.isEdge():
            a = np.argmax(self._Q[x,y,:])

        return a

    def bellman(self, Qs1a1, x, y, a):
        if self.isInitialState():
            return self._Q
        if self._fsm.isDestination():
            return self._Q

        reward = self._fsm.getReward()
        self._Q[x,y,a] = self._Q[x,y,a] + self._alpha * (reward + self._gamma * Qs1a1 - self._Q[x,y,a])
        return self._Q

    def plot_Q(self, filename='Q_hat.png'):
        import matplotlib.pyplot as plt

        numEvents = self._fsm.getNumEvents()
        for i in range(numEvents):
            plt.subplot(numEvents,1,i+1)
            plt.imshow(self._Q[:,:,i])
            plt.axis('off')
            plt.colorbar()
            if (i == 0):
                plt.title('Q-north')
            elif (i == 1):
                plt.title('Q-east')
            elif (i == 2):
                plt.title('Q-south')
            elif (i == 3):
                plt.title('Q-west')
        plt.savefig(filename)
        plt.clf()
        plt.close()
         
        
class SARSA_decorator(learner_decorator):

    def move(self, action, *a, **k):
        x_pvs, y_pvs, state_pvs = self._x, self._y, self.getState()
        
        self._fsm.processEvent( action, *a, **k)

        state = self.getState()
        
        if state == 'DESTINATION':
            # print('REACHED DESTINATION!')
            Qs1a1 = 0.0
            self._Q = self.bellman(Qs1a1, x_pvs, y_pvs, action)
        elif state == 'CLIFF':
            Qs1a1 = 0.0
            self._Q = self.bellman(Qs1a1, x_pvs, y_pvs, action)
        elif state == 'OOB':
            print('OOB - SHOULD NOT REACH THIS STATE')
            self._fsm_x,self._fsm._y = x_pvs ,y_pvs
            self.setState(state_pvs)
        else: # [ 'START", 'INTERIOR', 'EDGE' ], i.e. ~terminal
            a = self.explore_exploit()
            if self.getState() == 'START':
                Qs1a1 = 0.
            else:
                Qs1a1 = self._Q[self._x, self._y, a]
            self._Q = self.bellman(Qs1a1, x_pvs, y_pvs, action)
        return a
                    
class Q_learner_decorator(learner_decorator):

    def _max_Q(self):
        a = np.argmax(self._Q[self._x,self._y,:])
        return self._Q[self._x, self._y, a], a

    def move(self, action, *a, **k):
        x_pvs, y_pvs, state_pvs = self._x, self._y, self.getState()
        
        self._fsm.processEvent( action, *a, **k)

        state = self.getState()
        
        if state == 'DESTINATION':
            # print('REACHED DESTINATION!')
            Qs1a1 = 0.0
            self._Q = self.bellman(Qs1a1, x_pvs, y_pvs, action)
        elif state == 'CLIFF':
            Qs1a1 = 0.0
            self._Q = self.bellman(Qs1a1, x_pvs, y_pvs, action)
        elif state == 'OOB':
            print('OOB - SHOULD NOT REACH THIS STATE')
            self._fsm_x,self._fsm._y = x_pvs ,y_pvs
            self.setState(state_pvs)
        else: # [ 'START", 'INTERIOR', 'EDGE' ], i.e. ~terminal
            if self.getState() == 'START':
                Qs1a1 = 0.
            else:
                Qs1a1,a = self._max_Q()
            self._Q = self.bellman(Qs1a1, x_pvs, y_pvs, action)
        return a
        
        
