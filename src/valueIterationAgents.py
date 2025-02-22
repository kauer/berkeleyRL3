# valueIterationAgents.py
# -----------------------
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


import mdp, util, math

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        for state in self.mdp.getStates():
            self.values[state] = 0.0
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                self.values[state, action] = 0.0

        for it in range(iterations):
            new_values = util.Counter()
            for state in self.mdp.getStates():
                max = -math.inf
                if(self.mdp.isTerminal(state)):
                    self.values[state] = self.values[state]
                else:
                    actions = self.mdp.getPossibleActions(state)
                    for action in actions:
                        sum = 0.0
                        possible_states = self.mdp.getTransitionStatesAndProbs(state, action)
                        for next_state in possible_states:
                            sum = sum + next_state[1] * (self.values[next_state[0]])
                        if (sum > max):
                            max = sum
                    new_values[state] = self.mdp.getReward(state, 0, 0) + discount*max
            for state in self.mdp.getStates():
                self.values[state] = new_values[state]
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    self.values[state, action] = new_values[state, action]



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        """action = best action"""
        """Q[state,action] <- 0.5*Q[state,action] + 0.5*(self.mdp.getReward(state', 0, 0) + discount*(maxQ(state',action')))"""

        sum = 0.0
        possible_states = self.mdp.getTransitionStatesAndProbs(state, action)
        for next_state in possible_states:
            sum = sum + next_state[1]*self.values[next_state[0]]*self.discount

        return self.mdp.getReward(state, 0, 0) + sum



    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        max = -math.inf
        best_action = None
        actions = self.mdp.getPossibleActions(state)
        if(len(actions) == 0):
            return None

        for action in actions:
            sum = 0.0
            possible_states = self.mdp.getTransitionStatesAndProbs(state,action)
            for next_state in possible_states:
                sum = sum + next_state[1]*(self.values[next_state[0]])
            if(sum > max):
                max = sum
                best_action = action

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
