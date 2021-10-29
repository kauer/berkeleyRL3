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


import mdp, util
from learningAgents import ValueEstimationAgent
import copy


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0

        for i in range(iterations):
            old_v = copy.copy(self.values)
            for state in self.mdp.getStates():
                actions = self.mdp.getPossibleActions(state)

                if len(actions) == 0:  # prevents error in max
                    continue

                self.values[state] = max([  # max over actions
                    sum([  # sum over next states
                        prob * (self.mdp.getReward(state, a, sprime) + discount * old_v[sprime])
                        for (sprime,prob) in self.mdp.getTransitionStatesAndProbs(state, a)
                    ])
                    for a in actions
                ])

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
        # aliases:
        s = state
        a = action
        d = self.discount
        return sum([
            prob * (self.mdp.getReward(s, a, sprime) + d * self.values[sprime])
            for (sprime, prob) in self.mdp.getTransitionStatesAndProbs(s, a)
        ])

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        possible_actions = self.mdp.getPossibleActions(state)
        if len(possible_actions) == 0:
            return None

        # performs the argmax (not very pythonic)
        max_action = possible_actions[0]
        max_q = self.computeQValueFromValues(state, possible_actions[0])
        for action in self.mdp.getPossibleActions(state):
            q = self.computeQValueFromValues(state, action)
            if q > max_q:
                max_action = action
                max_q = q

        return max_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
