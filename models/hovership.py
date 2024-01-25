# Copyright (c) 2019 Max Planck Gesellschaft, Steve Heim and Alexander von Rohr

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# This piece of code is adapted version from Steve Heim and Alexander von Rohr

import numpy as np
import scipy.integrate as integrate
import pickle
import itertools
import torch

'''
A spaceship attempting to reconnoitre the surface of a planet.
However, the planet has an unusual gravitational field... getting too close to
the surface may result in getting sucked in, with no escape!
'''

class Hovership(object):
    def __init__(self, safety_mode=False, init_p=10, reward=0, goal_state=None, goal_state_reward=0):
        
        # initial state and action
        self.init_state = np.array([1.8])
        self.current_state = self.init_state
        self.init_action = np.array([.6])

        # mode
        self.safety_mode = safety_mode

        # reward and penalty
        self.penalty = init_p
        self.r = reward

        # goal
        self.goal_state = goal_state
        self.goal_state_reward = goal_state_reward

        # initial parameters
        self.p = {'n_states': 1,
                  'base_gravity': 0.1,
                  'gravity': 1,
                  'max_thrust': 0.8,
                  'min_thrust': 0,
                  'ceiling': 2,
                  'control_frequency': 1,  # hertz
                 }
        self.THRUST = np.min([self.p['max_thrust'], self.init_action[0]])
        self.BASE_GRAVITY = self.p['base_gravity']
        self.GRAVITY = self.p['gravity']
        self.MAX_TIME = 1.0/self.p['control_frequency']
        self.CEILING = self.p['ceiling']

        # for computing resulting q functions
        self.s_grid = np.linspace(-0.0, self.p['ceiling'], 201)
        # remember to map the action space
        self.a_grid = self.map_action(np.linspace(0.0, self.p['max_thrust'], 161))

        # assume no grund truth
        self.ground_truth = None


    def load_ground_truth(self):
        infile = open("ground_truth/hovership.pickle", 'rb')
        self.ground_truth = pickle.load(infile)
        infile.close()

    def update_penalty(self, p):
        self.penalty = p
    
    def reset(self, random_init=False, safety_supervisor=None):
        if random_init == False or safety_supervisor is None:
            self.current_state = self.init_state
            return self.current_state
        else:
            Q_values = np.zeros((self.s_grid.size*self.a_grid.size, 1))
            combinations = np.array(list(itertools.product(self.s_grid, self.a_grid)))
            grid_states =  torch.as_tensor(combinations[:,0].reshape(-1, 1), dtype=torch.float32)
            grid_actions =  torch.as_tensor(combinations[:,1].reshape(-1, 1), dtype=torch.float32)
            safety_supervisor.set_evaluation()
            with torch.no_grad():
                Q_values = torch.min(safety_supervisor.actor_critic.q1(grid_states, grid_actions), safety_supervisor.actor_critic.q2(grid_states, grid_actions)).numpy()
            safety_supervisor.set_training()

            # get safe state-action pairs index
            Q_values = Q_values.reshape(self.s_grid.size, self.a_grid.size)
            safe_pairs_index = np.array(Q_values > 0, dtype=bool)

            # if safe actions don't exist, set original initial state
            if not safe_pairs_index.any():
                self.current_state = self.init_state
                return self.current_state
            # else pick one of them
            else:
                idxs = np.argwhere(safe_pairs_index==True)
                idx = np.random.choice(len(idxs))

                self.current_state = self.s_grid[idxs[idx][0]].flatten()
                return self.current_state

    def step(self, u):
        '''
        The transition map of your system.
        inputs:
        u: action (control input)
        outputs:
        x: ndarray state at next iteration
        reward: int with -1 for failure, otherwise 0
        failed: boolean indicating if the system has failed
        terminate: boolean var. indicating if the system is at the goal state
        '''

        # * we first check if the state is already in the failure state
        # * this can happen if the initial state of a new sequence is chosen poorly
        # * this can also happen if the failure state depends on the action
        x = self.current_state
        if self.check_failure(x):
            self.current_state = np.zeros((1,))
            return self.current_state, self.reward(is_fail=True), self.indicator(True), True, False

        self.THRUST = np.min([self.p['max_thrust'], u[0]])
        
        # * for convenience, we define the continuous-time dynamics, and use
        # * scipy.integrate to solve this over one control time-step (MAX_TIME)
        # * what you put in here can be as complicated as you like.
        def continuous_dynamics(t, x):
            grav_field = np.max([0, np.tanh(0.75*(self.CEILING - x[0]))])*self.GRAVITY
            f = - self.BASE_GRAVITY - grav_field + self.THRUST
            return f

        def ceiling_event(t, x):
            '''
            Event function to detect the body reaching the ceiling
            '''
            return x-self.CEILING
        ceiling_event.terminal = True
        ceiling_event.direction = 1

        sol = integrate.solve_ivp(continuous_dynamics, t_span=[0, self.MAX_TIME], y0=x,
                                events=ceiling_event)

        # ceiling_event sometime ends integration over CEILING, and sometimes
        # under. This creates discretization problems with computing viability
        # so let's manually cap it at the CEILING.
        if sol.y[:, -1] > self.CEILING:
            sol.y[:, -1] = self.CEILING

        # * we return the final
        if self.check_failure(sol.y[:, -1]):
            self.current_state = np.zeros((1,))
            return self.current_state, self.reward(is_fail=True), self.indicator(True),  True, False
        else:
            self.current_state = sol.y[:, -1]
            # if it is close enough then we terminate
            if self.goal_state is None:
                return sol.y[:, -1], self.reward(is_fail=False), self.indicator(False), False, False
            else:
                goal = True if np.allclose(self.current_state, self.goal_state, rtol=1e-02) else False
                return sol.y[:, -1], self.reward(is_fail=False), self.indicator(False), False, goal

    def map_action(self, raw_action):
        """Map raw action to the interval [-1, 1]

        Args:
            raw_action (ndarray): raw action of the dynamics
        """
        # map [a, b] onto [c, d]
        # c + (d-c)/(b-a) * (raw_action-a)
        # in our case, map [min_thrust, max_thrust] onto [-1, 1]

        return -1 + 2/(self.p['max_thrust']-self.p['min_thrust']) * (raw_action-self.p['min_thrust'])
    
    def unmap_action(self, squash_action):
        """Map squash action to its original space

        Args:
            squash_action (ndarray): squash action in [-1. 1]
        """       
        # map [-1, 1] onto [min_thrust, max_thrust]
        return self.p['min_thrust'] + (self.p['max_thrust']-self.p['min_thrust'])/2 * (squash_action+1)

    def sample_action(self):
        l = self.p['min_thrust']
        u = self.p['max_thrust']
        a = np.random.rand(*self.init_action.shape)
        return self.map_action(l + (u-l) * a)
    
    def failure_state(self):
        return np.zeros((1,))

    def check_failure(self, x):
        '''
        Check if a state-action pair is in the failure set.
        inputs:
        x: ndarray of the state
        outputs:
        failed: bool indicating true if the system failed, false otehrwise

        '''

        # * For this example, the system has failed if it ever hits the ground
        if x[0] < 0:
            return True
        else:
            return False
    
    def reward(self, is_fail):
        """reward function

        """        

        if self.goal_state == None:
            return self.r + self.penalty * -1 if is_fail else self.r
        else:
            # if the current state is close to the goal state then give reward
            # otherwise the reward is zero
            if np.allclose(self.current_state, self.goal_state, rtol=1e-02):
                # # use normalized distance between goal and current state to make distance in [0, 1]
                # # such that 0.001^(distance) is also in (0, 1], 1 for d == 0, close to 0 for d == 1
                # # map [a, b] onto [c, d]
                # # c + (d-c)/(b-a) * (state-a)
                # # in our case, map [0, ceiling] onto [0, 1]
                # d = 0 + (1-0)/(self.p['ceiling']-0) * (np.sqrt((self.current_state - self.goal_state)**2) - 0)
                # # r reaches max as 0.001^(d) == 1 (d == 0) and r decreases as 0.001^(d) -> 0 (d -> 1)
                # r = self.goal_state_reward  * np.power(0.001, d)
                r = self.goal_state_reward
                return r + self.penalty * -1 if is_fail else r
            else:
                return self.r + self.penalty * -1 if is_fail else self.r

    def indicator(self, is_fail):
        return -1 if is_fail else 0