import gym, gym_cartpole_swingup
import numpy as np


class InvertedPendulum(object):

    def __init__(self, max_eps, safety_mode=False, init_p=30, reward=0):

        # making cart pole swing up as the environment
        # strictly positive reward
        self._env = gym.make("CartPoleSwingUp-v1", origin=safety_mode)
        self._env._max_episode_steps = max_eps

        # initial state
        # [x_pos, x_dot, np.cos(theta), np.sin(theta), theta_dot]
        self.init_state = self._env.reset()
        self.current_state = self.init_state

        self.init_action = np.zeros(self._env.action_space.shape)

        # # code for checking the env.
        # env = gym.make("CartPoleSwingUp-v1", origin=True)

        # for i in range(100):
        #     env.reset()
        #     done = False
        #     while not done:
        #         action = env.action_space.sample()
        #         obs, rew, done, info = env.step(action)
        #         env.render()

        # mode
        self.safety_mode = safety_mode

        # reward and penalty
        self.penalty = init_p
        self.r = reward

        # assume no grund truth
        self.ground_truth = None

        self.a_grid = np.linspace(-1, 1, 161)
    
    def render(self):
        self._env.render()

    def update_penalty(self, p):
        self.penalty = p
    
    def reset(self, random_init=False, safety_supervisor=None):

        # in the _env, reset pole point to the origin (upside + noise)
        self.init_state = self._env.reset()
        self.current_state = self.init_state
        return self.current_state

    def step(self, u):

        next_state, reward, failure, _ = self._env.step(u)
        self.current_state = next_state

        # failure occur only if it happens within max_eps
        reach_max_eps = self._env._max_episode_steps == self._env._elapsed_steps
        if reach_max_eps and (abs(self._env.state.x_pos) <= self._env.params.x_threshold and abs(self._env.state.theta) <= self._env.params.theta_threshold):
            failure = False 

        # failure occurs if bool(abs(state.x_pos) > self.params.x_threshold=2.4)
        # or bool(abs(state.theta) > self.params.theta_threshold=pi/4)
        if self.safety_mode:
            return next_state, reward - failure * self.penalty, self.indicator(failure), failure, False
        else:
            return next_state, - failure * self.penalty, self.indicator(failure), failure, False

    def indicator(self, fail):
        return -1 if fail else 0

    def map_action(self, raw_action):
        """Map raw action to the interval [-1, 1]

        Args:
            raw_action (ndarray): raw action of the dynamics
        """
        # map [a, b] onto [c, d]
        # c + (d-c)/(b-a) * (raw_action-a)
        # in our case, map [-1, 1] onto [-1, 1]

        return raw_action
    
    def unmap_action(self, squash_action):
        """Map squash action to its original space

        Args:
            squash_action (ndarray): squash action in [-1. 1]
        """       
        # map [-1, 1] onto [-1, 1]
        return squash_action
    
    def sample_action(self):
        return self._env.action_space.sample()

    def failure_state(self):

        state = self.current_state
        theta = self._env.state.theta

        # assign closest failure state
        if state[0] > 0:
            limit = self._env.params.x_threshold
        else:
            limit = -self._env.params.x_threshold
        if theta > 0:
            theta_limit = self._env.params.theta_threshold
        else:
            theta_limit = -self._env.params.theta_threshold
        
        state[0] = limit
        state[2] = np.cos(theta_limit)
        state[3] = np.sin(theta_limit)
        self.current_state = state
        
        return state
    