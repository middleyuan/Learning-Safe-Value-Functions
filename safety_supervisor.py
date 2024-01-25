import torch
import numpy as np
from network import network, DeepQNetwork, SquashedGaussianNetwork, ActorCritic
from matplotlib import pyplot as plt
from copy import deepcopy
import matplotlib.animation as animation
import logging
import os
import pandas as pd

class SafetySupervisor(object):
    def __init__(self, env, env_name, actor_critic=None, from_saved_model=''):
        """constructor for safety supervisor

        Args:
            env (environment): dynamics
            actor_critic (ActorCritic, optional): object of actor cirtic. Defaults to None.
            from_saved_model (str, optional): path of pre-trained model. Defaults to ''.

        Raises:
            ValueError: occur when neither object nor path are given
        """        

        # TODO: update safety supervisor
        # TODO: replay buffer

        self.env = env
        self.env_name = env_name
        self.from_saved_model = from_saved_model

        if actor_critic == None:
            if from_saved_model == '':
                raise ValueError('Please give either the object of actor-critic or the path for saved model')
            
            # init actor-critic
            action_limit = 1
            self.actor_critic = ActorCritic(env.init_state.shape, env.init_action.shape, action_limit)

            # load models
            with open(from_saved_model + "/q1.pt", "rb") as f:
                self.actor_critic.q1.load_state_dict(torch.load(f))
            with open(from_saved_model + "/q2.pt", "rb") as f:
                self.actor_critic.q2.load_state_dict(torch.load(f))
            with open(from_saved_model + "/pi.pt", "rb") as f:
                self.actor_critic.pi.load_state_dict(torch.load(f))
        else:
            self.actor_critic = actor_critic

    def set_evaluation(self):
        """set to the evaluation mode
        """       
        
        # model evaluation
        self.actor_critic.q1.eval()
        self.actor_critic.q2.eval()
        self.actor_critic.pi.eval()

    def set_training(self):
        """set to the training mode
        """        

        # model evaluation
        self.actor_critic.q1.train()
        self.actor_critic.q2.train()
        self.actor_critic.pi.train()

    def evaluate_state_action(self, state, action):
        """check if state-action pair is safe 

        Args:
            state (ndarray): state of dynamics
            action (ndarray): action of agent

        Returns:
            True or False: if safe, return True
        """        

        self.set_evaluation()

        state = torch.as_tensor(state.reshape(1, -1), dtype=torch.float32)
        action = torch.as_tensor(action.reshape(1, -1), dtype=torch.float32)

        with torch.no_grad():
            Q_values = torch.min(self.actor_critic.q1(state, action), self.actor_critic.q2(state, action)).numpy()

        return Q_values > 0

    def safe_actions(self, state):
        """get safe actions given the state

        Args:
            state (ndarray): the state of the dynamics
        
        Returns:
            action: if there are possible actions, return random one; else, return the safest one
            safest: boolean variable is True when returning safest action in case of empty action set
        """ 

        # TODO: action may have dimension > 1
        self.set_evaluation()
        safest = False
        state = state.reshape(1, -1)
        dim = state.shape[1]
        Q_values = np.zeros((self.env.a_grid.size, 1))

        # replicate state
        states = np.tile(state, (self.env.a_grid.size, 1))
        states =  torch.as_tensor(states, dtype=torch.float32)
        actions =  torch.as_tensor(self.env.a_grid.reshape(-1, 1), dtype=torch.float32)
        with torch.no_grad():
            Q_values = torch.min(self.actor_critic.q1(states, actions), self.actor_critic.q2(states, actions)).numpy()

        Q_values = Q_values.reshape(state.shape[0], self.env.a_grid.size)

        # get safe actions index
        safe_actions_index = np.array(Q_values > 0, dtype=bool)

        # if safe actions don't exist, pick safest one
        if not safe_actions_index.any():
            idx = np.argmax(Q_values)
            safest = True
        # else pick one of them
        else:
            idxs = np.argwhere(safe_actions_index==True)
            idx = np.random.choice(idxs[:, 1])

        self.set_training()

        # add some noise whose magnitude is smaller than the bin size
        bin_size = (1-(-1))/self.env.a_grid.size
        while(True):
            if idx < self.env.a_grid.size and idx > 0:
                action = actions[idx, :] + np.random.rand() * bin_size
                # clip the limits
                action = torch.where(action > 1, torch.as_tensor(1, dtype=torch.float32), action)
                action = torch.where(action < -1, torch.as_tensor(-1, dtype=torch.float32), action)
            else:
                action = actions[idx, :]

            # final check to make sure the state action pair is safe given that the action set is not empty
            # catch the safety exception for adding noise
            if self.evaluate_state_action(state, action) == False and safest != True:
                bin_size = bin_size * 0.5
            else:
                break

        return action.numpy(), safest

    def evaluate_agent(self, evaluation_steps, visualize=True):

        # perform the first interaction with env.
        # init state
        env = deepcopy(self.env)
        s = env.current_state
        # randomly sample 'safe' action
        a, safest = self.safe_actions(s)
        # observe the next state
        next_state, reward, _, failure, _ = env.step(env.unmap_action(a))

        # array with element indicating if action is safest
        self.is_empty_array = []
        self.failure_time = 0

        if visualize:
            if self.env_name == 'hovership':
                # visualization
                v = Visualization(env, self.actor_critic, s, a, next_state, reward, failure, safest)
                v.visualize(evaluation_steps)
                # get trajectory
                self.trajectory = v.visited_points
                self.is_empty_array = v.is_empty_array
            elif self.env_name == 'inverted_pendulum':
                step = 1
                while step < evaluation_steps:
                    if safest or failure:
                        env.reset()
                        s = env.init_state
                        a, safest = self.safe_actions(s)
                        next_state, reward, _, failure, _ = env.step(env.unmap_action(a))
                    else:
                        s = next_state
                        a, safest = self.safe_actions(s)
                        next_state, reward, _, failure, _ = env.step(env.unmap_action(a))
                    env.render()
                    step += 1
                    if failure:
                        self.failure_time += 1
                print('Failure rate: ', self.failure_time/evaluation_steps)
        else:
            self.trajectory = np.array([[s, a]])
            self.is_empty_array += [True] if safest else [False]
            step = 1
            while step < evaluation_steps:
                if safest or failure:
                    env.reset()
                    s = env.init_state
                    a, safest = self.safe_actions(s)
                    next_state, reward, _, failure, _ = env.step(env.unmap_action(a))
                else:
                    s = next_state
                    a, safest = self.safe_actions(s)
                    next_state, reward, _, failure, _ = env.step(env.unmap_action(a))
                step += 1
                self.trajectory = np.concatenate((self.trajectory, np.array([[s, a]])), axis=0)
                self.is_empty_array += [True] if safest else [False]
                if failure:
                        self.failure_time += 1
            print('Failure rate: ', self.failure_time/evaluation_steps)
        
        # check how often the learned q functions return empty action set
        if len(self.is_empty_array) > 0:
            logging.info('Frequency of empty action set: %s', sum(self.is_empty_array)/len(self.is_empty_array))
            self.save(self.is_empty_array, 'is_empty_set')
            self.save(self.trajectory.reshape(evaluation_steps, -1), 'trajectory')

    def save(self, data, name):
        file = pd.DataFrame(data)
        file.to_csv(os.path.join(self.from_saved_model, name + '.csv'), header=False)


class Visualization(SafetySupervisor):
    
    def __init__(self, env, actor_critic, s, a, next_s, r, failure, safest, animation=True):
        super().__init__(env, actor_critic)

        self.s = s
        self.a = a
        self.next_state = next_s
        self.r = r
        self.failure = failure
        self.is_empty_array = []
        self.safest = safest

        # visited points
        self.visited_points = np.array([[self.s, self.a]])
        self.is_empty_array += [True] if safest else [False]

        # First set up the figure, the axis, and the plot element we want to animate
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(0, self.env.p['max_thrust']), ylim=(0, self.env.p['ceiling']))
        self.line, = self.ax.plot([], [], 'o-', lw=2)
        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)

    def visualize(self, evaluation_steps):

        def init():
            self.line.set_data([self.a], [self.s])
            self.time_text.set_text('')
            return self.line, self.time_text, 

        def animate(i):
            
            if self.safest:
                self.env.reset()
                self.s = self.env.init_state
                self.a, self.safest = self.safe_actions(self.s)
                # observe the next state
                self.next_state, reward, _, self.failure, _ = self.env.step(self.env.unmap_action(self.a))
            else:
                self.s = self.next_state
                # randomly sample 'safe' action
                self.a, self.safest = self.safe_actions(self.s)
                # observe the next state
                self.next_state, reward, _, self.failure, _ = self.env.step(self.env.unmap_action(self.a))

            if self.failure:
                # TODO: save to 'transfer learning' folder
                # TODO: update threshold
                # file = pd.DataFrame(self.total_rewards)
                # file.to_csv(os.path.join(self.results_folder, 'reward.csv'), header=False)
                raise ValueError('Safety supervisor fails')
            # visited points
            self.visited_points = np.concatenate((self.visited_points, np.array([[self.s, self.a]])), axis=0)
            self.is_empty_array += [True] if self.safest else [False]
            
            self.line.set_data(self.env.unmap_action(self.a), self.s)
            self.time_text.set_text('step = %d' % (i+1))

            return self.line, self.time_text, 
        
        anim = animation.FuncAnimation(self.fig, animate, init_func=init, frames=evaluation_steps-1, blit=True, repeat=False)

        plt.show()

