import os
from datetime import datetime
from models.hovership import Hovership
from models.inverted_pendulum import InvertedPendulum
from replay_buffer import ReplayBuffer
from network import network, DeepQNetwork, SquashedGaussianNetwork, ActorCritic
from copy import deepcopy
import torch
import numpy as np
from matplotlib import pyplot as plt
import itertools
from torch.optim import Adam
import pandas as pd
from logging_class import Logger
import logging

class SoftActorCritic(object):

    def __init__(self, env, q_plot_every, mem_size, seed, gamma, alpha, learning_rate, polyak,
                 num_test_episodes, pi_update_every, max_ep_len, batch_size, epochs,
                 steps_per_epoch, update_every, update_after, start_steps, penalty_autotune,
                 init_penalty, reward, goal_state, goal_state_reward, learning_with_safety,
                 virtual_step, pretrained_actor_critic, pretrained_actor_critic_target):

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # saving path
        if learning_with_safety:
            self.results_folder = 'tfl/' + env + '/' + str(datetime.now().strftime("%Y%m%d-%H%M%S.%f"))
        else:
            self.results_folder = 'svf/' + env + '/' + str(datetime.now().strftime("%Y%m%d-%H%M%S.%f"))
        os.makedirs(self.results_folder, exist_ok=False)

        # process dynamics name
        env = env.split('-')[0].strip()

        # init logger
        self.logger = Logger(self.results_folder)
        self.logger.start_log()
        logging.info(self.results_folder)

        # init env.
        self.env_name = env
        self.max_ep_len = max_ep_len
        self.init_env(env, learning_with_safety, init_penalty, reward, goal_state, goal_state_reward)

        self.test_env = None

        # init replay buffer
        self.buffer = ReplayBuffer(mem_size=mem_size, state_shape=self.env.init_state.shape, action_shape=self.env.init_action.shape)

        self.learning_with_safety = learning_with_safety

        # reward
        self.r = reward
        # initial penalty
        self.init_p = init_penalty

        # init penalty
        self.penalty_autotune = penalty_autotune
        self.penalty = torch.tensor(self.init_p, dtype=torch.float32)
        self.penalty.requires_grad = True if penalty_autotune else False
        # penalty optimizers
        self.p_optimizer = Adam([self.penalty], lr=learning_rate*1.5)

        self.max_alpha_entropy = None
        self.max_reward = None


        # init parameters
        self.seed = seed
        self.gamma = gamma
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.polyak = polyak
        self.num_test_episodes = num_test_episodes
        self.q_plot_every = q_plot_every
        self.pi_update_every = pi_update_every
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = epochs * steps_per_epoch
        self.update_every = update_every
        self.update_after = update_after
        self.start_steps = start_steps
        self.virtual_step = virtual_step

        # init actor critic
        # TODO: every env may have different action limit
        if pretrained_actor_critic is not None and pretrained_actor_critic_target is not None:
            action_limit = 1
            self.actor_critic = pretrained_actor_critic
            self.actor_critic_target = pretrained_actor_critic_target
            self.pretrain_networks = True
        else:
            action_limit = 1
            self.actor_critic = ActorCritic(self.env.init_state.shape, self.env.init_action.shape, action_limit)
            self.actor_critic_target = deepcopy(self.actor_critic)
            self.pretrain_networks = False

        # Freeze target networks
        for p in self.actor_critic_target.parameters():
            p.requires_grad = False
        
        # List of parameters for both Q-networks
        self.q_params = itertools.chain(self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters())

        # actor_critic optimizers
        self.pi_optimizer = Adam(self.actor_critic.pi.parameters(), lr=learning_rate)
        # q_params contain params of q1 and q2
        self.q_optimizer = Adam(self.q_params, lr=learning_rate)

    def init_env(self, env='hovership', learning_with_safety=False, init_p=1, reward=0, goal_state=None, goal_state_reward=0):
        if env == 'hovership':
            self.env = Hovership(learning_with_safety, init_p, reward, goal_state, goal_state_reward)
            self.env.load_ground_truth()
        elif env == 'inverted_pendulum':
            self.env = InvertedPendulum( self.max_ep_len, learning_with_safety,  init_p, reward)
        else:
            raise ValueError('Unknown environment')

    def compute_q_loss(self, states, actions, rewards, next_states, is_terminal):
        
        # foward pass
        q1 = self.actor_critic.q1(states, actions)
        q2 = self.actor_critic.q2(states, actions)

        # disabled gradient calculation
        with torch.no_grad():
            # actions from current policy
            next_actions, logp_next_actions = self.actor_critic.pi(next_states)

            # target Q-values
            q1_pi_target = self.actor_critic_target.q1(next_states, next_actions)
            q2_pi_target = self.actor_critic_target.q2(next_states, next_actions)
            q_pi_target = torch.min(q1_pi_target, q2_pi_target)

            backup = rewards + self.gamma * (1 - is_terminal) * (q_pi_target - self.alpha * logp_next_actions)

        # MSE loss for Q functions
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q, logp_next_actions
    
    def compute_pi_loss(self, states):
        next_actions, logp_next_actions = self.actor_critic.pi(states)
        q1_pi = self.actor_critic.q1(states, next_actions)
        q2_pi = self.actor_critic.q2(states, next_actions)
        q_pi = torch.min(q1_pi, q2_pi)

        # policy loss
        loss_pi = (self.alpha * logp_next_actions - q_pi).mean()

        return loss_pi

    def update(self, states, actions, rewards, indicators, next_states, is_terminal, is_update_pi):

        # do one gradient descent step for Q networks
        self.q_optimizer.zero_grad() # zero out gradients
        loss_q, logp_next_actions = self.compute_q_loss(states, actions, rewards, next_states, is_terminal)
        loss_q.backward()
        self.q_optimizer.step()

        # record the max alpha * entropy term
        if self.max_alpha_entropy is None:
            self.max_alpha_entropy = (-logp_next_actions * self.alpha).max()
        else:
            if self.max_alpha_entropy < (-logp_next_actions * self.alpha).max():
                self.max_alpha_entropy = (-logp_next_actions * self.alpha).max()
        
        # record the max reward
        if self.max_reward is None:
            self.max_reward = rewards.max()
        else:
            if self.max_reward < rewards.max():
                self.max_reward = rewards.max()

        # Freeze Q networks so no computational effort wastes during the policy learning step
        for p in self.q_params:
            p.requires_grad = False

        # do one gradient descent step for policy network
        if is_update_pi:
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_pi_loss(states)
            loss_pi.backward()
            self.pi_optimizer.step()

        # update penalty
        is_p_enough = (self.max_reward + self.max_alpha_entropy) < self.penalty
        # for testing because of time-to-failure is hard to estimate
        is_p_enough = False
        if self.penalty_autotune and is_p_enough == False:
            self.p_optimizer.zero_grad()
            p_loss = (self.penalty * indicators).mean()
            p_loss.backward()
            self.p_optimizer.step()
            self.env.update_penalty(self.penalty.item())

        # Unfreeze Q networks in the next iteration
        for p in self.q_params:
            p.requires_grad = True

        # update target networks by polyak averaging
        with torch.no_grad():
            for p, p_target in zip(self.actor_critic.parameters(), self.actor_critic_target.parameters()):
                # in-place operations "mul_", "add_" to update target params
                p_target.data.mul_(self.polyak)
                p_target.data.add_((1 - self.polyak) * p.data)

    def temporal_q_functions(self, s_grid, a_grid):
        # remember to feed the raw action to tanh function
        Q_values = np.zeros((s_grid.size*a_grid.size, 1))
        combinations = np.array(list(itertools.product(s_grid, a_grid)))
        grid_states =  torch.as_tensor(combinations[:,0].reshape(-1, 1), dtype=torch.float32)
        grid_actions =  torch.as_tensor(combinations[:,1].reshape(-1, 1), dtype=torch.float32)
        with torch.no_grad():
            Q_values = torch.min(self.actor_critic.q1(grid_states, grid_actions), self.actor_critic.q2(grid_states, grid_actions)).numpy()
        
        return Q_values.reshape(s_grid.size, a_grid.size)

    def train_for_svf(self):

        # check the mode
        if self.learning_with_safety == True:
            logging.warning('This function is targeting for learning on safe value functions')

        # init state
        s = self.env.init_state

        self.test_reward = None
        self.total_rewards = []
        self.errors = []
        self.penalties = []
        self.penalties += [self.penalty.item()]
        ep_rewards = []
        ep_steps = 0
        for t in range(self.total_steps):

            # before exceeding start_steps, randomly sample action
            if t <= self.start_steps:
                a = self.env.sample_action()
            else:
                a = self.actor_critic.act(torch.as_tensor(s, dtype=torch.float32), deterministic=False)

            # observe the next state
            next_state, reward, indicator, failure, goal = self.env.step(self.env.unmap_action(a))
            ep_rewards.append(reward)
            ep_steps += 1

            # add interaction experience to buffer
            self.buffer.add(s, a, reward, indicator, next_state, failure or goal)

            # if failure occurs, reset the env., else; update the state
            # TODO: strategic of re-initializing state 
            if failure or (ep_steps % self.max_ep_len) == 0:
                self.env.reset()
                s = self.env.init_state
                self.total_rewards.append(self.accumulated_return(ep_rewards))
                ep_steps = 0
                ep_rewards = []
                logging.info('Rewards: %f', self.total_rewards[-1])

                if self.env.ground_truth is not None:
                    # screen shot the q functions
                    _Q = self.temporal_q_functions(self.env.s_grid, self.env.a_grid)
                    # thresholding
                    _Q_thresholding = np.where(_Q >= 0, 0, -1)
                    # compute errors
                    error = np.sqrt((np.where(np.array(self.env.ground_truth['Q_V'], dtype=int) == 1, 0, -1) - _Q_thresholding) ** 2).mean()
                    self.errors.append(error)
                    logging.info('Error: %f', error)
            else:
                s = next_state

            # perform update when update_after elapses and every update_every steps
            if t >= self.update_after and t % self.update_every == 0:
                for j in range(self.update_every):
                    states, actions, rewards, indicators, next_states, is_terminal = self.buffer.sample_batch(self.batch_size)
                    self.update(states, actions, rewards, indicators, next_states, is_terminal, j % self.pi_update_every == 0)
                    self.penalties += [self.penalty.item()]
                # if self.env_name == 'inverted_pendulum':
                #     self.test_agent(False)

            if  t % self.q_plot_every == 0 and self.env.ground_truth is not None:
                # screen shot the q functions
                _Q = self.temporal_q_functions(self.env.s_grid, self.env.a_grid)
                plt.imshow(_Q, origin='lower')
                cb = plt.colorbar()
                plt.xlabel("action")
                plt.ylabel("state")
                plt.tight_layout()
                fig = plt.gcf()
                fig.savefig(self.results_folder + '/' + str(t) + '.png')
                cb.remove()
                # thresholding
                _Q_thresholding = np.where(_Q >= 0, 0, -1)
                plt.imshow(_Q_thresholding, origin='lower')
                cb = plt.colorbar()
                plt.xlabel("action")
                plt.ylabel("state")
                plt.tight_layout()
                fig = plt.gcf()
                fig.savefig(self.results_folder + '/' + str(t) + '_logic.png')
                cb.remove()
        
        # done the training
        self.save_models()
        self.save_results()

        return self.results_folder
    
    def train_for_tfl(self, safety_supervisor, iteration_for_safe_action=10):

        # check the mode
        if self.learning_with_safety == False:
            logging.warning('This function is targeting for transfer learning with safety supervisor')

        # init state
        s = self.env.init_state

        self.test_reward = None
        self.test_metric = None
        self.penalties = []
        self.penalties += [self.penalty.item()]
        self.total_rewards = []
        self.errors = []
        ep_rewards = []
        ep_steps = 0
        fail_counter = 0
        for t in range(self.total_steps):
            
            # 1. safety supervisor as a filter
            # before exceeding start_steps, randomly sample safe action
            safest = True
            if t <= self.start_steps:
                # uniformly and randomly pick safe action if safety supervisor is none
                if safety_supervisor is None:
                    a = self.env.sample_action()
                    # set to false because no information
                    safest = False
                else:
                # if the exploring step exceed the max iteration, risk with safest action
                    counter = 0
                    while safest == True and counter < iteration_for_safe_action:
                        a, safest = safety_supervisor.safe_actions(s)
                        counter += 1
            else:
                # pick safe action suggested by actor
                # if the exploring step exceed the max iteration, pick with safest action
                a = self.actor_critic.act(torch.as_tensor(s, dtype=torch.float32), deterministic=False)
                if safety_supervisor is not None:
                    counter = 0
                    safest = False
                    while safety_supervisor.evaluate_state_action(s, a) == False and counter < iteration_for_safe_action:
                        a = self.actor_critic.act(torch.as_tensor(s, dtype=torch.float32), deterministic=False)
                        counter += 1
                        safest = False
                        if safety_supervisor.evaluate_state_action(s, a) == False and counter == iteration_for_safe_action:
                            a = self.actor_critic.act(torch.as_tensor(s, dtype=torch.float32), deterministic=True)
                            # although we set safest True, it is not safest
                            safest = True
                else:
                    # set to false because no information
                    safest = False

            # observe the next state
            if safest and self.virtual_step:
                next_state, reward, indicator, failure, goal = self.env.failure_state(), -self.penalty.item(), -1, False, False
                # add interaction experience to buffer
                self.buffer.add(s, a, reward, indicator, next_state, goal or failure or safest)
            else:
                next_state, reward, indicator, failure, goal = self.env.step(self.env.unmap_action(a))
                # add interaction experience to buffer
                self.buffer.add(s, a, reward, indicator, next_state, goal or failure)
            ep_rewards.append(reward)
            ep_steps += 1

            # if failure or empty action set (safest == True) occurs, reset the env., else; update the state
            # TODO: safest action
            # TODO: strategic of re-initializing state 
            if failure or (ep_steps % self.max_ep_len) == 0 or goal or safest:

                s = self.env.reset(random_init=True, safety_supervisor=safety_supervisor)
                self.total_rewards.append(self.accumulated_return(ep_rewards))

                # logging
                # TODO: if the mem of buffer reach the limit, the trajectory is wrong
                if failure and safest == False:
                    fail_counter += 1
                    T = np.concatenate((self.buffer.state_memory[-ep_steps+self.buffer.mem_counter:self.buffer.mem_counter], next_state.reshape(1, -1)))
                    logging.info('Trajectory: %s', T)
                    logging.warning('Failure occurs')
                elif failure and safest == True:
                    fail_counter += 1
                    T = np.concatenate((self.buffer.state_memory[-ep_steps+self.buffer.mem_counter:self.buffer.mem_counter], next_state.reshape(1, -1)))
                    logging.info('Trajectory: %s', T)
                    logging.warning('Failure occurs due to risking safest action')
                if goal:
                    T = np.concatenate((self.buffer.state_memory[-ep_steps+self.buffer.mem_counter:self.buffer.mem_counter], next_state.reshape(1, -1)))
                    logging.info('Trajectory: %s', T)
                    logging.info('Reach the goal within %d steps', ep_steps)
                if ep_steps == self.max_ep_len:
                    T = np.concatenate((self.buffer.state_memory[-ep_steps+self.buffer.mem_counter:self.buffer.mem_counter], next_state.reshape(1, -1)))
                    if self.total_rewards[-1] > 0:
                        logging.info('Trajectory: %s', T)
                    logging.info('Elapse %d steps', ep_steps)
                if safest and self.virtual_step == False:
                    T = np.concatenate((self.buffer.state_memory[-ep_steps+self.buffer.mem_counter:self.buffer.mem_counter], next_state.reshape(1, -1)))
                    logging.info('Trajectory: %s', T)
                    logging.info('Risk the safest action')
                elif safest and self.virtual_step:
                    T = np.concatenate((self.buffer.state_memory[-ep_steps+self.buffer.mem_counter:self.buffer.mem_counter], next_state.reshape(1, -1)))
                    logging.info('Trajectory: %s', T)
                    logging.info('Add virtual step into buffer without risking the safest action')
                
                ep_steps = 0
                ep_rewards = []
                logging.info('Rewards: %f', self.total_rewards[-1])
            else:
                s = next_state

            # perform update when update_after elapses and every update_every steps
            if t >= self.update_after and t % self.update_every == 0:
                for j in range(self.update_every):
                    states, actions, rewards, indicator, next_states, is_terminal = self.buffer.sample_batch(self.batch_size)
                    self.update(states, actions, rewards, indicator, next_states, is_terminal, j % self.pi_update_every == 0)
                    self.penalties += [self.penalty.item()]
                # if safety_supervisor is not None:
                #     safety_supervisor.actor_critic = self.actor_critic
                # evaluate agent after update
                self.test_agent()

            if  t % self.q_plot_every == 0 and self.env.ground_truth is not None:
                # screen shot the q functions
                _Q = self.temporal_q_functions(self.env.s_grid, self.env.a_grid)
                plt.imshow(_Q, origin='lower')
                cb = plt.colorbar()
                plt.xlabel("action")
                plt.ylabel("state")
                plt.tight_layout()
                fig = plt.gcf()
                fig.savefig(self.results_folder + '/' + str(t) + '.png')
                cb.remove()
        
        # done the training
        self.save_results()

        logging.info('Failure frequency: %s', fail_counter/len(self.total_rewards))
        self.logger.stop_log()

        return safety_supervisor

    def accumulated_return(self, rewards):
        """accumulated return along the trajectory

        Args:
            rewards (list): A list containing element of reward observed at each step

        Returns:
            accumulated_reward: discounted reward
        """

        accumulated_reward = 0
        returns = deepcopy(rewards)
        for i in reversed(range(len(rewards))):
            returns[i] =  returns[i] + self.gamma * accumulated_reward
            accumulated_reward = returns[i]

        return accumulated_reward

    
    def test_agent(self, visualize=False):
        
        if visualize:
            if self.env_name == 'hovership':
                raise ValueError('Not implemented error')
            
            if self.test_env is None:
                self.test_env = deepcopy(self.env)
            env = self.test_env
            ep_rewards = []
            for i in range(self.num_test_episodes):
                s, goal, ep_reward, ep_len, failure = env.reset(), False, [], 0, False
                while not((goal or failure) or (ep_len == self.max_ep_len)):
                    # take deterministic action
                    a = self.actor_critic.act(torch.as_tensor(s, dtype=torch.float32), deterministic=True)
                    next_state, reward, indicator, failure, goal = env.step(env.unmap_action(a))
                    s = next_state
                    ep_reward += [reward]
                    ep_len += 1
                    env.render()
                # env._env.close()
                ep_rewards += [self.accumulated_return(ep_reward)]
        else:
            env = deepcopy(self.env)
            ep_rewards = []
            for i in range(self.num_test_episodes):
                s, goal, ep_reward, ep_len, failure = env.reset(), False, [], 0, False
                while not((goal or failure) or (ep_len == self.max_ep_len)):
                    # take deterministic action
                    a = self.actor_critic.act(torch.as_tensor(s, dtype=torch.float32), deterministic=True)
                    next_state, reward, indicator, failure, goal = env.step(env.unmap_action(a))
                    s = next_state
                    ep_reward += [reward]
                    ep_len += 1
                    # logging
                    if goal:
                        logging.info('Reach the goal within %d steps during test time', ep_len)
                    elif failure:
                        logging.warning('Fail during test time')

                ep_rewards += [self.accumulated_return(ep_reward)]
            if self.test_reward is None:
                self.test_reward = np.array(ep_rewards).reshape(1, -1)
            else:
                self.test_reward = np.concatenate((self.test_reward, np.array(ep_rewards).reshape(1, -1)), axis=0)

            if self.test_metric is None:
                self.test_metric = np.mean(ep_rewards)
                logging.info('Saving the first model with reward = %s', self.test_metric)
                self.save_models()
            else:
                # if better, save the models
                if self.test_metric < np.mean(ep_rewards):
                    self.test_metric = np.mean(ep_rewards)
                    logging.info('Saving the better model with reward = %s', self.test_metric)
                    self.save_models()
                    
            self.save_results()
            logging.info('Average test reward: %f', np.mean(ep_rewards))

    def save_models(self):
        # save models
        with open(self.results_folder + "/q1.pt", "wb") as f:
            torch.save(self.actor_critic.q1.state_dict(), f)
        with open(self.results_folder + "/q2.pt", "wb") as f:
            torch.save(self.actor_critic.q2.state_dict(), f)
        with open(self.results_folder + "/pi.pt", "wb") as f:
            torch.save(self.actor_critic.pi.state_dict(), f)
    
    def save_results(self):

        # plot training
        plt.close('all')
        plt.plot(self.total_rewards, label="per episode")
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.results_folder + "/reward.png")

        if self.test_reward is not None:
            plt.close('all')
            mean = self.test_reward.mean(axis=1).flatten()
            plt.plot(np.linspace(1, mean.shape[0], mean.shape[0], dtype=int), mean, 'b',label="per episode")
            var = self.test_reward.var(axis=1).flatten()
            plt.fill_between(np.linspace(1, mean.shape[0], mean.shape[0], dtype=int), mean-var, mean+var, alpha=0.3, color='b')
            plt.xlabel("episode")
            plt.ylabel("test reward")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.results_folder + "/test_reward.png")

        plt.close('all')
        plt.plot(self.errors, label="per episode")
        plt.xlabel("episode")
        plt.ylabel("error")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.results_folder + "/error.png")
        plt.close('all')

        if self.penalties is not None:
            plt.close('all')
            plt.plot(self.penalties, label="per update")
            plt.xlabel("update")
            plt.ylabel("penalty")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.results_folder + "/penalty.png")
            plt.close('all')

        # save to .csv
        # pd.read_csv(self.results_folder + '/reward.csv', header=None, index_col=0)
        file = pd.DataFrame(self.total_rewards)
        file.to_csv(os.path.join(self.results_folder, 'reward.csv'), header=False)
        if self.test_reward is not None:
            file = pd.DataFrame(self.test_reward)
            file.to_csv(os.path.join(self.results_folder, 'test_reward.csv'), header=False)
        file = pd.DataFrame(self.errors)
        file.to_csv(os.path.join(self.results_folder, 'error.csv'), header=False)
        file = pd.DataFrame(self.penalties)
        file.to_csv(os.path.join(self.results_folder, 'penalty.csv'), header=False)