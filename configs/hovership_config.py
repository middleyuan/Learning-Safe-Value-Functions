import sys

sys.path.append('/home/bx007231/deep-q-learning-on-safe-value-functions')

import numpy as np
import start_q_learning
import start_learning_with_safety
import learning_pipeline

settings = dict(
    execute_whole = False, # execute learning safety supervisor -> transfer learning (the whole pipeline)
    learning_with_safety = True, # True if we have already had the safe value function
    make_comparison = True, # make comparison between training from scratch and with safety supervisor
    model = 'hovership-perfect',

    # specifications of transfer learning given the learned safe value function
    TFL = dict(
        saved_model_path = '/media/alextseng/T7/Plotting/svf/hovership-final/20220710-020147.905600', # models for pre-trained safe value function
        evaluate_svf = True, # evaluate the performance of learned q functions
        evaluation_steps = 10000, # steps to evaluate learned q functions
        goal_state = [1.3], # goal state for new task e.g. hovership: [1.5]
        goal_state_reward = 50, # the reward when reaching the goal state
        memory_size = 250000,
        init_p = 50, # initial penalty
        penalty_autotune = True,
        reward = 0, # reward when staying in viable kernel
        virtual_step = True, # when safety supervisor return empty safe action set, the agent doesn't risk the safest action
        SAC = dict(
            q_plot_every = 1000,
            seed = None,
            gamma = 0.8,
            alpha = 1.5, # the scaler in front of entropy term
            lr = 1e-3,
            polyak = 0.995,
            num_test_episodes = 1, # because the policy is deterministic
            max_ep_len = 1000,
            batch_size = 100, # batch data size for training
            steps_per_epoch = 5000,
            epochs = 1,
            pi_update_every = 2,
            update_every = 50,
            update_after = 1000,
            start_steps = 3000, # the number of steps for randomly selecting actions
            update_after_scratch = 1000,
            start_steps_scratch = 3000
        )
    ),

    # specifications of learning for safe value functions 
    SVF = dict(
        memory_size = 250000,
        init_p = 10, # initial penalty
        penalty_autotune = True,
        reward = 0,
        goal_state = None, # goal state for new task e.g. hovership: [1.5]
        goal_state_reward = None, # the reward when reaching the goal state
        
        SAC = dict(
            q_plot_every = 1000,
            seed = None,
            gamma = 1,
            alpha = 1.5, # the scaler in front of entropy term
            lr = 1e-3,
            polyak = 0.995,
            num_test_episodes = 1,
            max_ep_len = 1000,
            batch_size = 100, # batch data size for training
            steps_per_epoch = 5000,
            epochs = 1,
            pi_update_every = 2,
            update_every = 50,
            update_after = 1000,
            start_steps = 10000 # the number of steps for randomly selecting actions
        )
    )
)

def main():
    if settings['execute_whole']:
        learning_pipeline.run(settings)
    else:
        if settings['learning_with_safety']:
            start_learning_with_safety.run(settings)
        else:
            start_q_learning.run(settings)

if __name__ == '__main__':
    main()