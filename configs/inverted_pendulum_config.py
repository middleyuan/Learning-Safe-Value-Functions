import sys

sys.path.append('/home/bx007231/deep-q-learning-on-safe-value-functions')

import numpy as np
import start_q_learning
import start_learning_with_safety
import learning_pipeline

settings = dict(
    execute_whole = False, # learning safety supervisor -> transfer learning (the whole pipeline)
    learning_with_safety = True, # True if we have already had the safe value function
    make_comparison = True, # make comparison between training from scratch and with safety supervisor
    model = 'inverted_pendulum-perfect',

    # specifications of transfer learning given the learned safe value function
    TFL = dict(
        saved_model_path = '/media/alextseng/T7/Plotting/svf/inverted_pendulum-please/20220719-201346.077471', # models for pre-trained safe value function
        evaluate_svf = True, # evaluate the performance of learned q functions
        evaluation_steps = 10000, # steps to evaluate learned q functions
        goal_state = None,
        goal_state_reward = None,
        memory_size = 500000,
        init_p = 50, # initial penalty
        penalty_autotune = True,
        reward = 0, # this makes no effect
        virtual_step = True, # when safety supervisor return empty safe action set, the agent 
                              # doesn't risk the safest action. Set to False if numerical issues occur.
        SAC = dict(
            q_plot_every = 1000, 
            seed = None,
            gamma = 1,
            alpha = 1, # the scaler in front of entropy term
            lr = 4e-4,
            polyak = 0.995,
            num_test_episodes = 1, # because the policy is deterministic
            max_ep_len = 1000,
            batch_size = 100, # batch data size for training
            steps_per_epoch = 5000,
            epochs = 20,
            pi_update_every = 2,
            update_every = 50,
            update_after = 0, # if pretrained actor-critic is availiable, update_after and start_steps are 0
            start_steps = 0, # the number of steps for randomly selecting actions
            update_after_scratch = 1000,
            start_steps_scratch = 5000
        )
    ),

    # specifications of learning for safe value functions 
    SVF = dict(
        memory_size = 500000,
        init_p = 30, # initial penalty
        penalty_autotune = True,
        reward = 0, # this makes no effect
        goal_state = None,
        goal_state_reward = None,
        
        SAC = dict(
            q_plot_every = 1000, # if ground truth is not given, q functions will not be plotted
            seed = None,
            gamma = 1,
            alpha = 1, # the scaler in front of entropy term
            lr = 4e-4,
            polyak = 0.995,
            num_test_episodes = 1,
            max_ep_len = 1000,
            batch_size = 100, # batch data size for training
            steps_per_epoch = 5000,
            epochs = 17,
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