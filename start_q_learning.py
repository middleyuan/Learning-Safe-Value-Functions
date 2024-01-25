from soft_actor_critic import SoftActorCritic
import json
import numpy as np


def run(settings):
    """run the algorithm for learning safe value functions

    Args:
        settings (dict): configuration for specific task (dynamics)
    """   

    # size of replay buffer
    mem_size = settings['SVF']['memory_size']

    # initial penalty
    init_penalty = settings['SVF']['init_p']
    # reward
    reward = settings['SVF']['reward']
    goal_state = np.array(settings['SVF']['goal_state'])
    goal_state_reward = settings['SVF']['goal_state_reward']

    #**** Adapted SAC algorithm with penalty auto-tune****#
    env = settings['model']
    q_plot_every = settings['SVF']['SAC']['q_plot_every']
    penalty_autotune = settings['SVF']['penalty_autotune']
    seed = settings['SVF']['SAC']['seed']
    gamma = settings['SVF']['SAC']['gamma']
    alpha = settings['SVF']['SAC']['alpha']
    learning_rate = settings['SVF']['SAC']['lr']
    polyak = settings['SVF']['SAC']['polyak']
    num_test_episodes = settings['SVF']['SAC']['num_test_episodes']
    pi_update_every = settings['SVF']['SAC']['pi_update_every']
    max_ep_len = settings['SVF']['SAC']['max_ep_len']
        
    # training loop
    batch_size = settings['SVF']['SAC']['batch_size']
    epochs = settings['SVF']['SAC']['epochs']
    steps_per_epoch = settings['SVF']['SAC']['steps_per_epoch']
    total_steps = steps_per_epoch * epochs
    update_every = settings['SVF']['SAC']['update_every']
    update_after = settings['SVF']['SAC']['update_after']
    start_steps = settings['SVF']['SAC']['start_steps']

    sac = SoftActorCritic(env, q_plot_every, mem_size, seed, gamma, alpha, learning_rate, polyak,
                 num_test_episodes, pi_update_every, max_ep_len, batch_size, epochs,
                 steps_per_epoch, update_every, update_after, start_steps, penalty_autotune,
                 init_penalty, reward, goal_state, goal_state_reward, False, False, None, None)
    
    sac.train_for_svf()

    # save settings to .json file
    with open(sac.results_folder + '/settings.json', 'w') as wf:
        json.dump(settings, wf, indent=4)