from models.hovership import Hovership
from models.inverted_pendulum import InvertedPendulum
from safety_supervisor import SafetySupervisor
from soft_actor_critic import SoftActorCritic
import numpy as np 
import json

def run(settings):

    # *** learning safety supervisor *** #
    # *** first part of the pipeline *** #
    # size of replay buffer
    mem_size = settings['SVF']['memory_size']

    # initial penalty
    init_penalty = settings['SVF']['init_p']
    # reward
    reward = settings['SVF']['reward']
    goal_state = np.array(settings['SVF']['goal_state']) if settings['SVF']['goal_state'] is not None else None
    goal_state_reward = settings['SVF']['goal_state_reward']

    #**** Adapted SAC algorithm with penalty auto-tune****#
    dynamics = settings['model']
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

    sac = SoftActorCritic(dynamics, q_plot_every, mem_size, seed, gamma, alpha, learning_rate, polyak,
                 num_test_episodes, pi_update_every, max_ep_len, batch_size, epochs,
                 steps_per_epoch, update_every, update_after, start_steps, penalty_autotune,
                 init_penalty, reward, goal_state, goal_state_reward, False, False, None, None)
    
    saved_model_path = sac.train_for_svf()
    pretrained_actor_critic = sac.actor_critic
    pretrained_actor_critic_target = sac.actor_critic_target
    old_buffer = sac.buffer

    # save settings to .json file
    with open(sac.results_folder + '/settings.json', 'w') as wf:
        json.dump(settings, wf, indent=4)


    # *** learning with safety supervisor *** #
    # *** second part of the pipeline     *** #
    ## ** to check learned safe value function: **#
    evaluate_svf = settings['TFL']['evaluate_svf']
    evaluation_steps = settings['TFL']['evaluation_steps']
    if evaluate_svf:
        if dynamics.split('-')[0].strip() == 'hovership':
            # init env.
            env = Hovership()
            env.load_ground_truth()

            # init safety supervisor
            safety_supervisor = SafetySupervisor(env, 'hovership', from_saved_model=saved_model_path)
            safety_supervisor.evaluate_agent(evaluation_steps, visualize=False)
        elif dynamics.split('-')[0].strip() == 'inverted_pendulum':
            env = InvertedPendulum(max_ep_len)
            safety_supervisor = SafetySupervisor(env, 'inverted_pendulum', from_saved_model=saved_model_path)
            safety_supervisor.evaluate_agent(evaluation_steps, visualize=False)
        else:
            raise ValueError('unkown dynamics')

    # load paramters
    learning_with_safety = True
    goal_state = np.array(settings['TFL']['goal_state']) if settings['TFL']['goal_state'] is not None else None
    goal_state_reward = settings['TFL']['goal_state_reward']
    memory_size = settings['TFL']['memory_size']
    init_p = settings['TFL']['init_p']
    penalty_autotune = settings['TFL']['penalty_autotune']
    reward = settings['TFL']['reward']
    virtual_step = settings['TFL']['virtual_step']

    # init env
    if dynamics.split('-')[0].strip() == 'hovership':
        env = Hovership(safety_mode=learning_with_safety, init_p=init_p, reward=reward, goal_state=goal_state, goal_state_reward=goal_state_reward)
        env.load_ground_truth()
    elif dynamics.split('-')[0].strip() == 'inverted_pendulum':
        env = InvertedPendulum(max_ep_len, safety_mode=learning_with_safety, init_p=init_p)
    else:
        raise ValueError('unkown dynamics')

    # init safety supervisor
    safety_supervisor = SafetySupervisor(env, dynamics.split('-')[0].strip(), from_saved_model=saved_model_path)

    # SAC
    seed = settings['TFL']['SAC']['seed']
    gamma = settings['TFL']['SAC']['gamma']
    alpha = settings['TFL']['SAC']['alpha']
    learning_rate = settings['TFL']['SAC']['lr']
    polyak = settings['TFL']['SAC']['polyak']
    num_test_episodes = settings['TFL']['SAC']['num_test_episodes']
    max_ep_len = settings['TFL']['SAC']['max_ep_len']
    batch_size = settings['TFL']['SAC']['batch_size']
    steps_per_epoch = settings['TFL']['SAC']['steps_per_epoch']
    epochs = settings['TFL']['SAC']['epochs']
    pi_update_every = settings['TFL']['SAC']['pi_update_every']
    update_every = settings['TFL']['SAC']['update_every']
    update_after = settings['TFL']['SAC']['update_after']
    start_steps = settings['TFL']['SAC']['start_steps']
    q_plot_every = settings['TFL']['SAC']['q_plot_every']

    sac = SoftActorCritic(dynamics, q_plot_every, memory_size, seed, gamma, alpha, learning_rate, polyak,
                 num_test_episodes, pi_update_every, max_ep_len, batch_size, epochs,
                 steps_per_epoch, update_every, update_after, start_steps, penalty_autotune,
                 init_p, reward, goal_state, goal_state_reward, learning_with_safety, virtual_step,
                 pretrained_actor_critic, pretrained_actor_critic_target)

    # start transfer learning
    # states, actions, rewards, indicator, next_states, is_terminal = old_buffer.sample_batch(4000)
    # sac.buffer.add_batch(states, actions, rewards, indicator, next_states, is_terminal)
    sac.train_for_tfl(safety_supervisor)

    # compare with training from scratch (without safety supervisor)
    update_after = settings['TFL']['SAC']['update_after_scratch']
    start_steps = settings['TFL']['SAC']['start_steps_scratch']
    virtual_step = False
    sac = SoftActorCritic(dynamics, q_plot_every, memory_size, seed, gamma, alpha, learning_rate, polyak,
                 num_test_episodes, pi_update_every, max_ep_len, batch_size, epochs,
                 steps_per_epoch, update_every, update_after, start_steps, penalty_autotune,
                 init_p, reward, goal_state, goal_state_reward, learning_with_safety, virtual_step,
                 None, None)
                 
    sac.train_for_tfl(safety_supervisor=None)
    
    # save settings to .json file
    with open(sac.results_folder + '/settings.json', 'w') as wf:
        json.dump(settings, wf, indent=4)