from models.hovership import Hovership
from models.inverted_pendulum import InvertedPendulum
from safety_supervisor import SafetySupervisor
from soft_actor_critic import SoftActorCritic
import numpy as np 
import json
from copy import deepcopy

def run(settings):

    ## ** to check learned safe value function: **#
    dynamics = settings['model']
    evaluate_svf = settings['TFL']['evaluate_svf']
    evaluation_steps = settings['TFL']['evaluation_steps']
    saved_model_path = settings['TFL']['saved_model_path']
    if evaluate_svf:
        if dynamics.split('-')[0].strip() == 'hovership':
            # init env.
            env = Hovership()
            env.load_ground_truth()

            # init safety supervisor
            safety_supervisor = SafetySupervisor(env, 'hovership', from_saved_model=saved_model_path)
            safety_supervisor.evaluate_agent(evaluation_steps, visualize=False)
        elif dynamics.split('-')[0].strip() == 'inverted_pendulum':
            env = InvertedPendulum(1000)
            safety_supervisor = SafetySupervisor(env, 'inverted_pendulum', from_saved_model=saved_model_path)
            safety_supervisor.evaluate_agent(evaluation_steps, visualize=False)

    # load paramters
    learning_with_safety = settings['learning_with_safety']
    saved_model_path = settings['TFL']['saved_model_path']
    goal_state = np.array(settings['TFL']['goal_state'])
    goal_state_reward = settings['TFL']['goal_state_reward']
    memory_size = settings['TFL']['memory_size']
    init_p = settings['TFL']['init_p']
    penalty_autotune = settings['TFL']['penalty_autotune']
    reward = settings['TFL']['reward']
    virtual_step = settings['TFL']['virtual_step']

    # init env
    dynamics = settings['model']
    if dynamics.split('-')[0].strip() == 'hovership':
        env = Hovership(safety_mode=learning_with_safety, init_p=init_p, reward=reward, goal_state=goal_state, goal_state_reward=goal_state_reward)
        env.load_ground_truth()
    elif dynamics.split('-')[0].strip() == 'inverted_pendulum':
        env = InvertedPendulum(1000, safety_mode=learning_with_safety, init_p=init_p)

    # init safety supervisor
    safety_supervisor = SafetySupervisor(env, dynamics.split('-')[0].strip(), from_saved_model=saved_model_path)
    pretrained_actor_critic = deepcopy(safety_supervisor.actor_critic)
    pretrained_actor_critic_target = deepcopy(safety_supervisor.actor_critic)

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
    sac.train_for_tfl(safety_supervisor)
    
    # save settings to .json file
    with open(sac.results_folder + '/settings.json', 'w') as wf:
        json.dump(settings, wf, indent=4)
