# Deep Q learning on safe value functions

To extend the theoretical work in [Safe Value Functions](https://arxiv.org/abs/2105.12204) to practical reinforcement learning (RL) algorithms and provide empirical evaluations.
Specifically, we want to show that we can learn safe value functions and, consequently, viable sets
as defined by [A Learnable Safety Measure](https://arxiv.org/abs/1910.02835) and [Beyond Basins of Attraction: Quantifying Robustness of Natural Dynamics](https://arxiv.org/abs/1806.08081) using RL algorithms. We plan to show, that using this framework,
we can learn a safety supervisor that knows the set of all safe policies and therefore enable safe
learning after we we have learned an initial safe policy.

## Setup

If you want to use virtual enviroment

    $ conda env create -f DQL-SVF.yml

Activate the enviroment

    $ conda activate DQL-SVF

Install [gym-cartpole-swingup](https://github.com/0xangelo/gym-cartpole-swingup)

    $ pip install gym-cartpole-swingup

## Run

Start to learn safe value functions for hovership dynamics

    $ python configs/hovership_config.py 

## Issues

1. Module not found: one way to solve this is to add your working space path to PYTHONPATH in `.bashrc`
        
        $ gedit ~/.bashrc
        $ export PYTHONPATH="${PYTHONPATH}:/home/alextseng/deep-q-learning-on-safe-value-functions

## Results

1. On hovership example as a proof of concept, we show that once we learn accurate
enough safety supervisor, in the transfer learning stage, the learning is safe and
sampling efficiently compared to learning from scratch.
2. . We evaluate our framework on the high-dimensional task, i.e., inverted pendulum,
to shed light on future works.

To see more details, see [Report](https://middleyuan.github.io/old.middleyuan.github.io/materials/Project_Report_learning_safe_value_functions_Tsung_with_signature.pdf) and [Slides](https://middleyuan.github.io/old.middleyuan.github.io/materials/Project_Presentation.pdf).
