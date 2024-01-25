import sys, os
import numpy as np
import pandas as pd
from varname import nameof
from matplotlib import pyplot as plt
import argparse
parser = argparse.ArgumentParser()

# flag definitions
parser.add_argument("-svf", dest='svf_folder', default='svf',
                    help="The folder where the data during learning safety supervisor are located")
parser.add_argument("-tfl", dest='tfl_folder', default='tfl',
                    help="The folder where the data during learning safety supervisor are located")
parser.add_argument("-env", dest='dynamics', default='hovership',
                    help="State for which results of the dynamics you would like to plot")


'''
This script is used to plot the results and give some statistic yielded by RWTH cluster.
By default, it will assume the folder 'svf' and 'tfl' are under the same working directory 
as plot.py. Otherwise please pass two arguments to state where are these two folder.
'''

def run(args):
    # get the path of this script
    script_path = os.path.dirname(__file__)

    # get all the folders' name located in 'svf/[env]'
    target_folder = os.path.join(args.svf_folder, args.dynamics)
    target_folder = os.path.join(script_path, target_folder)
    storing_path = target_folder
    dir_list = os.listdir(target_folder)
    dir_list = [filename for filename in dir_list if os.path.isdir(os.path.join(target_folder, filename))]
    
    svf_errors = []
    svf_is_empty_sets = []
    svf_rewards = []
    vanilla_sac_rewards = []
    vanilla_sac_test_rewards = []
    advanced_sac_rewards = []
    advanced_sac_test_rewards = []
    vanilla_failure_freq = []
    advanced_failure_freq = []
    test_failure_freq = []
    # loop over the dir_list
    for idx, folder in enumerate(dir_list):
        # analyse the folder in the list
        analysed_folder = os.path.join(target_folder, folder)

        if os.path.isfile(analysed_folder + '/error.csv') and os.path.getsize(analysed_folder + '/error.csv') > 0:
            # read error
            svf_errors += [pd.read_csv(analysed_folder + '/error.csv', header=None, index_col=0).values]
        if os.path.isfile(analysed_folder + '/is_empty_set.csv') and os.path.getsize(analysed_folder + '/is_empty_set.csv') > 0:
            # read is_empty_set (evalutation on how often does safety supervisor return empty action set)
            svf_is_empty_sets += [pd.read_csv(analysed_folder + '/is_empty_set.csv', header=None, index_col=0).values]
        
        # read reward
        svf_rewards += [pd.read_csv(analysed_folder + '/reward.csv', header=None, index_col=0).values]


        # search for the storing folder for 'SAC from scratch' and 'SAC with safety supervisor'
        # format: [INFO ]  tfl/hovership/20220624-155817
        # tfl_folders contains 'SAC with safety supervisor' as the first element and 'SAC from scratch'
        # as the second element.
        tfl_folders = []
        failure_freq = []
        test_failure_counter = 0
        with open(analysed_folder + '/log.log', 'r') as f:
            for line_number, line_buffer in enumerate(f):
                if 'tfl' in line_buffer:
                    line_buffer_list = line_buffer.split('/')
                    # put the last one in the list
                    tfl_folders += [line_buffer_list[-1].strip()]
                if 'Failure frequency' in line_buffer:
                    line_buffer_list = line_buffer.split(':')
                    # put the last one in the list
                    failure_freq += [float(line_buffer_list[-1].strip())]
                if 'Fail during test time' in line_buffer:
                    test_failure_counter += 1
        
        print(folder, '->: ', tfl_folders[0], '->: ', tfl_folders[1])
        # add failure_freq
        advanced_failure_freq += [failure_freq[0]]
        vanilla_failure_freq += [failure_freq[1]]
        test_failure_freq += [test_failure_counter]

        # 'SAC with safety supervisor'
        target_tfl_folder = os.path.join(args.tfl_folder, args.dynamics)
        target_tfl_folder = os.path.join(script_path, target_tfl_folder)
        analysed_tfl_folder = os.path.join(target_tfl_folder, tfl_folders[0])

        advanced_sac_rewards += [pd.read_csv(analysed_tfl_folder + '/reward.csv', header=None, index_col=0).values]
        advanced_sac_test_rewards += [pd.read_csv(analysed_tfl_folder + '/test_reward.csv', header=None, index_col=0).values]

        # 'SAC from scratch'
        analysed_tfl_folder = os.path.join(target_tfl_folder, tfl_folders[1])

        vanilla_sac_rewards += [pd.read_csv(analysed_tfl_folder + '/reward.csv', header=None, index_col=0).values]
        vanilla_sac_test_rewards += [pd.read_csv(analysed_tfl_folder + '/test_reward.csv', header=None, index_col=0).values]

    # post-process the data
    if len(svf_errors) > 0:
        min_size = min(map(len, svf_errors))
        svf_errors_array = np.array(svf_errors[0][:min_size])
        for i in range(1, len(dir_list)):
            svf_errors_array = np.concatenate((svf_errors_array, np.array(svf_errors[i][:min_size])), axis=1)
    else:
        svf_errors_array = np.array([])
    
    if len(svf_is_empty_sets) > 0:
        min_size = min(map(len, svf_is_empty_sets))
        svf_is_empty_sets_array = np.array(svf_is_empty_sets[0][:min_size])
        for i in range(1, len(dir_list)):
            svf_is_empty_sets_array = np.concatenate((svf_is_empty_sets_array, np.array(svf_is_empty_sets[i][:min_size])), axis=1)
    else:
        svf_is_empty_sets_array = np.array([])

    min_size = min(map(len, svf_rewards))
    svf_rewards_array = np.array(svf_rewards[0][:min_size])
    for i in range(1, len(dir_list)):
        svf_rewards_array = np.concatenate((svf_rewards_array, np.array(svf_rewards[i][:min_size])), axis=1)

    min_size = min(map(len, advanced_sac_rewards))
    advanced_sac_rewards_array = np.array(advanced_sac_rewards[0][:min_size])
    for i in range(1, len(dir_list)):
        advanced_sac_rewards_array = np.concatenate((advanced_sac_rewards_array, np.array(advanced_sac_rewards[i][:min_size])), axis=1)
    
    min_size = min(map(len, advanced_sac_test_rewards))
    advanced_sac_test_rewards_array = np.array(advanced_sac_test_rewards[0][:min_size])
    for i in range(1, len(dir_list)):
        advanced_sac_test_rewards_array = np.concatenate((advanced_sac_test_rewards_array, np.array(advanced_sac_test_rewards[i][:min_size])), axis=1)
    
    min_size = min(map(len, vanilla_sac_rewards))
    vanilla_sac_rewards_array = np.array(vanilla_sac_rewards[0][:min_size])
    for i in range(1, len(dir_list)):
        vanilla_sac_rewards_array = np.concatenate((vanilla_sac_rewards_array, np.array(vanilla_sac_rewards[i][:min_size])), axis=1)

    min_size = min(map(len, vanilla_sac_test_rewards))
    vanilla_sac_test_rewards_array = np.array(vanilla_sac_test_rewards[0][:min_size])
    for i in range(1, len(dir_list)):
        vanilla_sac_test_rewards_array = np.concatenate((vanilla_sac_test_rewards_array, np.array(vanilla_sac_test_rewards[i][:min_size])), axis=1)

    vanilla_failure_freq_array = np.array(vanilla_failure_freq).reshape(-1, 1)
    advanced_failure_freq_array = np.array(advanced_failure_freq).reshape(-1, 1)
    test_failure_freq_array = np.array(test_failure_freq).reshape(-1, 1)

    def draw_plot(data, legend, ylabel, ylim=(-50, 50)):
        """draw the plot

        Args:
            data (nparray): it must be in the shape of (time series, # runs)
        """     
        plt.close('all')

        if len(data) == 1:
            mean = data[0].mean(axis=1).flatten()
            plt.plot(np.linspace(1, mean.shape[0], mean.shape[0], dtype=int), mean, 'b', label=legend[0])
            var = data[0].var(axis=1).flatten()
            plt.fill_between(np.linspace(1, mean.shape[0], mean.shape[0], dtype=int), mean-var, mean+var, alpha=0.3, color='b')
            plt.xlabel("episode")
            plt.ylabel(ylabel)
            if ylim is not None:
                plt.ylim(ylim)
            plt.legend()
            plt.tight_layout()
            plt.savefig(storing_path + '/' + ylabel + ".png")
        else:
            for i in range(len(data)):
                mean = data[i].mean(axis=1).flatten()
                plt.plot(np.linspace(1, mean.shape[0], mean.shape[0], dtype=int), mean, label=legend[i])
                std = data[i].std(axis=1).flatten()
                plt.fill_between(np.linspace(1, mean.shape[0], mean.shape[0], dtype=int), mean-std, mean+std, alpha=0.3)
                
            if ylim is not None:
                plt.ylim(ylim)
            plt.xlabel("episode")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(storing_path + '/' + ylabel + ".png")

    if len(svf_errors) > 0:
        draw_plot([svf_errors_array], ['RMSE of Viable Set'], 'RMSE of Viable Set', None)
    if len(svf_is_empty_sets) > 0:
        draw_plot([svf_rewards_array], ['Reward of Safe Value Function'], 'Reward of Safe Value Function', None)
    draw_plot([advanced_sac_rewards_array, vanilla_sac_rewards_array], ['Advanced SAC Reward', 'Vanilla SAC Reward'], 'Reward', None)
    draw_plot([advanced_sac_test_rewards_array, vanilla_sac_test_rewards_array], ['Advanced SAC Reward', 'Vanilla SAC Reward'], 'Test Reward', None)

    data = [['Empty Set Freq', svf_is_empty_sets_array.mean(), svf_is_empty_sets_array.std()],
            ['Vanilla Failure Freq', vanilla_failure_freq_array.mean(), vanilla_failure_freq_array.std()],
            ['Advanced Failure Freq', advanced_failure_freq_array.mean(), advanced_failure_freq_array.std()],
            ['Test Failure Freq', test_failure_freq_array.mean(), test_failure_freq_array.std()]]
    df = pd.DataFrame(data, columns=['name', 'mean', 'std'])
    print(df)
    df.to_csv(storing_path + '/statistic.csv')

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)