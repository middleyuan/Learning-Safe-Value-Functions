import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, mem_size, state_shape, action_shape):
        """Initialization of the replay buffer.
        
        The memories have the following data types:
            states: float32
            next_states: float32
            actions: float32
            rewards: float32
            is_terminal: bool

        Args:
            mem_size: Capacity of this buffer
            state_shape: Shape of state and next_state
        """
        self.mem_size = mem_size  # Capacity of the buffer
        self.mem_counter = 0         # Number of added elements
        self.state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *state_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, *action_shape), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.indicator_memory = np.zeros(self.mem_size, dtype=np.float32)
    
    def is_filled(self):
        """Check if the memory is filled."""
        return self.mem_counter >= self.mem_size

    def add(self, state, action, reward, indicator, next_state, is_terminal):
        """Add one transition to the buffer.

        Replaces the oldest transition in memory.
        """
        
        i = self.mem_counter % self.mem_size
        self.mem_counter += 1
        
        self.state_memory[i] = state
        self.action_memory[i] = action
        self.reward_memory[i] = reward
        self.next_state_memory[i] = next_state
        self.terminal_memory[i] = is_terminal
        self.indicator_memory[i] = indicator

    def add_batch(self, states, actions, rewards, indicator, next_states, is_terminal):

        states = states.numpy()
        actions = actions.numpy()
        rewards = rewards.numpy()
        next_states = next_states.numpy()
        is_terminal = is_terminal.numpy()
        indicator = indicator.numpy()

        size = len(states)

        i = self.mem_counter % self.mem_size
        self.mem_counter += size

        self.state_memory[i:self.mem_counter] = states
        self.action_memory[i:self.mem_counter] = actions
        self.reward_memory[i:self.mem_counter] = rewards
        self.next_state_memory[i:self.mem_counter] = next_states
        self.terminal_memory[i:self.mem_counter] = is_terminal
        self.indicator_memory[i:self.mem_counter] = indicator


    def sample_batch(self, batch_size, use_latest=False):
        """Sample one batch from the memory."""

        # TODO: imbalance failure sample
        upperlimit = self.mem_size if self.mem_counter >= self.mem_size else self.mem_counter
        if use_latest == False:
            # self.mem_counter is not included
            batch_index = np.random.randint(0, upperlimit, size=batch_size)
        else:
            batch_index = np.random.randint(0, upperlimit-1, size=batch_size-1)
            # include the latest sample
            batch_index = np.concatenate((batch_index, np.array([upperlimit-1])))
        
        states = self.state_memory[batch_index]
        actions = self.action_memory[batch_index]
        rewards = self.reward_memory[batch_index]
        next_states = self.next_state_memory[batch_index]
        is_terminal = self.terminal_memory[batch_index]
        indicator = self.indicator_memory[batch_index]

        # recast to tensor
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        is_terminal = torch.as_tensor(is_terminal, dtype=torch.float32)
        indicator = torch.as_tensor(indicator, dtype=torch.float32)

        return states, actions, rewards, indicator, next_states, is_terminal