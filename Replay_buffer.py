import torch
import numpy as np
np.bool = np.bool_

# 经验回放池，随机采样
class replay_buffer:
    def __init__(self, buffer_size, batch_size, state_dim, action_dim, max_nodes, device):
        self.memory = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((buffer_size, max_nodes, state_dim), dtype=np.float32)
        self.mask = np.zeros((buffer_size, max_nodes), dtype=np.int32)
        self.action =np.zeros((buffer_size, max_nodes, action_dim), dtype=np.float32)
        self.reward = np.zeros((buffer_size, max_nodes), dtype=np.float32)
        self.done = np.zeros((buffer_size, max_nodes), dtype=np.bool)
        self.state_next = np.zeros((buffer_size, max_nodes, state_dim), dtype=np.float32)
        self.mask_next = np.zeros((buffer_size, max_nodes), dtype=np.int32)
        self.goal_state = np.zeros((buffer_size, max_nodes * state_dim), dtype=np.float32)


    def add(self, state, mask, action, reward, state_next, mask_next, done):

        self.state[self.ptr] = state
        self.mask[self.ptr] = mask
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.state_next[self.ptr] = state_next
        self.mask_next[self.ptr] = mask_next
        self.done[self.ptr] = done
        self.goal_state[self.ptr] = state.reshape(-1)

        self.ptr = (self.ptr + 1) % self.memory
        self.size = min(self.size + 1, self.memory)

    def sample(self):
        ind = np.random.randint(0, self.size, size=(self.batch_size,))

        return (torch.from_numpy(self.state[ind]).to(dtype=torch.float32, device=self.device),
                torch.from_numpy(self.mask[ind]).to(dtype=torch.int32, device=self.device),
                torch.from_numpy(self.action[ind]).to(dtype=torch.float32, device=self.device),
                torch.from_numpy(self.reward[ind]).to(dtype=torch.float32, device=self.device),
                torch.from_numpy(self.done[ind]).to(dtype=torch.bool, device=self.device),
                torch.from_numpy(self.state_next[ind]).to(dtype=torch.float32, device=self.device),
                torch.from_numpy(self.mask_next[ind]).to(dtype=torch.int32, device=self.device))

    def len(self):
        return self.size
