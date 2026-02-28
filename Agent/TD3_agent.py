import numpy as np
import torch
import torch.optim as optm
import torch.nn.functional as F
import copy
import os

from Utils.Nerual_Network_TD3 import Policy_Network, Critic_Network
from Replay_buffer import replay_buffer


# SAC Agent
class TD3_agent():

    def __init__(self, **kwargs):
        # 参数继承
        self.__dict__.update(kwargs)

        # 定义策略网络(actor)
        self.actor = Policy_Network \
            (self.feature_dim,
             self.hidden_dim,
             self.action_dim,
             self.action_bound,
             self.dropout,
             self.beta).to(self.device)

        # 定义Q网络(critic)
        self.critic = Critic_Network \
            (self.feature_dim,
             self.hidden_dim,
             self.action_dim,
             self.dropout,
             self.beta).to(self.device)

        self.policy_noise = 0.2 * self.action_bound
        self.noise_clip = 0.5 * self.action_bound
        self.explore_noise = 0.15
        self.explore_decay = 0.995
        self.delay_counter = 0

        # 优化器设置
        self.actor_optimizer = optm.Adam \
            (self.actor.parameters(),
             lr=self.actor_lr, weight_decay=self.weight_decay)
        self.critic_optimizer = optm.Adam \
            (self.critic.parameters(),
             lr=self.critic_lr, weight_decay=self.weight_decay)

        # 定义目标Q网络(target_critic)
        self.target_critic = copy.deepcopy(self.critic)
        for params in self.target_critic.parameters():
            params.requires_grad = False

        # 定义经验回放池
        self.replay_buffer = replay_buffer(self.buffer_size,
                                           self.batch_size,
                                           self.state_dim,
                                           self.action_dim,
                                           self.max_nodes,
                                           self.device)

        # 定义loss收集器
        self.actor_loss = []
        self.critic_loss = []

    # 动作函数
    def take_action(self, state_, mask):
        with torch.no_grad():
            state = torch.from_numpy(state_).to(dtype=torch.float32, device=self.device)
            mask = torch.from_numpy(mask).to(dtype=torch.int32, device=self.device)
            action = self.actor(state, mask)
            if self.training:
                noise = torch.normal(0.0 * self.action_bound,
                                         self.explore_noise * self.action_bound)
                action = (action + noise).clamp(-self.action_bound, self.action_bound)


        return action.to(dtype=torch.float32).cpu().detach().numpy().astype(np.float32)

    # Q值函数评估
    def calc_target(self, rewards, next_state, next_mask, dones):

        next_actions, next_log_prob = self.actor(next_state, next_mask)
        action_noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_actions = (action_noise + next_actions).clamp(-self.action_bound, self.action_bound)

        q1_value, q2_value = self.target_critic(next_state, next_actions, next_mask)

        if self.RL_agent == 'ISAC':
            # Independent Soft Actor-Critic
            target_q = rewards + self.gamma * (~dones) * torch.min(q1_value, q2_value)
            target_q = target_q * next_mask
        elif self.RL_agent == 'VDN':
            # Value-Decomposition Network
            reward = rewards.sum(dim=-1)
            q_next = self.gamma * (~dones) * torch.min(q1_value, q2_value)
            target_q = reward + q_next.sum(dim=-1)
        elif self.RL_agent == 'M_VDN':
            # Mean Value-Decomposition Network
            reward = rewards.sum(dim=-1) / next_mask.sum(dim=-1)
            q_next = self.gamma * (~dones) * torch.min(q1_value, q2_value)
            target_q = reward + q_next.sum(dim=-1) / next_mask.sum(dim=-1)

        return target_q

    # 目标网络更新
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau)
                                    + param.data * self.tau)

    # 主网络更新(actor, critic)
    def update(self, ep_i_steps):

        if self.replay_buffer.len() < self.replay_buffer.batch_size:
            return

        self.delay_counter += 1
        # 随机采样
        # state, mask, action, reward, done, state_next, mask_next
        states, mask, actions, rewards, dones, next_states, mask_next = \
            self.replay_buffer.sample()

        # TD error估计, Q网络更新(critic)
        # ----------------------------- ↓↓↓↓↓ Update QValue Net ↓↓↓↓↓ ------------------------------#
        Q_target = self.calc_target(rewards, next_states, mask_next, dones)

        Q1_current, Q2_current = self.critic(states, actions, mask)

        if self.RL_agent == 'ISAC':
            # Independent Soft Actor-Critic
            car_count = mask.sum()
            Q1_current = Q1_current * mask
            Q2_current = Q2_current * mask
            td_error = (1 / car_count) * ((Q_target - Q1_current) ** 2).sum() \
                        + (1 / car_count) * ((Q2_current - Q1_current) ** 2).sum()
        elif self.RL_agent == 'VDN':
            # Value-Decomposition Network
            td_error = (F.mse_loss(Q_target, Q1_current.sum(dim=-1))
                        + F.mse_loss(Q_target, Q2_current.sum(dim=-1)))
        elif self.RL_agent == 'M_VDN':
            # Mean Value-Decomposition Network
            td_error = (F.mse_loss(Q_target, Q1_current.sum(dim=-1) / mask.sum(dim=-1))
                        + F.mse_loss(Q_target, Q2_current.sum(dim=-1) / mask.sum(dim=-1)))


        # 计算Q网络损失，反向传播，使用梯度剪切避免梯度爆炸
        self.critic_loss.append(td_error.item())

        self.critic_optimizer.zero_grad()
        td_error.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # 延迟更新(TD3机制)
        if ep_i_steps % self.policy_delay == 0:

            for params in self.critic.parameters(): params.requires_grad = False
            # Policy策略网络估计(actor)
            # ----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
            new_actions = self.actor(states, mask)

            Q1_value, Q2_value = self.critic(states, new_actions, mask)
            min_q = torch.min(Q1_value, Q2_value)

            if self.RL_agent == 'ISAC':
                actor_loss = -(1 / car_count) * min_q.sum()
            elif self.RL_agent == 'VDN':
                actor_loss = - (min_q.sum(dim=-1)).mean()
            elif self.RL_agent == 'M_VDN':
                actor_loss = -(min_q.sum(dim=-1) / mask.sum(dim=-1)).mean()

            self.actor_loss.append(actor_loss.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            # 目标网络软更新
            self.soft_update(self.critic, self.target_critic)

    # 模型存储
    def save(self, ep_i, CAV_PR, time_step, control_strategy, RL_agent):
        EnvName = self.Env_name
        curr_path = os.path.dirname(__file__)
        parent_path = os.path.dirname(curr_path)

        output_dir = parent_path + "/model./{}_{}./".format(control_strategy, RL_agent)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path_actor = output_dir + "TD3_actor_{}_timestep_{}_CAVPR_{}_{}_{}.pth".format(time_step, ep_i, CAV_PR, control_strategy, RL_agent)
        output_path_critic = output_dir + "TD3_critic_{}_timestep_{}_CAVPR_{}_{}_{}.pth".format(time_step, ep_i, CAV_PR, control_strategy, RL_agent)

        torch.save(self.actor.state_dict(), output_path_actor)
        torch.save(self.critic.state_dict(), output_path_critic)

    # 模型加载
    def load(self, epoch, CAV_PR, time_step, control_strategy, RL_agent):
        EnvName = self.Env_name

        curr_path = os.path.dirname(__file__)
        parent_path = os.path.dirname(curr_path)

        input_dir = parent_path + "/model./{}_{}./".format(control_strategy, RL_agent)

        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

        input_path_actor = input_dir + "actor_{}_timestep_{}_CAVPR_{}_{}_{}.pth".format(time_step, epoch, CAV_PR, control_strategy, RL_agent)
        input_path_critic = input_dir + "critic_{}_timestep_{}_CAVPR_{}_{}_{}.pth".format(time_step, epoch, CAV_PR, control_strategy, RL_agent)

        state_dict_before = self.actor.state_dict()
        actor_state_dict = torch.load(input_path_actor, weights_only=False)
        critic_state_dict = torch.load(input_path_critic, weights_only=False)

        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)
        # 检查差异
        for key in self.actor.state_dict():
            diff = state_dict_before[key] - self.actor.state_dict()[key]
            if diff.sum() != 0 and key == 'residual_1.weight':
                print('加载模型参数有误')
            else:
                print('加载模型参数成功')
            break

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    # 网络重制，避免网络参数被前期数据污染
    def reset(self):
        self.actor = Policy_Network \
                    (self.feature_dim,
                     self.hidden_dim,
                     self.action_dim,
                     self.action_bound,
                     self.dropout,
                     self.beta).to(self.device)
        self.critic = Critic_Network \
                    (self.feature_dim,
                     self.hidden_dim,
                     self.action_dim,
                     self.dropout,
                     self.beta).to(self.device)

        self.target_critic = copy.deepcopy(self.critic)
        for params in self.target_critic.parameters():
            params.requires_grad = False

        self.actor_optimizer = optm.Adam \
            (self.actor.parameters(), lr=self.actor_lr, weight_decay=self.weight_decay)
        self.critic_optimizer = optm.Adam \
            (self.critic.parameters(), lr=self.critic_lr, weight_decay=self.weight_decay)

        if self.adaptive_alpha:
            self.target_entropy = torch.tensor(-self.action_dim * self.max_nodes,
                                               dtype=float,
                                               requires_grad=True,
                                               device=self.device)
            self.log_alpha = torch.tensor(np.log(self.alpha),
                                          dtype=torch.float,
                                          requires_grad=True,
                                          device=self.device)
            self.alpha_optimizer = optm.Adam \
                ([self.log_alpha], lr=self.alpha_lr)