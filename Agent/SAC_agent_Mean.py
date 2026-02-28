import numpy as np
import torch
import torch.optim as optm
import torch.nn.functional as F
import copy

from Nerual_Network import Policy_Network, Critic_Network
from Replay_buffer import replay_buffer


# SAC Agent
class SAC_agent:

    def __init__(self, **kwargs):
        # 参数继承
        self.__dict__.update(kwargs)

        del self.CAV_PR

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
        self.alpha_loss = []

        # 定义温度系数(alpha)
        if self.adaptive_alpha:
            # 目标熵，一般为动作维度的负数
            self.target_entropy = torch.tensor(-self.action_dim * 10,
                                               dtype=float,
                                               requires_grad=True,
                                               device=self.device)
            self.log_alpha = torch.tensor(np.log(self.alpha),
                                          dtype=torch.float,
                                          requires_grad=True,
                                          device=self.device)
            self.alpha_optimizer = optm.Adam \
                ([self.log_alpha], lr=self.alpha_lr)

    # 动作函数
    def take_action(self, state_, mask):
        with torch.no_grad():
            state = torch.from_numpy(state_).to(dtype=torch.float32, device=self.device)
            mask = torch.from_numpy(mask).to(dtype=torch.int32, device=self.device)
            action, _ = self.actor(state, mask)
            action = action.to(dtype=torch.float32).cpu().detach().numpy()

        return action.astype(np.float32)

    # Q值函数评估
    def calc_target(self, rewards, next_state, next_mask, dones):

        next_actions, next_log_prob = self.actor(next_state, next_mask)
        entropy = -next_log_prob
        q1_value, q2_value = self.target_critic(next_state, next_actions, next_mask)

        target_q = rewards.sum(dim=-1).unsqueeze(-1) + self.gamma * (~dones) *  \
                   (torch.min(q1_value, q2_value) +
                    self.log_alpha.exp() * entropy)

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

        # 随机采样
        # state, mask, action, reward, done, state_next, mask_next
        states, mask, actions, rewards, dones, next_states, mask_next = \
            self.replay_buffer.sample()

        # TD error估计, Q网络更新(critic)
        # ----------------------------- ↓↓↓↓↓ Update QValue Net ↓↓↓↓↓ ------------------------------#
        Q_target = self.calc_target(rewards, next_states, mask_next, dones)
        Q_target = Q_target * mask

        Q1_current, Q2_current = self.critic(states, actions, mask)

        # Multi-agent cooperation: sum Q first, then mean
        td_error = (F.mse_loss(Q_target.sum(dim=-1), Q1_current.sum(dim=-1))
                    + F.mse_loss(Q_target.sum(dim=-1), Q2_current.sum(dim=-1)))

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
            new_actions, new_log_prob = self.actor(states, mask)

            Q1_value, Q2_value = self.critic(states, new_actions, mask)
            min_q = torch.min(Q1_value, Q2_value)

            actor_loss = (self.log_alpha.exp() * new_log_prob.sum(dim=-1) - min_q.sum(dim=-1)).mean()
            self.actor_loss.append(actor_loss.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.actor_optimizer.step()

            for params in self.critic.parameters(): params.requires_grad = True

            # 温度系数更新(alpha)
            # ----------------------------- ↓↓↓↓↓ Update Alpha ↓↓↓↓↓ ------------------------------#
            if self.Dynamic_target_entropy:
                self.target_entropy = -mask_next.sum(dim=-1)

            if self.adaptive_alpha:
                alpha_loss = -(self.log_alpha *
                               (new_log_prob.sum(dim=-1) + self.target_entropy).detach()).mean()
                self.alpha_loss.append(alpha_loss.item())

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            # 目标网络软更新
            self.soft_update(self.critic, self.target_critic)

    # 模型存储
    def save(self, timestep, CAV_PR, time_step, NN_reset, Hidden_dim):
        EnvName = self.Env_name

        torch.save(self.actor.state_dict(), "./model./{}_actor_{}_time_step_{}_hidden_dim_{}_NN_Reset_{}_CAV_PR_{}_rand.pth"
                                   .format(EnvName, timestep, time_step, Hidden_dim, NN_reset, CAV_PR))
        torch.save(self.critic.state_dict(), "./model./{}_critic_{}_time_step_{}_hidden_dim_{}_NN_Reset_{}_CAV_PR_{}_rand.pth"
                                   .format( EnvName, timestep, time_step, Hidden_dim, NN_reset, CAV_PR))

    # 模型加载
    def load(self, epoch, time_step, CAV_PR, NN_reset):
        EnvName = self.Env_name

        state_dict_before = self.actor.state_dict()
        actor_state_dict = torch.load("./model./{}_actor_{}_time_step_{}_NN_Reset_{}_CAV_PR_{}_rand.pth"
                                .format(EnvName, epoch, time_step, NN_reset, CAV_PR),
                                    weights_only=False)
        critic_state_dict = torch.load("./model./{}_critic_{}_time_step_{}_NN_Reset_{}_CAV_PR_{}_rand.pth"
                                .format(EnvName, epoch, time_step, NN_reset, CAV_PR),
                                    weights_only=False)
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