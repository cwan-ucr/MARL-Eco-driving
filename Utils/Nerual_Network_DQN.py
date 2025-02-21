import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# 训练神经网络，包括策略网络，两个Q函数网络
# 策略网络
class Policy_Network(nn.Module):
    def __init__(self,
                 feature_dim,
                 hidden_dim,
                 action_dim,
                 action_bound,
                 dropout,
                 beta):
        # 定义GAT神经网络结构
        super(Policy_Network, self).__init__()
        self.dropout = dropout
        self.beta = beta
        self.hidden_dim = hidden_dim

        self.L1 = nn.Linear(feature_dim, hidden_dim)
        self.LN1_MLP = nn.LayerNorm(hidden_dim, bias=False)
        self.L2 = nn.Linear(hidden_dim, hidden_dim)

        self.LN2 = nn.LayerNorm(hidden_dim, bias=False)

        self.L3 = nn.Linear(hidden_dim, hidden_dim )
        self.LN3 = nn.LayerNorm(hidden_dim, bias=False)

        self.L4 = nn.Linear(hidden_dim, action_dim * 2)

        self.ResNet = nn.Linear(feature_dim, action_dim * 2, bias=False)

        self.action_bound = action_bound
        self.action_dim = action_dim

    def forward(self, x, mask):

        # Linear 1
        x1 = self.L1(x) * mask.unsqueeze(-1)
        x1 = self.LN1_MLP(x1)
        x1 = F.leaky_relu(x1, self.beta)

        # Linear 2
        x2 = self.L2(x1) * mask.unsqueeze(-1)
        x2 = self.LN2(x2)
        x2 = F.leaky_relu(x2, self.beta)

        # Linear 3
        x3 = self.L3(x2) * mask.unsqueeze(-1)
        x3 = self.LN3(x3)
        x3 = F.leaky_relu(x3, self.beta)

        # Linear 4
        x_output = self.L4(x3) + self.ResNet(x)


        # 拆分输出为均值和标准差
        if x_output.shape[1] == self.action_dim * 2:
            x_mu = x_output[:, :self.action_dim]
            x_log_std = torch.clamp(x_output[:, self.action_dim:],
                                    min=-20,
                                    max=2)
        else:
            x_mu = x_output[:, :, :self.action_dim]
            x_log_std = torch.clamp(x_output[:, :, self.action_dim:],
                                    min=-20,
                                    max=2)

        x_std = torch.exp(x_log_std + 1e-6)

        if torch.any(torch.isnan(x_mu)):
            print("x_mu contains NaN values")
            x_mu = torch.nan_to_num(x_mu, nan=1e-3)  # 替换 NaN
        if torch.any(torch.isnan(x_std)):
            print("x_std contains NaN values")
            x_std = torch.nan_to_num(x_mu, nan=1.0) + 1e-6  # 确保标准差为正

        dist = Normal(x_mu, x_std)
        normal_sample = dist.rsample()
        action = torch.tanh(normal_sample)
        log_prob = dist.log_prob(normal_sample)

        # 掩码机制，将无效的log_prob置为0
        mask = mask.unsqueeze(-1)

        # 修正log_prob, 根据tanh变换
        log_prob -= torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob * mask

        if log_prob.shape[1] == self.action_dim:
            log_prob = (log_prob * mask).sum(dim=1, keepdim=True)
        else:
            log_prob = (log_prob * mask).sum(dim=2, keepdim=True)

        action = self.action_bound * action

        action = action.to(dtype=torch.float32)

        return action, log_prob.squeeze(-1)

class Critic_Network(nn.Module):
    def __init__(self,
                 feature_dim,
                 hidden_dim,
                 action_dim,
                 dropout,
                 beta):
        super(Critic_Network, self).__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.beta = beta

        self.Q_1 = QValue_Network(feature_dim,
                                  hidden_dim,
                                  action_dim,
                                  dropout,
                                  beta)

        self.Q_2 = QValue_Network(feature_dim,
                                  hidden_dim,
                                  action_dim,
                                  dropout,
                                  beta)

    def forward(self, x, a, mask):
        Q1 = self.Q_1(x, a, mask)
        Q2 = self.Q_2(x, a, mask)
        return Q1, Q2

    def reset_parameters(self):
        self.Q_1.reset()
        self.Q_2.reset()


# Q价值网络
class QValue_Network(nn.Module):
    def __init__(self,
                 feature_dim,
                 hidden_dim,
                 action_dim,
                 dropout,
                 beta):
        super(QValue_Network, self).__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.beta = beta

        self.L1 = nn.Linear(feature_dim + action_dim, hidden_dim)
        self.LN1_MLP = nn.LayerNorm(hidden_dim, bias=False)
        self.L2 = nn.Linear(hidden_dim, hidden_dim)

        self.LN_2 = nn.LayerNorm(hidden_dim, bias=False)

        self.L3 = nn.Linear(hidden_dim, hidden_dim)
        self.LN3 = nn.LayerNorm(hidden_dim, bias=False)

        self.L4 = nn.Linear(hidden_dim, 1)
        self.ResNet = nn.Linear(feature_dim + action_dim, 1, bias=False)

    def forward(self, x, a, mask):
        # State and Action concat
        if x.shape[1] == self.feature_dim:
            cat = torch.cat([x, a], dim=1)
        else:
            cat = torch.cat([x, a], dim=2)

        # Linear 1
        x1 = self.L1(cat) * mask.unsqueeze(-1)
        x1 = self.LN1_MLP(x1)
        x1 = F.leaky_relu(x1, self.beta)

        # Linear 2
        x2 = self.L2(x1) * mask.unsqueeze(-1)
        x2 = self.LN_2(x2)
        x2 = F.leaky_relu(x2, self.beta)

        # Linear 3
        x3 = self.L3(x2) * mask.unsqueeze(-1)
        x3 = self.LN3(x3)
        x3 = F.leaky_relu(x3, self.beta)

        x_output = self.L4(x3)
        x_output = x_output + self.ResNet(cat)

        if x2.shape[1] == 1:
            q_nodes = x_output.t() * mask
        else:
            q_nodes = x_output * mask.unsqueeze(-1)

        return q_nodes.squeeze(-1)