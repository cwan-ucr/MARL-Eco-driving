import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math

# 训练神经网络，包括策略网络，两个Q函数网络
# 策略网络
class Policy_Network(nn.Module):
    def __init__(self,
                 feature_dim,
                 hidden_dim,
                 action_dim,
                 action_bound,
                 dropout,
                 beta,
                 training):
        # 定义神经网络结构
        super(Policy_Network, self).__init__()
        self.dropout = dropout
        self.beta = beta
        self.hidden_dim = hidden_dim
        self.training = training

        self.L1 = nn.Linear(feature_dim, hidden_dim)
        self.LN1_MLP = nn.RMSNorm(hidden_dim)
        self.L2 = nn.Linear(hidden_dim , hidden_dim)

        self.LN2 = nn.RMSNorm(hidden_dim)

        self.L3 = nn.Linear(hidden_dim, hidden_dim )
        self.LN3 = nn.RMSNorm(hidden_dim)

        self.L4 = nn.Linear(hidden_dim, action_dim * 2)

        self.ResNet = nn.Linear(feature_dim, action_dim * 2)

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
        x_output = self.L4(x3)


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
        u = dist.rsample() if self.training else x_mu
        action = torch.tanh(u)
        log_prob = dist.log_prob(u)

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
                 beta,
                 RL_agent):
        super(Critic_Network, self).__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.beta = beta
        self.RL_agent = RL_agent

        self.Q_1 = QValue_Network(feature_dim,
                                  hidden_dim,
                                  action_dim,
                                  dropout,
                                  beta,
                                  self.RL_agent)

        self.Q_2 = QValue_Network(feature_dim,
                                  hidden_dim,
                                  action_dim,
                                  dropout,
                                  beta,
                                  self.RL_agent)

    def forward(self, x, a, mask):
        Q1 = self.Q_1(x, a, mask)
        Q2 = self.Q_2(x, a, mask)
        return Q1, Q2

    def reset_parameters(self):
        self.Q_1.reset()
        self.Q_2.reset()


# Q价值网络
class QValue_Network(nn.Module):
    def __init__(self, feature_dim, hidden_dim, action_dim, dropout, beta, RL_agent):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.beta = beta
        self.RL_agent = RL_agent

        self.attention = True if RL_agent == 'QMHA' else False

        if self.attention:
            # 1) state embedding: (B,N,feature_dim) -> (B,N,hidden_dim/2)
            self.state_emb = nn.Linear(feature_dim, hidden_dim // 2)
            self.state_norm = nn.RMSNorm(hidden_dim // 2)

            # 2) neighbor attention on state_emb only, output (B,N,hidden_dim)
            self.nei_attn = NeighborAttentionConcat(
                d_model=hidden_dim // 2,
                n_heads=2,
                dropout=dropout,
                max_len=80,
                use_sincos_pos=False,
            )
            self.nei_norm = nn.RMSNorm(hidden_dim)

            # 3) combine with action: (hidden_dim + action_dim) -> hidden_dim
            self.L3 = nn.Linear(hidden_dim + action_dim, hidden_dim)
            self.LN3 = nn.RMSNorm(hidden_dim)

            self.L4 = nn.Linear(hidden_dim, 1)

            # residual from (state,action) direct to Q
            self.ResNet = nn.Linear(feature_dim + action_dim, 1)

        else:
            # ===== 原来的无 attention 分支（保持不变）=====
            self.L1 = nn.Linear(feature_dim + action_dim, hidden_dim)
            self.LN1 = nn.RMSNorm(hidden_dim)

            self.L2 = nn.Linear(hidden_dim, hidden_dim)
            self.LN_2 = nn.RMSNorm(hidden_dim)

            self.L3 = nn.Linear(hidden_dim, hidden_dim)
            self.LN3 = nn.RMSNorm(hidden_dim)

            self.L4 = nn.Linear(hidden_dim, 1)
            self.ResNet = nn.Linear(feature_dim + action_dim, 1)

    def forward(self, x, a, mask):
        """
        x: (B,N,feature_dim)
        a: (B,N,action_dim)
        mask: (B,N) 1=valid
        """
        mask_ = mask.unsqueeze(-1).to(x.dtype)

        if self.attention:
            # ---- state branch ----
            s = self.state_emb(x) * mask_
            s = self.state_norm(s)
            s = F.leaky_relu(s, self.beta)

            # ---- neighbor context from state only ----
            h = self.nei_attn(s, mask)          # (B,N,hidden_dim) = concat([s, ctx])
            h = self.nei_norm(h)
            h = F.leaky_relu(h, self.beta)

            # ---- now inject action AFTER neighbor aggregation ----
            ha = torch.cat([h, a], dim=-1)      # (B,N,hidden_dim + action_dim)

            x3 = self.L3(ha) * mask_
            x3 = self.LN3(x3)
            x3 = F.leaky_relu(x3, self.beta)

            q = self.L4(x3)                     # (B,N,1)
            q = q + self.ResNet(torch.cat([x, a], dim=-1))

            q_nodes = q * mask_
            return q_nodes.squeeze(-1)

        # ===== 原来的无 attention =====
        cat = torch.cat([x, a], dim=-1)

        x1 = self.L1(cat) * mask_
        x1 = self.LN1(x1)
        x1 = F.leaky_relu(x1, self.beta)

        x2 = self.L2(x1) * mask_
        x2 = self.LN_2(x2)
        x2 = F.leaky_relu(x2, self.beta)

        x3 = self.L3(x2) * mask_
        x3 = self.LN3(x3)
        x3 = F.leaky_relu(x3, self.beta)

        q = self.L4(x3) + self.ResNet(cat)
        q = self.L4(x3)
        q_nodes = q * mask_
        return q_nodes.squeeze(-1)
# class QValue_Network(nn.Module):
#     def __init__(self,
#                  feature_dim,
#                  hidden_dim,
#                  action_dim,
#                  dropout,
#                  beta,
#                  RL_agent):
#         super(QValue_Network, self).__init__()
#         self.feature_dim = feature_dim
#         self.action_dim = action_dim
#         self.hidden_dim = hidden_dim
#         self.dropout = dropout
#         self.beta = beta
#         self.RL_agent = RL_agent
        
#         self.attention = True if RL_agent == 'QMHA' else False

#         self.L1 = nn.Linear(feature_dim + action_dim, hidden_dim)
#         self.LN1 = nn.RMSNorm(hidden_dim)

#         self.L2 = nn.Linear(hidden_dim, hidden_dim)
#         self.LN_2 = nn.RMSNorm(hidden_dim)
        
#         if self.attention:
#             self.L1 = nn.Linear(feature_dim + action_dim, int(hidden_dim/2))
#             self.LN1 = nn.RMSNorm(int(hidden_dim/2))
#             self.L2 = NeighborAttentionConcat(d_model=int(hidden_dim/2), n_heads=1)

#         self.L3 = nn.Linear(hidden_dim, hidden_dim)
#         self.LN3 = nn.RMSNorm(hidden_dim)

#         self.L4 = nn.Linear(hidden_dim, 1)
#         self.ResNet = nn.Linear(feature_dim + action_dim, 1)

#     def forward(self, x, a, mask):
#         # State and Action concat
#         cat = torch.cat([x, a], dim=-1)

#         mask_ = mask.unsqueeze(-1)
#         # Linear 1
#         x1 = self.L1(cat) * mask_
#         x1 = self.LN1(x1)
#         x1 = F.leaky_relu(x1, self.beta)

#         # Linear 2
#         if self.attention:
#             x2 = self.L2(x1, mask)
#             x2 = self.LN_2(x2)
#         else:
#             x2 = self.L2(x1) * mask_
#         x2 = F.leaky_relu(x2, self.beta)

#         # Linear 3
#         x3 = self.L3(x2) * mask_
#         x3 = self.LN3(x3)
#         x3 = F.leaky_relu(x3, self.beta)

#         x_output = self.L4(x3)
#         x_output = x_output + self.ResNet(cat)

#         q_nodes = x_output * mask_

#         return q_nodes.squeeze(-1)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 80):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # (L,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,L,D)

    def forward(self, x: torch.Tensor):
        # x: (B,N,D)
        N = x.size(1)
        return x + self.pe[:, :N, :]


class NeighborAttentionConcat(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        max_len: int = 80,
        use_sincos_pos: bool = False,
        topk: int | None = None   # ✅ 新增：只看 top-k
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.topk = topk

        self.use_sincos_pos = use_sincos_pos
        if use_sincos_pos:
            self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        self.Wq = nn.Linear(d_model, d_model, bias=bias)
        self.Wk = nn.Linear(d_model, d_model, bias=bias)
        self.Wv = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, return_attn: bool = False):
        B, N, D = x.shape
        H, Dh = self.n_heads, self.dh

        x_in = x
        if self.use_sincos_pos:
            x = self.pos(x)

        q = self.Wq(x).view(B, N, H, Dh).transpose(1, 2)  # (B,H,N,Dh)
        k = self.Wk(x).view(B, N, H, Dh).transpose(1, 2)
        v = self.Wv(x).view(B, N, H, Dh).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)  # (B,H,N,N)

        neg_large = torch.finfo(scores.dtype).min

        # 1) 禁止 self-attention（对角线）
        diag = torch.eye(N, device=x.device, dtype=torch.bool)[None, None, :, :]
        scores = scores.masked_fill(diag, neg_large)

        # 2) padding key mask（列）
        if mask is not None:
            key_mask = mask[:, None, None, :].to(torch.bool)  # (B,1,1,N)
            scores = scores.masked_fill(~key_mask, neg_large)

        # 3) ✅ top-k sparsify（只保留 top-k keys）
        if self.topk is not None and self.topk > 0:
            # 有效 key 数可能 < topk，所以 k 取 min
            if mask is None:
                k_eff = min(self.topk, N - 1)  # 减 1 是因为去掉了 self
            else:
                # 每个样本有效节点数不同，k_eff 取一个安全上界
                max_valid = int(mask.sum(dim=1).max().item())
                k_eff = min(self.topk, max(1, max_valid - 1))

            # topk indices: (B,H,N,k_eff)
            topk_idx = torch.topk(scores, k=k_eff, dim=-1).indices

            # 构造 keep mask: (B,H,N,N)
            keep = torch.zeros_like(scores, dtype=torch.bool)
            keep.scatter_(-1, topk_idx, True)

            # 把非 top-k 全部置为极小值
            scores = scores.masked_fill(~keep, neg_large)

        # 4) 避免整行全 neg_large -> softmax NaN（比如只剩自己、或有效节点太少）
        all_masked = (scores == neg_large).all(dim=-1, keepdim=True)  # (B,H,N,1)
        scores = torch.where(all_masked, torch.zeros_like(scores), scores)

        attn = torch.softmax(scores, dim=-1)  # (B,H,N,N)
        attn = self.dropout(attn)

        neigh_h = attn @ v  # (B,H,N,Dh)
        neigh = neigh_h.transpose(1, 2).contiguous().view(B, N, D)
        v = v.transpose(1, 2).contiguous().view(B, N, D)

        if mask is not None:
            neigh = neigh * mask[:, :, None].to(neigh.dtype)

        y = torch.cat([v, v + 0.1*neigh], dim=-1)  # (B,N,2D)

        if return_attn:
            return y, attn
        return y