import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 原始图注意力层
class GAT(nn.Module):
    """
    Simple MARL layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_L = nn.Linear(out_features, 1, bias=False)
        self.a_R = nn.Linear(out_features, 1, bias=False)

        self.leaky_ReLU = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = self.W(h) # h.shape: (Batch_size, N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input_gat(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        e = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(e, dim=1)
        attention = attention * adj
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)


        return h_prime

    def _prepare_attentional_mechanism_input_gat(self, Wh):
        # Wh.shape (N, out_feature)
        # Adj.shape (Batch_size, N, N)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (Batch_size, N, 1)
        # e.shape (Bacth_size, N, N)

        e_ij = self.a_L(Wh)
        e_ji = self.a_R(Wh)

        # broadcast add
        if e_ij.shape[1] == 1:
            e = e_ij + e_ji.T
        else:
            e = e_ij + e_ji.permute(0, 2, 1)

        return self.leaky_ReLU(e)

# 图注意力机制, dot product, 注意力eij = Wh1*Wh2
class GAT_DP(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GAT_DP, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)

    def forward(self, h, adj):
        """
        :param h: (Batch_size, N, in_features)
        :param adj: (Batch_size, N, N)
        :return: (Batch_size, N, out_features)
        """
        Wh = self.W(h) # h.shape: (Batch_size, N, in_features), Wh.shape: (N, out_features)

        if Wh.shape[1] == self.out_features:
            e = torch.matmul(Wh, Wh.T) # (Batch_size, N, N)
        else:
            e = torch.matmul(Wh, Wh.permute(0, 2, 1)) # (Batch_size, N, N)

        e = e / (self.out_features ** 0.5)

        zero_vec = -9e15*torch.ones_like(e)
        e = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(e, dim=1)
        attention = attention * adj
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)


        return h_prime

# 图注意力机制，利用Q和K的点积计算注意力，再乘以V, e_ij = qi^T*kj
class GAT_QKV(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GAT_QKV, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.Q = nn.Linear(in_features, out_features, bias=False)
        self.K = nn.Linear(in_features, out_features, bias=False)
        self.V = nn.Linear(in_features, out_features, bias=False)

    def forward(self, h, adj):
        """
        :param h: (Batch_size, N, in_features)
        :param adj: (Batch_size, N, N)
        :return: (Batch_size, N, out_features)
        """

        Q = self.Q(h) # (Batch_size, N, out_features)
        K = self.K(h) # (Batch_size, N, out_features)
        V = self.V(h) # (Batch_size, N, out_features)

        # 1. 用Q和K的点积计算注意力
        if K.shape[1] == self.out_features:
            e = torch.matmul(Q, K.T) # (Batch_size, N, N)
            e = e / (self.out_features ** 0.5)
        else:
            e = torch.matmul(Q, K.permute(0, 2, 1)) # (Batch_size, N, N)
            e = e / (self.out_features ** 0.5)

        # 2. 将注意力归一化
        zero_vec = -9e15*torch.ones_like(e)
        e = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(e, dim=1)
        attention = attention * adj
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 3. 将注意力乘以V
        h_prime = torch.matmul(attention, V) # (Batch_size, N, out_features)

        return h_prime# (Batch_size, N, out_features)


# 图卷积网络
class GCN(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)

    def forward(self, h, adj):
        """
        :param h: (Batch_size, N, in_features)
        :param adj: (Batch_size, N, N)
        :return: (Batch_size, N, out_features)
        """
        Wh = self.W(h)  # h.shape: (Batch_size, N, in_features), Wh.shape: (N, out_features)

        # 1. 计算聚合矩阵 A = D^-0.5 * A * D^-0.5, D是度矩阵: D = sum(adj, dim=1)
        D = torch.sum(adj, dim=1) # (Batch_size, N)
        D_inv = torch.pow(D, -0.5) # (Batch_size, N)
        D_inv[D_inv == float('inf')] = 0
        D_inv = torch.diag_embed(D_inv) # (Batch_size, N, N)

        A = torch.matmul(torch.matmul(D_inv, adj), D_inv) # (Batch_size, N, N)
        A = F.dropout(A, self.dropout, training=self.training)

        # 2. 用A矩阵和Wh计算输出
        h_prime = torch.matmul(A, Wh) # (Batch_size, N, out_features)

        return h_prime # (Batch_size, N, out_features)

