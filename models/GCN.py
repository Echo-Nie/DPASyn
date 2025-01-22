import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GCNConv, SAGPooling
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
import numpy as np


class AttenSyn(nn.Module):
    def __init__(
        self,
        molecule_channels: int = 78,  # 分子特征维度
        hidden_channels: int = 128,   # 隐藏层维度
        middle_channels: int = 64,    # 中间层维度
        layer_count: int = 2,         # GCN层数
        out_channels: int = 2,        # 输出维度
        dropout_rate: float = 0.2     # Dropout率
    ):
        super().__init__()
        # GCN层列表
        self.graph_convolutions = torch.nn.ModuleList()
        self.graph_convolutions.append(GCNConv(molecule_channels, hidden_channels))
        for _ in range(1, layer_count):
            self.graph_convolutions.append(GCNConv(hidden_channels, hidden_channels))

        # LSTM层
        self.border_rnn = torch.nn.LSTM(hidden_channels, hidden_channels, 1)

        # 最终的全连接层
        self.final = torch.nn.Sequential(
            torch.nn.Linear(4 * hidden_channels + 256, middle_channels),  # 瓶颈层
            torch.nn.ReLU(),
            torch.nn.Linear(middle_channels, out_channels),  # 输出层
        )

        # 特征降维模块1
        self.reduction = nn.Sequential(
            nn.Linear(954, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # 特征降维模块2
        self.reduction2 = nn.Sequential(
            nn.Linear(954, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 78),
            nn.ReLU()
        )

        # 注意力池化层
        self.pool1 = Attention(hidden_channels, 4)
        self.pool2 = Attention(hidden_channels, 4)

    def _forward_molecules(self, conv, x, edge_index, batch, states) -> torch.FloatTensor:
        # GCN前向传播
        gcn_hidden = conv(x, edge_index)
        gcn_hidden_detach = gcn_hidden.detach()

        # LSTM前向传播
        rnn_out, (hidden_state, cell_state) = self.border_rnn(gcn_hidden_detach[None, :, :], states)
        rnn_out = rnn_out.squeeze()

        return gcn_hidden, rnn_out, (hidden_state, cell_state)

    def forward(self, molecules_left, molecules_right) -> torch.FloatTensor:
        # 提取左分子和右分子的特征
        x1, edge_index1, batch1, cell, mask1 = molecules_left.x, molecules_left.edge_index, molecules_left.batch, molecules_left.cell, molecules_left.mask
        x2, edge_index2, batch2, mask2 = molecules_right.x, molecules_right.edge_index, molecules_right.batch, molecules_right.mask

        # 对cell特征进行归一化
        cell = F.normalize(cell, 2, 1)
        cell_expand = self.reduction2(cell)  # 降维
        cell = self.reduction(cell)  # 降维

        # 扩展cell特征
        cell_expand = cell_expand.unsqueeze(1)
        cell_expand = cell_expand.expand(cell.shape[0], 100, -1)
        cell_expand = cell_expand.reshape(-1, 78)

        # 调整mask形状
        batch_size = torch.max(molecules_left.batch) + 1
        mask1 = mask1.reshape((batch_size, 100))
        mask2 = mask2.reshape((batch_size, 100))

        # 初始化状态
        left_states, right_states = None, None
        gcn_hidden_left = molecules_left.x + cell_expand
        gcn_hidden_right = molecules_right.x + cell_expand

        # 应用GCN和LSTM
        for conv in self.graph_convolutions:
            gcn_hidden_left, rnn_out_left, left_states = self._forward_molecules(
                conv, gcn_hidden_left, molecules_left.edge_index, molecules_left.batch, left_states
            )
            gcn_hidden_right, rnn_out_right, right_states = self._forward_molecules(
                conv, gcn_hidden_right, molecules_right.edge_index, molecules_right.batch, right_states
            )

        # 调整形状并进行注意力池化
        rnn_out_left, rnn_out_right = rnn_out_left.reshape(batch_size, 100, -1), rnn_out_right.reshape(batch_size, 100, -1)
        rnn_pooled_left, rnn_pooled_right = self.pool1(rnn_out_left, rnn_out_right, (mask1, mask2))

        gcn_hidden_left, gcn_hidden_right = gcn_hidden_left.reshape(batch_size, 100, -1), gcn_hidden_right.reshape(batch_size, 100, -1)
        gcn_hidden_left, gcn_hidden_right = self.pool2(gcn_hidden_left, gcn_hidden_right, (mask1, mask2))

        # 拼接特征
        shared_graph_level = torch.cat([gcn_hidden_left, gcn_hidden_right], dim=1)
        out = torch.cat([shared_graph_level, rnn_pooled_left, rnn_pooled_right, cell], dim=1)

        # 最终输出
        out = self.final(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(Attention, self).__init__()
        self.num_heads = num_heads  # 注意力头数
        self.dim_per_head = dim // num_heads  # 每个头的维度

        # 线性变换层
        self.linear_q = nn.Linear(dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(dim, self.dim_per_head * num_heads)

        # 归一化层
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=0.2)  # Dropout层

    def attention(self, q1, k1, v1, q2, k2, v2, attn_mask=None):
        # 计算注意力分数
        a1 = torch.tanh(torch.bmm(k1, q2.transpose(1, 2)))
        a2 = torch.tanh(torch.bmm(k2, q1.transpose(1, 2)))

        if attn_mask is not None:
            # 如果有掩码，应用掩码
            mask1, mask2 = attn_mask
            a1 = torch.softmax(torch.sum(a1, dim=2).masked_fill(mask1, -np.inf), dim=-1).unsqueeze(dim=1)
            a2 = torch.softmax(torch.sum(a2, dim=2).masked_fill(mask2, -np.inf), dim=-1).unsqueeze(dim=1)
        else:
            # 如果没有掩码，直接计算softmax
            a1 = torch.softmax(torch.sum(a1, dim=2), dim=1).unsqueeze(dim=1)
            a2 = torch.softmax(torch.sum(a2, dim=2), dim=1).unsqueeze(dim=1)

        # 应用Dropout
        a1 = self.dropout(a1)
        a2 = self.dropout(a2)

        # 计算加权后的特征向量
        vector1 = torch.bmm(a1, v1).squeeze()
        vector2 = torch.bmm(a2, v2).squeeze()

        return vector1, vector2

    def forward(self, fingerprint_vectors1, fingerprint_vectors2, attn_mask=None):
        # 线性变换
        q1, q2 = torch.relu(self.linear_q(fingerprint_vectors1)), torch.relu(self.linear_q(fingerprint_vectors2))
        k1, k2 = torch.relu(self.linear_k(fingerprint_vectors1)), torch.relu(self.linear_k(fingerprint_vectors2))
        v1, v2 = torch.relu(self.linear_v(fingerprint_vectors1)), torch.relu(self.linear_v(fingerprint_vectors2))

        # 计算注意力
        vector1, vector2 = self.attention(q1, k1, v1, q2, k2, v2, attn_mask)

        # 归一化并返回结果
        vector1 = self.norm(torch.mean(fingerprint_vectors1, dim=1) + vector1)
        vector2 = self.norm(torch.mean(fingerprint_vectors2, dim=1) + vector2)

        return vector1, vector2