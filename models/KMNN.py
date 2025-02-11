import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GCNConv
import numpy as np


class DualAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(DualAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = dim // num_heads

        self.linear_q = nn.Linear(dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(dim, self.dim_per_head * num_heads)

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=0.2)

    def attention(self, q1, k1, v1, q2, k2, v2, attn_mask=None):
        a1 = torch.tanh(torch.bmm(k1, q2.transpose(1, 2)))
        a2 = torch.tanh(torch.bmm(k2, q1.transpose(1, 2)))

        if attn_mask is not None:
            mask1, mask2 = attn_mask
            a1 = torch.softmax(torch.sum(a1, dim=2).masked_fill(mask1, -np.inf), dim=-1).unsqueeze(dim=1)
            a2 = torch.softmax(torch.sum(a2, dim=2).masked_fill(mask2, -np.inf), dim=-1).unsqueeze(dim=1)
        else:
            a1 = torch.softmax(torch.sum(a1, dim=2), dim=1).unsqueeze(dim=1)
            a2 = torch.softmax(torch.sum(a2, dim=2), dim=1).unsqueeze(dim=1)

        a1 = self.dropout(a1)
        a2 = self.dropout(a2)

        vector1 = torch.bmm(a1, v1).squeeze()
        vector2 = torch.bmm(a2, v2).squeeze()

        return vector1, vector2

    def forward(self, x1, x2, attn_mask=None):
        q1, k1, v1 = torch.relu(self.linear_q(x1)), torch.relu(self.linear_k(x1)), torch.relu(self.linear_v(x1))
        q2, k2, v2 = torch.relu(self.linear_q(x2)), torch.relu(self.linear_k(x2)), torch.relu(self.linear_v(x2))

        vector1, vector2 = self.attention(q1, k1, v1, q2, k2, v2, attn_mask)

        vector1 = self.norm(torch.mean(x1, dim=1) + vector1)
        vector2 = self.norm(torch.mean(x2, dim=1) + vector2)

        return vector1, vector2


class AttenSyn(nn.Module):
    def __init__(
            self,
            molecule_channels: int = 78,
            hidden_channels: int = 128,
            middle_channels: int = 64,
            layer_count: int = 2,
            out_channels: int = 2,
            dropout_rate: float = 0.2,
            heads: int = 4,
    ):
        super().__init__()

        self.graph_convolutions = torch.nn.ModuleList()
        self.graph_convolutions.append(GATConv(molecule_channels, hidden_channels // heads, heads=heads))
        for _ in range(1, layer_count):
            self.graph_convolutions.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))

        self.border_rnn = torch.nn.LSTM(hidden_channels, hidden_channels, 1)

        # 添加残差连接相关层
        self.residual1 = nn.Linear(hidden_channels, hidden_channels)
        self.residual2 = nn.Linear(hidden_channels * 2, hidden_channels * 2)

        self.reduction = nn.Sequential(
            nn.Linear(954, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.reduction2 = nn.Sequential(
            nn.Linear(954, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 78),
            nn.ReLU(),
        )

        self.pool1 = DualAttention(hidden_channels, num_heads=heads)
        self.pool2 = DualAttention(hidden_channels, num_heads=heads)

        self.final = nn.Sequential(
            nn.Linear(4 * hidden_channels + 256, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )

    def _forward_molecules(self, conv, x, edge_index, batch, states):
        gcn_hidden = conv(x, edge_index)
        rnn_out, (hidden_state, cell_state) = self.border_rnn(gcn_hidden.unsqueeze(0), states)
        rnn_out = rnn_out.squeeze(0)

        # 应用残差连接
        gcn_hidden = F.relu(gcn_hidden + self.residual1(x))  # 添加残差连接

        return gcn_hidden, rnn_out, (hidden_state, cell_state)

    def forward(self, molecules_left, molecules_right):
        x1, edge_index1, batch1, cell, mask1 = (
            molecules_left.x,
            molecules_left.edge_index,
            molecules_left.batch,
            molecules_left.cell,
            molecules_left.mask,
        )
        x2, edge_index2, batch2, mask2 = (
            molecules_right.x,
            molecules_right.edge_index,
            molecules_right.batch,
            molecules_right.mask,
        )
        cell = F.normalize(cell, 2, 1)
        cell_expand = self.reduction2(cell).unsqueeze(1)
        cell_expand = cell_expand.expand(cell.shape[0], 100, -1).reshape(-1, 78)
        cell = self.reduction(cell)

        batch_size = torch.max(molecules_left.batch) + 1
        mask1, mask2 = mask1.reshape(batch_size, 100), mask2.reshape(batch_size, 100)
        left_states, right_states = None, None
        gcn_hidden_left = molecules_left.x + cell_expand
        gcn_hidden_right = molecules_right.x + cell_expand

        for conv in self.graph_convolutions:
            gcn_hidden_left, rnn_out_left, left_states = self._forward_molecules(
                conv, gcn_hidden_left, molecules_left.edge_index, molecules_left.batch, left_states
            )
            gcn_hidden_right, rnn_out_right, right_states = self._forward_molecules(
                conv, gcn_hidden_right, molecules_right.edge_index, molecules_right.batch, right_states
            )

        rnn_out_left, rnn_out_right = (
            rnn_out_left.reshape(batch_size, 100, -1),
            rnn_out_right.reshape(batch_size, 100, -1),
        )
        rnn_pooled_left, rnn_pooled_right = self.pool1(rnn_out_left, rnn_out_right, (mask1, mask2))
        gcn_hidden_left, gcn_hidden_right = (
            gcn_hidden_left.reshape(batch_size, 100, -1),
            gcn_hidden_right.reshape(batch_size, 100, -1),
        )
        gcn_hidden_left, gcn_hidden_right = self.pool2(gcn_hidden_left, gcn_hidden_right, (mask1, mask2))
        shared_graph_level = torch.cat([gcn_hidden_left, gcn_hidden_right], dim=1)

        # 添加残差连接
        shared_graph_level = F.relu(
            shared_graph_level + self.residual2(torch.cat([rnn_pooled_left, rnn_pooled_right], dim=1)))

        out = torch.cat([shared_graph_level, cell], dim=1)
        out = self.final(out)
        return out