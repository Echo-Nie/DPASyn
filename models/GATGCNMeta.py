import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GCNConv
import numpy as np


class AttenSyn(nn.Module):
    def __init__(
            self,
            molecule_channels: int = 78,
            hidden_channels: int = 128,
            middle_channels: int = 64,
            layer_count: int = 2,
            out_channels: int = 2,
            dropout_rate: int = 0.2,
            heads: int = 4
    ):
        super().__init__()
        # 分别创建GAT和GCN的ModuleList
        self.gat_convolutions = torch.nn.ModuleList()
        self.gcn_convolutions = torch.nn.ModuleList()

        # 第一层：GAT和GCN并行
        self.gat_convolutions.append(GATConv(molecule_channels, hidden_channels // 2 // heads, heads=heads))
        self.gcn_convolutions.append(GCNConv(molecule_channels, hidden_channels // 2))

        # 后续层：GAT和GCN并行
        for _ in range(1, layer_count):
            self.gat_convolutions.append(GATConv(hidden_channels, hidden_channels // 2 // heads, heads=heads))
            self.gcn_convolutions.append(GCNConv(hidden_channels, hidden_channels // 2))

        # LSTM层
        self.border_rnn = torch.nn.LSTM(hidden_channels, hidden_channels, 1)

        # 其他层保持不变
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

        self.pool1 = Attention(hidden_channels, 4)
        self.pool2 = Attention(hidden_channels, 4)

        self.final = torch.nn.Sequential(
            torch.nn.Linear(4 * hidden_channels + 256, middle_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(middle_channels, out_channels)
        )

    # GAT
    def _forward_molecules(
            self,
            gat_conv,
            gcn_conv,
            x,
            edge_index,
            batch,
            states,
    ) -> torch.FloatTensor:
        # 分别通过GAT和GCN
        gat_hidden = gat_conv(x, edge_index)
        gcn_hidden = gcn_conv(x, edge_index)

        # 合并GAT和GCN的输出
        gcn_hidden = torch.cat([gat_hidden, gcn_hidden], dim=-1)

        gcn_hidden_detach = gcn_hidden.detach()
        rnn_out, (hidden_state, cell_state) = self.border_rnn(gcn_hidden_detach[None, :, :], states)
        rnn_out = rnn_out.squeeze()
        return gcn_hidden, rnn_out, (hidden_state, cell_state)

    def forward(self, molecules_left, molecules_right) -> torch.FloatTensor:
        x1, edge_index1, batch1, cell, mask1 = molecules_left.x, molecules_left.edge_index, molecules_left.batch, molecules_left.cell, molecules_left.mask
        x2, edge_index2, batch2, mask2 = molecules_right.x, molecules_right.edge_index, molecules_right.batch, molecules_right.mask
        cell = F.normalize(cell, 2, 1)
        cell_expand = self.reduction2(cell)
        cell = self.reduction(cell)
        cell_expand = cell_expand.unsqueeze(1)

        cell_expand = cell_expand.expand(cell.shape[0], 100, -1)
        cell_expand = cell_expand.reshape(-1, 78)
        batch_size = torch.max(molecules_left.batch) + 1

        mask1 = mask1.reshape((batch_size, 100))
        mask2 = mask2.reshape((batch_size, 100))
        left_states, right_states = None, None
        gcn_hidden_left = molecules_left.x + cell_expand
        gcn_hidden_right = molecules_right.x + cell_expand
        for gat_conv, gcn_conv in zip(self.gat_convolutions, self.gcn_convolutions):
            gcn_hidden_left, rnn_out_left, left_states = self._forward_molecules(
                gat_conv, gcn_conv, gcn_hidden_left, molecules_left.edge_index,
                molecules_left.batch, left_states
            )
            gcn_hidden_right, rnn_out_right, right_states = self._forward_molecules(
                gat_conv, gcn_conv, gcn_hidden_right, molecules_right.edge_index,
                molecules_right.batch, right_states
            )

        rnn_out_left, rnn_out_right = rnn_out_left.reshape(batch_size, 100, -1), rnn_out_right.reshape(batch_size, 100,
                                                                                                       -1)
        rnn_pooled_left, rnn_pooled_right = self.pool1(rnn_out_left, rnn_out_right, (mask1, mask2))
        gcn_hidden_left, gcn_hidden_right = gcn_hidden_left.reshape(batch_size, 100, -1), gcn_hidden_right.reshape(
            batch_size, 100, -1)

        gcn_hidden_left, gcn_hidden_right = self.pool2(gcn_hidden_left, gcn_hidden_right, (mask1, mask2))
        shared_graph_level = torch.cat([gcn_hidden_left, gcn_hidden_right], dim=1)
        out = torch.cat([shared_graph_level, rnn_pooled_left, rnn_pooled_right, cell], dim=1)
        out = self.final(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = dim // num_heads
        self.linear_q = nn.Linear(dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(dim, self.dim_per_head * num_heads)
        self.norm = nn.LayerNorm(dim)
        self.linear_final = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=0.2)

    def attention(self, q1, k1, v1, q2, k2, v2, attn_mask=None, flag=False):
        a1 = torch.tanh(torch.bmm(k1, q2.transpose(1, 2)))
        a2 = torch.tanh(torch.bmm(k2, q1.transpose(1, 2)))
        if attn_mask is not None:
            mask1 = attn_mask[0]
            mask2 = attn_mask[1]
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

    def forward(self, fingerprint_vectors1, fingerprint_vectors2, attn_mask=None, flag=False):
        q1, q2 = torch.relu(self.linear_q(fingerprint_vectors1)), torch.relu(self.linear_q(fingerprint_vectors2))
        k1, k2 = torch.relu(self.linear_k(fingerprint_vectors1)), torch.relu(self.linear_k(fingerprint_vectors2))
        v1, v2 = torch.relu(self.linear_v(fingerprint_vectors1)), torch.relu(self.linear_v(fingerprint_vectors2))

        vector1, vector2 = self.attention(q1, k1, v1, q2, k2, v2, attn_mask, flag)

        vector1 = self.norm(torch.mean(fingerprint_vectors1, dim=1) + vector1)
        vector2 = self.norm(torch.mean(fingerprint_vectors2, dim=1) + vector2)

        return vector1, vector2


class MAML:
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)

    def inner_update(self, loss):
        """在内部任务上更新模型参数"""
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        for param, grad in zip(self.model.parameters(), grads):
            param.data -= self.inner_lr * grad

    def meta_update(self, loss):
        """在元任务上更新模型参数"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class AttenSynWithMAML(AttenSyn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maml = MAML(self)

    def forward(self, molecules_left, molecules_right):
        return super().forward(molecules_left, molecules_right)

    def train_step(self, task_batch):
        """在单个任务上进行训练"""
        support_set, query_set = task_batch

        # 在支持集上进行内部更新
        for molecules_left, molecules_right, labels in support_set:
            output = self.forward(molecules_left, molecules_right)
            loss = F.cross_entropy(output, labels)
            self.maml.inner_update(loss)

        # 在查询集上计算元损失
        meta_loss = 0
        for molecules_left, molecules_right, labels in query_set:
            output = self.forward(molecules_left, molecules_right)
            meta_loss += F.cross_entropy(output, labels)

        # 在元任务上进行更新
        self.maml.meta_update(meta_loss)


def train(model, tasks, epochs=10):
    for epoch in range(epochs):
        for task_batch in tasks:
            model.train_step(task_batch)


def create_tasks(dataset, num_tasks, support_size=5, query_size=5):
    tasks = []
    for _ in range(num_tasks):
        indices = torch.randperm(len(dataset))
        support_set = [(dataset[i].x, dataset[i].edge_index, dataset[i].y) for i in indices[:support_size]]
        query_set = [(dataset[i].x, dataset[i].edge_index, dataset[i].y) for i in indices[support_size:support_size + query_size]]
        tasks.append((support_set, query_set))
    return tasks