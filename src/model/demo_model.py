import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GCNConv
import numpy as np

class SimpleDualAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=2):
        super(SimpleDualAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, features_1, features_2, mask=None):
        query_1 = self.query_proj(features_1)
        key_1 = self.key_proj(features_1)
        value_1 = self.value_proj(features_1)

        query_2 = self.query_proj(features_2)
        key_2 = self.key_proj(features_2)
        value_2 = self.value_proj(features_2)

        scores_1 = torch.matmul(query_1, key_2.transpose(-2, -1)) / np.sqrt(self.head_dim)
        scores_2 = torch.matmul(query_2, key_1.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mask is not None:
            mask_1, mask_2 = mask
            scores_1 = scores_1.masked_fill(mask_1.unsqueeze(-1), -1e9)
            scores_2 = scores_2.masked_fill(mask_2.unsqueeze(-1), -1e9)

        attn_weights_1 = F.softmax(scores_1, dim=-1)
        attn_weights_2 = F.softmax(scores_2, dim=-1)

        attn_weights_1 = self.dropout(attn_weights_1)
        attn_weights_2 = self.dropout(attn_weights_2)

        output_1 = torch.matmul(attn_weights_1, value_2)
        output_2 = torch.matmul(attn_weights_2, value_1)

        output_1 = self.layer_norm(features_1 + output_1)
        output_2 = self.layer_norm(features_2 + output_2)

        return output_1, output_2

class DemoDualStreamModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 2,
        dropout: float = 0.1,
        num_heads: int = 2,
    ):
        super().__init__()

        self.graph_layer_1 = GATConv(input_dim, hidden_dim // num_heads, heads=num_heads)
        self.graph_layer_2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)

        self.feature_transform = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.dual_attention = SimpleDualAttention(hidden_dim, num_heads)

        self.prediction_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, mol_1, mol_2):
        x1, edge_index1, batch1 = mol_1.x, mol_1.edge_index, mol_1.batch
        x2, edge_index2, batch2 = mol_2.x, mol_2.edge_index, mol_2.batch

        h1 = F.relu(self.graph_layer_1(x1, edge_index1))
        h1 = F.relu(self.graph_layer_2(h1, edge_index1))
        
        h2 = F.relu(self.graph_layer_1(x2, edge_index2))
        h2 = F.relu(self.graph_layer_2(h2, edge_index2))

        h1_global = torch.mean(h1, dim=0, keepdim=True)
        h2_global = torch.mean(h2, dim=0, keepdim=True)

        h1_attended, h2_attended = self.dual_attention(h1_global, h2_global)

        combined_features = torch.cat([h1_attended.squeeze(), h2_attended.squeeze()], dim=-1)
        
        output = self.prediction_head(combined_features)
        
        return output
