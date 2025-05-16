import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, GCNConv
import numpy as np

class DualAttention(nn.Module):
    def __init__(self, feature_dim, num_attention_heads=4):
        super(DualAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = feature_dim // num_attention_heads

        # Query, Key, Value projections for both directions
        self.query_proj = nn.Linear(feature_dim, self.head_dim * num_attention_heads)
        self.key_proj = nn.Linear(feature_dim, self.head_dim * num_attention_heads)
        self.value_proj = nn.Linear(feature_dim, self.head_dim * num_attention_heads)

        self.layer_norm = nn.LayerNorm(feature_dim)
        self.attention_dropout = nn.Dropout(p=0.2)

    def compute_attention(self, query_1, key_1, value_1, query_2, key_2, value_2, attention_mask=None):
        # Compute attention scores
        attention_scores_1 = torch.tanh(torch.bmm(key_1, query_2.transpose(1, 2)))
        attention_scores_2 = torch.tanh(torch.bmm(key_2, query_1.transpose(1, 2)))

        if attention_mask is not None:
            mask_1, mask_2 = attention_mask
            attention_weights_1 = torch.softmax(
                torch.sum(attention_scores_1, dim=2).masked_fill(mask_1, -np.inf), 
                dim=-1
            ).unsqueeze(dim=1)
            attention_weights_2 = torch.softmax(
                torch.sum(attention_scores_2, dim=2).masked_fill(mask_2, -np.inf), 
                dim=-1
            ).unsqueeze(dim=1)
        else:
            attention_weights_1 = torch.softmax(torch.sum(attention_scores_1, dim=2), dim=1).unsqueeze(dim=1)
            attention_weights_2 = torch.softmax(torch.sum(attention_scores_2, dim=2), dim=1).unsqueeze(dim=1)

        attention_weights_1 = self.attention_dropout(attention_weights_1)
        attention_weights_2 = self.attention_dropout(attention_weights_2)

        context_vector_1 = torch.bmm(attention_weights_1, value_1).squeeze()
        context_vector_2 = torch.bmm(attention_weights_2, value_2).squeeze()

        return context_vector_1, context_vector_2

    def forward(self, features_1, features_2, attention_mask=None):
        # Project features to query, key, value
        query_1 = torch.relu(self.query_proj(features_1))
        key_1 = torch.relu(self.key_proj(features_1))
        value_1 = torch.relu(self.value_proj(features_1))

        query_2 = torch.relu(self.query_proj(features_2))
        key_2 = torch.relu(self.key_proj(features_2))
        value_2 = torch.relu(self.value_proj(features_2))

        context_vector_1, context_vector_2 = self.compute_attention(
            query_1, key_1, value_1, query_2, key_2, value_2, attention_mask
        )

        # Residual connection and normalization
        context_vector_1 = self.layer_norm(torch.mean(features_1, dim=1) + context_vector_1)
        context_vector_2 = self.layer_norm(torch.mean(features_2, dim=1) + context_vector_2)

        return context_vector_1, context_vector_2

class MolecularInteractionModel(nn.Module):
    def __init__(
        self,
        input_feature_dim: int = 78,
        hidden_feature_dim: int = 128,
        intermediate_dim: int = 64,
        num_graph_layers: int = 2,
        output_dim: int = 2,
        dropout_prob: float = 0.2,
        num_attention_heads: int = 4,
    ):
        super().__init__()

        # Graph attention layers
        self.graph_attention_layers = torch.nn.ModuleList()
        self.graph_attention_layers.append(
            GATConv(input_feature_dim, hidden_feature_dim // num_attention_heads, heads=num_attention_heads)
        )
        for _ in range(1, num_graph_layers):
            self.graph_attention_layers.append(
                GATConv(hidden_feature_dim, hidden_feature_dim // num_attention_heads, heads=num_attention_heads)
            )

        # Sequence modeling
        self.sequence_model = torch.nn.LSTM(hidden_feature_dim, hidden_feature_dim, 1)

        # Feature transformation layers
        self.feature_transformer = nn.Sequential(
            nn.Linear(954, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.feature_transformer_2 = nn.Sequential(
            nn.Linear(954, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 78),
            nn.ReLU(),
        )

        # Attention pooling layers
        self.sequence_attention = DualAttention(hidden_feature_dim, num_attention_heads)
        self.graph_attention = DualAttention(hidden_feature_dim, num_attention_heads)

        # Final prediction layers
        self.prediction_head = nn.Sequential(
            nn.Linear(4 * hidden_feature_dim + 256, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, output_dim),
        )

    def process_molecular_features(self, conv_layer, features, edge_index, batch, hidden_states):
        graph_features = conv_layer(features, edge_index)
        sequence_output, (hidden_state, cell_state) = self.sequence_model(graph_features.unsqueeze(0), hidden_states)
        sequence_output = sequence_output.squeeze(0)
        return graph_features, sequence_output, (hidden_state, cell_state)

    def forward(self, left_molecule, right_molecule):
        # Extract input features
        left_features = left_molecule.x
        left_edge_index = left_molecule.edge_index
        left_batch = left_molecule.batch
        cell_features = left_molecule.cell
        left_mask = left_molecule.mask

        right_features = right_molecule.x
        right_edge_index = right_molecule.edge_index
        right_batch = right_molecule.batch
        right_mask = right_molecule.mask

        # Process cell features
        cell_features = F.normalize(cell_features, 2, 1)
        expanded_cell = self.feature_transformer_2(cell_features).unsqueeze(1)
        expanded_cell = expanded_cell.expand(cell_features.shape[0], 100, -1).reshape(-1, 78)
        transformed_cell = self.feature_transformer(cell_features)

        # Prepare masks
        batch_size = torch.max(left_molecule.batch) + 1
        left_mask = left_mask.reshape(batch_size, 100)
        right_mask = right_mask.reshape(batch_size, 100)

        # Initialize states
        left_states = right_states = None
        left_graph_features = left_features + expanded_cell
        right_graph_features = right_features + expanded_cell

        # Process through graph layers
        for conv_layer in self.graph_attention_layers:
            left_graph_features, left_sequence, left_states = self.process_molecular_features(
                conv_layer, left_graph_features, left_edge_index, left_batch, left_states
            )
            right_graph_features, right_sequence, right_states = self.process_molecular_features(
                conv_layer, right_graph_features, right_edge_index, right_batch, right_states
            )

        # Reshape sequence outputs
        left_sequence = left_sequence.reshape(batch_size, 100, -1)
        right_sequence = right_sequence.reshape(batch_size, 100, -1)

        # Apply attention pooling
        pooled_sequence_left, pooled_sequence_right = self.sequence_attention(
            left_sequence, right_sequence, (left_mask, right_mask)
        )

        # Process graph features
        left_graph_features = left_graph_features.reshape(batch_size, 100, -1)
        right_graph_features = right_graph_features.reshape(batch_size, 100, -1)
        pooled_graph_left, pooled_graph_right = self.graph_attention(
            left_graph_features, right_graph_features, (left_mask, right_mask)
        )

        # Combine features and make prediction
        graph_level_features = torch.cat([pooled_graph_left, pooled_graph_right], dim=1)
        combined_features = torch.cat([graph_level_features, pooled_sequence_left, pooled_sequence_right, transformed_cell], dim=1)
        prediction = self.prediction_head(combined_features)

        return prediction
