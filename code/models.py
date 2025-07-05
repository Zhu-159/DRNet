import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import SAGEConv, global_sort_pool, MessagePassing, GATConv, GCNConv, GINConv
from torch_geometric.utils import dropout_adj
import kafnets as kaf


# Semantic Attention Module
class SemanticAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super(SemanticAttention, self).__init__()
        reduction_dim = max(1, channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduction_dim),
            nn.ReLU(),
            nn.Linear(reduction_dim, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.fc(x)
        return x * attention_weights


# Structural Attention Module
class StructuralAttention(nn.Module):
    def __init__(self, input_dim):
        super(StructuralAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.tensor(input_dim, dtype=torch.float32))

    def forward(self, x, batch):
        # Process nodes from different graphs separately
        max_num_nodes = torch.bincount(batch).max().item()
        batch_size = batch.max().item() + 1

        # Initialize output tensor
        output = torch.zeros_like(x)

        # Apply self-attention to each graph separately
        for i in range(batch_size):
            mask = batch == i
            if not mask.any():
                continue

            nodes = x[mask]
            num_nodes = nodes.size(0)

            if num_nodes <= 1:
                output[mask] = nodes
                continue

            q = self.query(nodes)
            k = self.key(nodes)
            v = self.value(nodes)

            # Calculate attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Apply attention weights
            attended = torch.matmul(attn_weights, v)
            output[mask] = attended

        return output

    def reset_parameters(self):
        self.query.reset_parameters()
        self.key.reset_parameters()
        self.value.reset_parameters()

# Dynamic Gated Graph Attention
class DynamicGatedGraphAttention(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(DynamicGatedGraphAttention, self).__init__(aggr='add')
        self.feature = GATConv(in_channels, out_channels, heads=4, concat=False)
        self.gate = GATConv(in_channels, out_channels, heads=4, concat=False)
        self.lin = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.kaf = kaf.KAF(out_channels, D=20, kernel='gaussian')

    def forward(self, x, edge_index):
        feature_map = self.kaf(self.feature(x, edge_index))
        gate = torch.sigmoid(self.gate(x, edge_index))
        x = self.lin(x)
        out = feature_map * gate + x
        return self.bn(out)

    def reset_parameters(self):
        self.feature.reset_parameters()
        self.gate.reset_parameters()
        self.lin.reset_parameters()
        self.bn.reset_parameters()


class DRNet(torch.nn.Module):
    def __init__(self, dataset, gconv=DynamicGatedGraphAttention , latent_dim=[256, 128, 64], k=30,
                 dropout_n=0.4, dropout_e=0.1, force_undirected=False):
        super(DRNet, self).__init__()

        self.dropout_n = dropout_n
        self.dropout_e = dropout_e
        self.force_undirected = force_undirected

        self.conv1 = gconv(dataset.num_features, latent_dim[0])
        self.conv2 = gconv(latent_dim[0], latent_dim[1])
        self.conv3 = gconv(latent_dim[1], latent_dim[2])

        self.kaf1 = kaf.KAF(latent_dim[0], D=20, kernel='gaussian')
        self.kaf2 = kaf.KAF(latent_dim[1], D=20, kernel='gaussian')
        self.kaf3 = kaf.KAF(latent_dim[2], D=20, kernel='gaussian')

        # Semantic attention modules
        self.semantic_attn1 = SemanticAttention(latent_dim[0])
        self.semantic_attn2 = SemanticAttention(latent_dim[1])
        self.semantic_attn3 = SemanticAttention(latent_dim[2])

        # Structural attention modules
        self.structural_attn1 = StructuralAttention(latent_dim[0])
        self.structural_attn2 = StructuralAttention(latent_dim[1])
        self.structural_attn3 = StructuralAttention(latent_dim[2])

        if k < 1:
            node_nums = sorted([g.num_nodes for g in dataset])
            k = node_nums[int(math.ceil(k * len(node_nums))) - 1]
            k = max(10, k)

        self.k = int(k)

        # 1D convolution layers
        conv1d_channels = [16, 32]
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws = [self.total_latent_dim, 5]
        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        # Fully connected layers
        self.lin1 = nn.Linear(self.dense_dim, 128)
        self.lin2 = nn.Linear(128, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.structural_attn1.reset_parameters()
        self.structural_attn2.reset_parameters()
        self.structural_attn3.reset_parameters()
        self.conv1d_params1.reset_parameters()
        self.conv1d_params2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        edge_index, _ = dropout_adj(
            edge_index, p=self.dropout_e,
            force_undirected=self.force_undirected, num_nodes=len(x),
            training=self.training
        )

        x1 = self.kaf1(self.conv1(x, edge_index))
        x1 = self.semantic_attn1(x1)
        x1 = self.structural_attn1(x1, batch)

        x2 = self.kaf2(self.conv2(x1, edge_index))
        x2 = self.semantic_attn2(x2)
        x2 = self.structural_attn2(x2, batch)

        x3 = self.kaf3(self.conv3(x2, edge_index))
        x3 = self.semantic_attn3(x3)
        x3 = self.structural_attn3(x3, batch)

        X = [x1, x2, x3]
        concat_states = torch.cat(X, 1)

        x = global_sort_pool(concat_states, batch, self.k)
        x = x.unsqueeze(1)
        
        x = self.conv1d_params1(x)
        x = F.relu(x)
        x = self.maxpool1d(x)

        x = self.conv1d_params2(x)
        x = F.relu(x)

        x = x.view(len(x), -1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_n, training=self.training)
        x = self.lin2(x)

        return x[:, 0]
