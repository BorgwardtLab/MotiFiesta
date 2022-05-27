import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn.pool import EdgePooling


def get_gnn_layer(gnn_type, embed_dim=64):
    if gnn_type == 'gcn':
        return gnn.GCNConv(embed_dim, embed_dim)
    elif gnn_type =='gin':
        mlp = mlp = nn.Sequential( nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )
        return gnn.GINConv(mlp, train_eps=True)
    else:
        raise ValueError("Not implemented!")

class EdgePool(nn.Module):
    def __init__(self, in_dim=1, num_class=1, embed_dim=64, gnn_type='gcn', num_layers=2, dropout=0.0, global_pool='sum'):
        super().__init__()
        self.num_class = num_class
        self.gnn_type = gnn_type
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.global_pool = global_pool

        if global_pool == 'sum':
            self.global_pool = gnn.global_add_pool
        elif global_pool == 'mean':
            self.global_pool = gnn.global_mean_pool
        elif global_pool == 'max':
            self.global_pool = gnn.global_max_pool

        self.in_dim = in_dim
        if in_dim == 1:
            self.node_embed = nn.Embedding(in_dim, embed_dim)
        else:
            self.node_embed = nn.Linear(in_dim, embed_dim, bias=False)

        self.conv_layers = []
        for i in range(num_layers):
            self.conv_layers.append(EdgePooling(embed_dim))

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.fc = nn.Linear(embed_dim*(self.num_layers+1), num_class)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        z = self.node_embed(x)
        emb_stack = [self.global_pool(z, batch)]
        for layer in self.conv_layers:
            z, edge_index, batch,_ = layer(z, edge_index, batch)
            emb_stack.append(self.global_pool(z, batch))
        Z = torch.cat(emb_stack, dim=1)
        return self.fc(Z)

class GNN(nn.Module):
    def __init__(self, in_dim=1, num_class=1, embed_dim=64, gnn_type='gcn', num_layers=2, dropout=0.0, global_pool='sum'):
        super().__init__()
        self.num_class = num_class
        self.gnn_type = gnn_type
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.global_pool = global_pool

        self.in_dim = in_dim
        if in_dim == 1:
            self.node_embed = nn.Embedding(in_dim, embed_dim)
        else:
            self.node_embed = nn.Linear(in_dim, embed_dim, bias=False)

        self.conv_layers = []
        self.bn_layers = []

        for i in range(num_layers):
            conv_layer = get_gnn_layer(gnn_type, embed_dim)
            self.conv_layers.append(conv_layer)
            self.bn_layers.append(nn.BatchNorm1d(self.embed_dim))


        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.bn_layers = nn.ModuleList(self.bn_layers)

        self.dropout = dropout

        if global_pool == 'sum':
            self.global_pool = gnn.global_add_pool
        elif global_pool == 'mean':
            self.global_pool = gnn.global_mean_pool
        elif global_pool == 'max':
            self.global_pool = gnn.global_max_pool

        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h = self.node_embed(x)

        for i in range(self.num_layers):
            h = self.conv_layers[i](h, edge_index)
            h = self.bn_layers[i](h)

            if i < self.num_layers - 1:
                h = F.relu(h)

            h = F.dropout(h, self.dropout, training=self.training)

        h_pool = self.global_pool(h, data.batch)
        h_pool = self.fc(h_pool)

        return h_pool

if __name__ == "__main__":
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    data = TUDataset('.', 'PROTEINS')
    loader = DataLoader(data, batch_size=4)
    edgepool = EdgePool(in_dim=data.num_features, num_class=data[0].y.shape[-1])
    for batch in loader:
        y = edgepool(batch)
