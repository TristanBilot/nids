import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential
from torch.nn import functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    GINEConv,
    MessagePassing,
    NNConv,
    TransformerConv,
    SAGEConv,
)

class LaresEncoder(MessagePassing):
    def __init__(
        self,
        x_dim,
        h_dim,
        z_dim,
        edge_dim,
        node_dropout=0.0,
        flow="target_to_source",
        without_tanh=True,
        with_epsilon=False,
    ):
        super().__init__(aggr="add")

        self.convs = nn.Sequential(
            GATConv(x_dim, h_dim, heads=3, concat=False, flow=flow),
            GATConv(h_dim, z_dim, heads=3, concat=False, flow=flow),
        )
        self.mp_linear1 = nn.Linear(z_dim, z_dim)
        self.proj2 = nn.Linear(z_dim, z_dim)

        self.drop = nn.Dropout(node_dropout)
        self.tanh = nn.Tanh()

        self.eps = torch.nn.Parameter(torch.empty(1))
        self.without_tanh = without_tanh
        self.with_epsilon = with_epsilon
        
        self.proj = nn.Linear(x_dim, z_dim)
        self.edge_proj = nn.Linear(edge_dim, z_dim)

    def forward(self, x, ei, e_attr):
        x = self.proj(x)
        x = self.drop(x)

        if self.with_epsilon:
            x = x * (1 + self.eps)
        
        x = self.proj2(
            x + self.propagate(ei, x=x, e_attr=e_attr)
        )

        if not self.without_tanh:
            x = self.tanh(x)

        return x

    def message(self, x_j, e_attr):
        e_weights = e_attr[:, 0]
        e_weights = e_weights.view(-1, 1)
        return self.tanh(self.mp_linear1(x_j * e_weights + self.edge_proj(e_attr)))


class GCN(nn.Module):
    def __init__(
        self,
        x_dim,
        h_dim,
        z_dim,
        nb_layers=1,
        node_dropout=0.0,
        flow="target_to_source",
    ):
        super(GCN, self).__init__()

        assert nb_layers in [1, 2], "Only implemented for 1 and 2 layers."

        if nb_layers == 1:
            # Concat=False seems to work best
            self.convs = nn.Sequential(GCNConv(x_dim, z_dim, flow=flow))
        else:
            self.convs = nn.Sequential(
                GCNConv(x_dim, h_dim, flow=flow),
                GCNConv(h_dim, z_dim, flow=flow),
            )

        self.drop = nn.Dropout(node_dropout)
        self.tanh = nn.Tanh()

        self.first_layer = True

    def forward(self, x, ei):
        ei, ew = self.de(ei, ew=None)

        if len(self.convs) == 1:
            x = self.convs[0](x, ei)
        else:
            x = F.relu(self.convs[0](x, ei))
            x = self.drop(x)  # vérifier si c'est worth it
            x = self.convs[1](x, ei)

        # Experiments have shown this is the best activation for GCN+GRU
        return self.tanh(x)


class GAT(GCN):
    def __init__(
        self,
        x_dim,
        h_dim,
        z_dim,
        nb_layers=1,
        heads=3,
        flow="target_to_source",
        **kwargs,
    ):
        super().__init__(x_dim, h_dim, z_dim, **kwargs)

        assert nb_layers in [1, 2], "Only implemented for 1 and 2 layers."

        if nb_layers == 1:
            self.convs = nn.Sequential(
                GATConv(x_dim, z_dim, heads=heads, concat=False, flow=flow)
            )
        else:
            self.convs = nn.Sequential(
                GATConv(x_dim, h_dim, heads=heads, concat=False, flow=flow),
                GATConv(h_dim, z_dim, heads=heads, concat=False, flow=flow),
            )

        self.first_layer = True

    def forward(self, x, ei):
        ei, ew = self.de(ei, ew=None)

        if len(self.convs) == 1:
            x = self.convs[0](x, ei)
        else:
            x = F.relu(self.convs[0](x, ei))
            x = self.drop(x)  # vérifier si c'est worth it
            x = self.convs[1](x, ei)

        # Experiments have shown this is the best activation for GCN+GRU
        return self.tanh(x)


class GINENet(torch.nn.Module):
    def __init__(
        self,
        x_dim,
        h_dim,
        z_dim,
        edge_feat_size,
        node_dropout,
        nb_layers=1,
        flow="target_to_source",
    ):
        super(GINENet, self).__init__()

        h_dim = h_dim * 2

        nn1 = Sequential(Linear(x_dim, h_dim), ReLU(), Linear(h_dim, h_dim))
        self.conv1 = GINEConv(nn1, edge_dim=edge_feat_size)

        nn2 = Sequential(Linear(h_dim, h_dim), ReLU(), Linear(h_dim, h_dim))
        self.conv2 = GINEConv(nn2, edge_dim=edge_feat_size)

        self.fc1 = Linear(h_dim, h_dim)
        self.fc2 = Linear(h_dim, z_dim)

        self.drop = nn.Dropout(node_dropout)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.tanh(x)
        x = self.drop(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = torch.tanh(x)

        # NOTE: It's worst in inductive setting to use 2 dropouts
        # x = self.drop(x)

        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class GIN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, node_dropout=0.0):
        super().__init__()
        
        mlp1 = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
        )
        self.conv1 = GINConv(mlp1)
        
        mlp2 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
        )
        self.conv2 = GINConv(mlp2)

        self.drop = nn.Dropout(node_dropout)
        self.tanh = nn.Tanh()

    def forward(self, x, ei, edge_weight):
        x = F.relu(self.conv1(x, ei))
        x = self.drop(x)
        x = self.conv2(x, ei)
        return self.tanh(x)

class EULER_GCN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, node_dropout=0.0, flow="target_to_source"):
        super().__init__()

        self.conv1 = GCNConv(x_dim, h_dim, flow=flow)
        self.conv2 = GCNConv(h_dim, z_dim, flow=flow)

        self.drop = nn.Dropout(node_dropout)
        self.tanh = nn.Tanh()

    def forward(self, x, ei, edge_weight):
        edge_weight = edge_weight[:, 0]  # standardized edge weights
        x = F.relu(self.conv1(x, ei, edge_weight=edge_weight))
        x = self.drop(x)
        x = self.conv2(x, ei, edge_weight=edge_weight)

        return self.tanh(x)


class GraphSAGE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, node_dropout=0.0, flow="target_to_source"):
        super().__init__()

        self.conv1 = SAGEConv(x_dim, h_dim, flow=flow)
        self.conv2 = SAGEConv(h_dim, z_dim, flow=flow)

        self.drop = nn.Dropout(node_dropout)
        self.tanh = nn.Tanh()

    def forward(self, x, ei, *args, **kwargs):
        x = F.relu(self.conv1(x, ei))
        x = self.drop(x)
        x = self.conv2(x, ei)

        return self.tanh(x)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim=None):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim if out_dim is None else out_dim),
            nn.Sigmoid(),  # Sigmoid activation for output in [0, 1] range
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)
        self.linear3 = nn.Linear(hidden_dim, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
        return z


class VariationalDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalDecoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z


# Inspired by: https://avandekleut.github.io/vae/
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VariationalDecoder(input_dim, hidden_dim, latent_dim)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def loss(self, h_hat, h):
        return self.criterion(h_hat, h) + self.encoder.kl

class Argus_LANL(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, e_feat_size, device):
        super().__init__()

        self.c1 = GCNConv(x_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, h_dim, add_self_loops=True)
        self.drop = nn.Dropout(0.1)
        self.ac = nn.Tanh()
        self.c3 = GCNConv(h_dim, h_dim, add_self_loops=True)
        nn4 = nn.Sequential(nn.Linear(e_feat_size, 8), nn.ReLU(),
                            nn.Linear(8, h_dim * z_dim))
        self.c4 = NNConv(h_dim, z_dim, nn4, aggr='mean')


    def forward(self, x, ei, e_attr):
        ew = e_attr[:, 0]
        x1 = self.c1(x, ei, edge_weight=ew)
        x = self.c2(x1, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c3(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c4(x, ei, edge_attr=e_attr)
        return self.ac(x)

class Argus_OPTC(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, e_feat_size, device):
        super().__init__()
        self.de = DropEdge(0.5)
        self.c1 = GCNConv(x_dim, h_dim, add_self_loops=True)
        self.c2 = GCNConv(h_dim, h_dim, add_self_loops=True)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.c3 = GCNConv(h_dim, h_dim, add_self_loops=True)
        self.c4 = GCNConv(h_dim, z_dim, add_self_loops=True)
        self.ac = nn.Tanh()

    def forward(self, x, ei, e_attr):
        ew = e_attr[:, 0]
        ei, ew = self.de(ei, ew=ew) # increase 2%
        x = self.c1(x, ei, edge_weight=ew)
        x = self.c2(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c3(x, ei, edge_weight=ew)
        x = self.relu(x)
        x = self.drop(x)
        x = self.c4(x, ei, edge_weight=ew)
        return self.ac(x)

class GraphAttentionEmbedding(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, edge_dim, node_dropout):
        super(GraphAttentionEmbedding, self).__init__()
        
        self.conv = TransformerConv(x_dim, h_dim, heads=8, dropout=node_dropout, edge_dim=edge_dim)
        self.conv2 = TransformerConv(h_dim * 8, z_dim, heads=1, concat=False, dropout=node_dropout, edge_dim=edge_dim)
        self.dropout = nn.Dropout(node_dropout)

    def forward(self, x, ei, e_attr):
        x = torch.relu(self.conv(x, ei, e_attr))
        x = self.dropout(x)
        x = self.conv2(x, ei, e_attr)
        return torch.tanh(x)

class DropEdge(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, ei, ew=None):
        if self.training and self.p > 0:
            mask = torch.rand(ei.size(1))
            if ew is None:
                return ei[:, mask > self.p], None
            else:
                return ei[:, mask > self.p], ew[mask > self.p]

        if ew is None:
            return ei, None
        else:
            return ei, ew
