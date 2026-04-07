import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import SAGEConv, to_hetero

class Encoder(nn.Module):
    def __init__(self, hidden_channels, dropout=0.2):
        # hidden_channels is the size of the learned node embeddings after each SAGE layer
        super().__init__()
        # Each message-passing layer expands the distance from the starting node
        # 1 layer = immediate neighbours, 2 layers = neighbours-of-neighbours

        # SAGE works well for heterogenous graphs
        # (-1, -1) says to infer the input features automatically from nodes
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.dropout = nn.Dropout(dropout)

    # Updates embeddings by using features from combing a nodes own feature and neighbouring ndoes
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        
        # Returns updated node embeddings
        return x
    
class GraphAE(nn.Module):
    def __init__(self, metadata, hidden_channels, txn_input_dim, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(hidden_channels, dropout=dropout)
        self.encoder = to_hetero(self.encoder, metadata=metadata) # Converts encoder so it can work on heterogenous graphs

        # Tries to reconstruct the original txn features
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, txn_input_dim)
        )
    
    def forward(self, x_dict, edge_index_dict):
        # z_dict contains learned embeddings for each node type
        z_dict = self.encoder(x_dict, edge_index_dict)
        
        # Only looks at txn nodes for detecting anomalies
        recon_txn = self.decoder(z_dict["transaction"])
        return recon_txn, z_dict
