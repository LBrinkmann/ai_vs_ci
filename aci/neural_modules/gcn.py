from torch_geometric.nn import GCNConv, GraphConv
import torch.nn as nn
import torch.nn.functional as F
import torch as th


def get_graph_layer(in_channels, out_channels, class_name, **kwargs):
    if class_name == 'GraphConv':
        layer_class = GraphConv
    elif class_name == 'GCNConv':
        layer_class = GCNConv
    return layer_class(in_channels, out_channels, **kwargs)


class GCNModel(nn.Module):
    def __init__(self, n_actions, hidden_size, layer_args, device):
        super(GCNModel, self).__init__()

        self.conv1 = get_graph_layer(n_actions, hidden_size, **layer_args)
        self.conv2 = get_graph_layer(hidden_size, n_actions, **layer_args)

    def reset(self):
        pass

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
