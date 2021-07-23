from torch_geometric.nn import GCNConv, GraphConv
import torch.nn as nn


def get_graph_layer(in_channels, out_channels, class_name, **kwargs):
    if class_name == 'GraphConv':
        layer_class = GraphConv
    elif class_name == 'GCNConv':
        layer_class = GCNConv
    return layer_class(in_channels, out_channels, **kwargs)


class GCNModel(nn.Module):
    def __init__(self, view_size, n_actions, hidden_size, graph_layer_name, device, **kwargs):
        super(GCNModel, self).__init__()

        input_size = view_size['view'][-1]

        self.conv1 = get_graph_layer(input_size, hidden_size, class_name=graph_layer_name)
        self.conv2 = get_graph_layer(hidden_size, n_actions, class_name=graph_layer_name)

    def reset(self):
        pass

    def forward(self, view, edge_index):
        x = self.conv1(view, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
