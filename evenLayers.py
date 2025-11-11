import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import get_laplacian
import numpy as np
from torch_geometric.nn.conv.gcn_conv import gcn_norm
# from torch_geometric.utils import to_dense_adj

class EvenNetLayer(MessagePassing):
    def __init__(self, K, alpha):
        super(EvenNetLayer, self).__init__()
        self.max_order = K if K % 2 == 0 else K - 1  # 若 K 是奇数，取 K-1
        self.K = (self.max_order // 2) + 1  # 计算循环次数，例如 K=10 → 6次循环（0,2,...,10）

        self.alpha = alpha
        self.weight = Parameter(torch.tensor([
            alpha * (1 - alpha) ** (2 * k) for k in range(self.K)
        ]))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.weight)
        for k in range(self.K):
            self.weight.data[k] = self.alpha * (1 - self.alpha) ** (2 * k)

    def forward(self, x, edge_index, edge_weight=None):

        edge_index2, norm2 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                           num_nodes=x.size(self.node_dim))

        edge_index3, norm3 = add_self_loops(edge_index2, -norm2, fill_value=1.0, num_nodes=x.size(self.node_dim))

        output = x * self.weight[0]
        # output = self.weight[0] * x
        current_x = x.clone()
        for k in range(1, self.K):
            # 执行 (2) 次传播
            for _ in range(2):
                current_x = self.propagate(edge_index3, x=current_x, norm=norm3)
            output += self.weight[k]* current_x

        return output

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, weight={})'.format(self.__class__.__name__, self.max_order, self.weight)
