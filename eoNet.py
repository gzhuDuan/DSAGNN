import torch
import torch.nn.functional as F

from torch import nn

from evenLayers import EvenNetLayer
from oddLayers import OddNetLayer
from torch.nn import Linear

from torch_geometric.utils import add_self_loops, degree


class EONet(nn.Module):
    def __init__(self, features, classes, Ko, Ke, dropoutClassifier1,dropoutClassifier2, hidden, Gamma, Epsilon):
        super(EONet, self).__init__()

        self.lin1 = Linear(features, hidden)
        self.lin2 = Linear(hidden, classes)


        self.lin5 = Linear(features * 2, 2)


        self.dropoutClassifier1 = dropoutClassifier1
        self.dropoutClassifier2 = dropoutClassifier2

        self.Gamma = Gamma
        self.Epsilon = Epsilon

        self.even_hop = EvenNetLayer(Ke, self.Gamma)
        self.odd_hop = OddNetLayer(Ko, self.Epsilon)


        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.lin5.reset_parameters()

        self.even_hop.reset_parameters()
        self.odd_hop.reset_parameters()

    def forward(self, data):


        # 计算原始度数矩阵D
        row = data.edge_index[0]  # 源节点
        deg = degree(row, data.num_nodes, dtype=torch.float)  # 原始度数向量

        #################################################
        x, edge_index = data.x, data.edge_index

        beta = torch.log(deg + 1)

        beta = beta.view(-1, 1)  # [N, 1]

        ##############################
        # 奇偶跳特征传播
        He = self.even_hop(x, edge_index)  # [N, features]

        Ho = self.odd_hop(x, edge_index)  # [N, features]


        He = He * beta  # [N, features]
        Ho = Ho * beta  # [N, features]

        # 动态权重计算
        alpha_input = torch.cat([He, Ho], dim=-1)  # [N, 2*features]
        alpha_scores = self.lin5(alpha_input)

        alpha_probs = F.sigmoid(alpha_scores)


        alpha1 = alpha_probs[:, 0: 1]
        alpha2 = alpha_probs[:, 1: 2]
        H = alpha1 * He + alpha2 * Ho


        H = F.dropout(H, p=self.dropoutClassifier1, training=self.training)
        H = self.lin1(H)
        H = F.relu(H)
        H = F.dropout(H, p=self.dropoutClassifier2, training=self.training)
        H = self.lin2(H)

        return F.log_softmax(H, dim=1)


