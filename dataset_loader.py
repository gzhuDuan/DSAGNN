import scipy.sparse as sp
import torch
import math
import pickle
import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Actor, WikipediaNetwork,LINKXDataset,HeterophilousGraphDataset
import os
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from torch_sparse import coalesce
from torch_geometric.utils.undirected import to_undirected
from datasets import load_fb100, DATAPATH
from sklearn.preprocessing import label_binarize
from util import rand_train_test_idx, to_sparse_tensor,sparse_mx_to_torch_sparse_tensor

class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='new_data/', name=None,
                 p2raw=None,
                 train_percent=0.01,
                 transform=None, pre_transform=None):
        if name == 'actor':
            name = 'film'
        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.train_percent = self.data.train_percent.item()

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class WebKB(InMemoryDataset):
    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_fb100_dataset(filename):
    A, metadata = load_fb100(filename)
    dataset = NCDataset(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes, 'class': 2}
    dataset.label = torch.tensor(label)


    data = Data(x=dataset.graph['node_feat'], y=torch.squeeze(dataset.label),
                edge_index=dataset.graph['edge_index'], num_nodes=dataset.graph['num_nodes'],
                num_classes=dataset.graph['class'])
    data.edge_index = to_undirected(data.edge_index)

    return data


def DataLoader(name):
    name = name.lower()
    if name in ['cora', 'citeseer', 'pubmed']:

        dataset = Planetoid(root='./data', name=name, transform=T.NormalizeFeatures())
        num_class1 = dataset.num_classes
        dataset = dataset[0]
        dataset.num_classes = num_class1
    elif name in ['chameleon', 'squirrel']:

        dataset = WikipediaNetwork(root='./new_data', name=name, transform=T.NormalizeFeatures())

    elif name in ['film']:
        # dataset = dataset_heterophily(root='./data/', name=name, transform=T.NormalizeFeatures())
        dataset = Actor(root='./new_data/film', transform=T.NormalizeFeatures())
    elif name in ['ogbn-arxiv', 'ogbn-products', 'pokec', 'arxiv-year', 'genius', 'twitch-gamer', 'snap-patents',
                  'penn94']:
        dataset = load_nc_dataset(name)

    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root='./new_data', name=name, transform=T.NormalizeFeatures())
        num_class1 = dataset.num_classes
        dataset = dataset[0]
        dataset.num_classes = num_class1
    elif name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        dataset = HeterophilousGraphDataset(root='./new_data3', name=name, transform=T.NormalizeFeatures())
        num_class1 = dataset.num_classes
        dataset = dataset[0]
        dataset.num_classes = num_class1


    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset


def load_nc_dataset(dataname):
    """ Loader for NCDataset, returns NCDataset. """
    # if dataname == 'twitch-e':
    #     # twitch-explicit graph
    #
    #     dataset = load_twitch_dataset(dataname)
    if dataname == 'penn94':
        dataname = 'Penn94'
        dataset = load_fb100_dataset(dataname)
    # elif dataname == 'ogbn-proteins':
    #     dataset = load_proteins_dataset()
    # elif dataname == 'deezer-europe':
    #     dataset = load_deezer_dataset()
    # elif dataname == 'arxiv-year':
    #     dataset = load_arxiv_year_dataset()
    # elif dataname == 'pokec':
    #     dataset = load_pokec_mat()
    # elif dataname == 'snap-patents':
    #     dataset = load_snap_patents_mat()
    # elif dataname == 'yelp-chi':
    #     dataset = load_yelpchi_dataset()
    # elif dataname in ('ogbn-arxiv', 'ogbn-products'):
    #     dataset = load_ogb_dataset(dataname)
    # elif dataname in ('Cora', 'CiteSeer', 'PubMed'):
    #     dataset = load_planetoid_dataset(dataname)
    # elif dataname in ('chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin'):
    #     dataset = load_geom_gcn_dataset(dataname)
    # elif dataname == "genius":
    #     dataset = load_genius()
    # elif dataname == "twitch-gamer":
    #     dataset = load_twitch_gamer_dataset()
    # elif dataname == "wiki":
    #     dataset = load_wiki()
    else:
        raise ValueError('Invalid dataname')
    return dataset
