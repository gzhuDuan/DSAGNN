# from torch_geometric.datasets import WebKB,WikipediaNetwork,Actor,Planetoid
import torch
import random
import os
import numpy as np
import os.path as osp
import pickle
import gdown
from sympy import print_tree
from torch_sparse import SparseTensor




def index_to_mask(index, size):
    # mask = torch.zeros(size, dtype=torch.bool)
    # mask[index] = 1
    mask = torch.zeros(size, dtype=torch.bool)
    # 确保 index 是 Python 列表或展平的 NumPy 数组
    index = np.asarray(index).flatten().tolist()
    mask[index] = 1
    return mask


# 随机拆分
def random_splits0(data, num_classes, percls_trn, val_lb, seed):


    if not hasattr(data, 'y') or not hasattr(data, 'num_nodes'):
        raise ValueError("数据对象缺少必要属性：'y' 或 'num_nodes'。")


    y = data.y.cpu() if torch.is_tensor(data.y) and data.y.is_cuda else data.y
    num_nodes = data.num_nodes


    target_train_size = percls_trn * num_classes  # 333 * 6 = 1998
    target_test_size = num_nodes - target_train_size - val_lb  # 3327 - 1998 - 665 = 664


    if target_train_size + val_lb + target_test_size != num_nodes:
        raise ValueError(
            f"划分大小不匹配：训练 {target_train_size} + 验证 {val_lb} + 测试 {target_test_size} != {num_nodes}")


    index = list(range(num_nodes))
    train_idx = []
    rnd_state = np.random.RandomState(seed)


    for c in range(num_classes):
        class_idx = np.where(y == c)[0]
        if len(class_idx) == 0:
            print(f"类别 {c} 没有样本。")
            continue
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx.tolist())
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False).tolist())

    if len(train_idx) < target_train_size:
        remaining = list(set(index) - set(train_idx))
        extra = rnd_state.choice(remaining, target_train_size - len(train_idx), replace=False).tolist()
        train_idx.extend(extra)


    rest_index = list(set(index) - set(train_idx))


    if len(rest_index) < val_lb:
        raise ValueError(f"验证集样本不足：可用 {len(rest_index)}，需要 {val_lb}。")


    val_idx = rnd_state.choice(rest_index, val_lb, replace=False).tolist()  # 转换为列表


    test_idx = [i for i in rest_index if i not in set(val_idx)]


    if len(train_idx) != target_train_size:
        raise ValueError(f"训练集大小 {len(train_idx)} 未达预期 {target_train_size}。")
    if len(val_idx) != val_lb:
        raise ValueError(f"验证集大小 {len(val_idx)} 未达预期 {val_lb}。")
    if len(test_idx) != target_test_size:
        raise ValueError(f"测试集大小 {len(test_idx)} 未达预期 {target_test_size}。")


    if len(set(train_idx) & set(val_idx)) > 0 or len(set(val_idx) & set(test_idx)) > 0 or len(
            set(train_idx) & set(test_idx)) > 0:
        raise ValueError("训练、验证或测试索引存在重叠。")


    all_idx = train_idx + val_idx + test_idx
    if max(all_idx, default=-1) >= num_nodes:
        raise ValueError("索引超出 data.num_nodes。")

    # 创建掩码
    data.train_mask = index_to_mask(train_idx, size=num_nodes)
    data.val_mask = index_to_mask(val_idx, size=num_nodes)
    data.test_mask = index_to_mask(test_idx, size=num_nodes)

    return data



def full_supervised_random_splits1(data, runs, dataset_str):
    if dataset_str == 'Actor':
        dataset_str = 'film'
    if dataset_str == 'Squirrel_filtered' or dataset_str == 'Chameleon_filtered':
        if dataset_str == 'Squirrel_filtered':
            dataset_str = 'squirrel'
        if dataset_str == 'Chameleon_filtered':
            dataset_str = 'chameleon'

        data.train_mask = data.train_mask_total[runs].bool()
        data.val_mask = data.val_mask_total[runs].bool()
        data.test_mask = data.test_mask_total[runs].bool()
        return data
    else:
        splits_file_path = './new_data/' + dataset_str.lower() + '/raw/' + dataset_str.lower() + \
                           '_split_0.6_0.2_' + str(runs - 1) + '.npz'
        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']
        data.train_mask = torch.from_numpy(train_mask).bool()
        data.val_mask = torch.from_numpy(val_mask).bool()
        data.test_mask = torch.from_numpy(test_mask).bool()

    return data


# 20/500/1000
def semi_supervisedrandom_splits2(data, num_classes, seed):
    percls_trn = 20
    val_lb = 500
    test_num = 1000
    index = [i for i in range(0, data.y.shape[0])]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]
    test_idx = rnd_state.choice(test_idx, test_num, replace=False)

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)

    return data

def random_facebook10_splits3(data,dataset,runs):
    name = dataset
    # if name == 'Penn94':
    name = 'fb100-Penn94'
    # if not os.path.exists(f'./data/splits/{name}-splits.npy'):
    #     assert dataset in splits_drive_url.keys()
    #     gdown.download(
    #         id=splits_drive_url[dataset], \
    #         output=f'./data/splits/{name}-splits.npy', quiet=False)

    splits_lst = np.load(f'./new_data2/splits/{name}-splits.npy', allow_pickle=True)
    # for i in range(len(splits_lst)):
    #     print(f"Run {i}: Train {len(splits_lst[i]['train'])}, Val {len(splits_lst[i]['valid'])}, Test {len(splits_lst[i]['test'])}")

    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])

    train_idx = splits_lst[runs]['train']
    val_idx=splits_lst[runs]['valid']
    test_idx=splits_lst[runs]['test']

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    # label = data.y
    # unlabeled_count = (label == -1).sum().item()
    # print(f"未标记节点数: {unlabeled_count}")
    return data

def dataset_splits(args, data):
    split_case = args.split  # 0 1 2
    num_classes = data.num_classes #总类别数
    percls_trn = int(round(args.train_rate * len(data.y) / data.num_classes)) #每个类别选取的训练样本数
    val_lb = int(round(args.val_rate * len(data.y))) #验证集固定大小
    seed = args.seed
    runs = args.runs
    dataset = args.dataset
    if split_case == 0:
        data1 = random_splits0(data, num_classes, percls_trn, val_lb, seed)
    elif split_case == 1:
        data1 = full_supervised_random_splits1(data, runs, dataset)
    elif split_case == 2:
        data1 = semi_supervisedrandom_splits2(data, num_classes, seed)
    elif split_case == 3:
        data1 = random_facebook10_splits3(data, dataset, runs)
    return data1

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx




def to_sparse_tensor(edge_index, edge_feat, num_nodes):
    """ converts the edge_index into SparseTensor
    """
    num_edges = edge_index.size(1)

    (row, col), N, E = edge_index, num_nodes, num_edges
    perm = (col * N + row).argsort()
    row, col = row[perm], col[perm]

    value = edge_feat[perm]
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(N, N), is_sorted=True)

    # Pre-process some important attributes.
    adj_t.storage.rowptr()
    adj_t.storage.csr2csc()

    return adj_t

import scipy.sparse as sp

def normalize_tensor(mx, eqvar = None):
    """Row-normalize sparse matrix"""
    mx = sp.csr_matrix(mx)
    rowsum = np.array(mx.sum(1))
    if eqvar:
        r_inv = np.power(rowsum, -1.0/eqvar).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv, 0)
        mx = r_mat_inv.dot(mx)
    else:
        r_inv = np.power(rowsum, -1.0).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv, 0)
        mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_fixed_splits(dataset, sub_dataset):
    """ loads saved fixed splits for dataset
    """
    name = dataset
    if sub_dataset and sub_dataset != 'None':
        name += f'-{sub_dataset}'

    if not os.path.exists(f'./new_data2/splits/{name}-splits.npy'):
        assert dataset in splits_drive_url.keys()
        gdown.download(
            id=splits_drive_url[dataset], \
            output=f'./new_data2/splits/{name}-splits.npy', quiet=False)

    splits_lst = np.load(f'./new_data2/splits/{name}-splits.npy', allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    return splits_lst

splits_drive_url = {
    'snap-patents' : '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N',
    'pokec' : '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_',
}