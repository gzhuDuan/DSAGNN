import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor

# from sympy.physics.quantum.tests.test_qubit import epsilon
from dataset_loader import DataLoader
from eoNet import EONet

from util import dataset_splits, to_sparse_tensor,load_fixed_splits
import matplotlib.pyplot as plt
import random
import numpy as np

import os
import argparse
import datasets
from tqdm import tqdm
import seaborn as sns
import time


# 固定随机种子保证可重复性
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(data, num_node_features, num_classes, Ko, Ke,dropoutClassifier1,dropoutClassifier2,
        hidden,
        Gamma, Epsilon):
    def train(model, optimizer, data):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        reg_loss = None
        loss.backward()
        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        with torch.no_grad():
            logits, accs, losses, preds = model(data), [], [], []
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                pred = logits[mask].max(1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                loss = F.nll_loss(logits[mask], data.y[mask])
                preds.append(pred)
                accs.append(acc)
                losses.append(loss)
        return accs, preds, losses

    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

    permute_masks = dataset_splits
    data = permute_masks(args, data)
    data = data.to(device)


    print(f"训练样本数: {data.train_mask.sum().item()}")
    print(f"验证样本数: {data.val_mask.sum().item()}")
    print(f"测试样本数: {data.test_mask.sum().item()}")
    model = EONet(num_node_features, num_classes, Ko, Ke, dropoutClassifier1,dropoutClassifier2,
                  hidden,
                  Gamma, Epsilon)
    model = model.to(device)

    if args.optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW([
            {'params': model.even_hop.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': model.odd_hop.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': model.lin1.parameters(), 'lr': args.lin_lr1, 'weight_decay': args.lin_weight_decay1},
            {'params': model.lin2.parameters(), 'lr': args.lin_lr2, 'weight_decay': args.lin_weight_decay2},
            {'params': model.lin5.parameters(), 'lr': args.alpha_lin_lr, 'weight_decay': args.alpha_lin_weight_decay}

        ])
    elif args.optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop([
            {'params': model.even_hop.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': model.odd_hop.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': model.lin1.parameters(), 'lr': args.lin_lr1, 'weight_decay': args.lin_weight_decay1},
            {'params': model.lin2.parameters(), 'lr': args.lin_lr2, 'weight_decay': args.lin_weight_decay2},
            {'params': model.lin5.parameters(), 'lr': args.alpha_lin_lr, 'weight_decay': args.alpha_lin_weight_decay},

        ])
    elif args.optimizer_name == 'Adam':
        optimizer = torch.optim.Adam([
            {'params': model.even_hop.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': model.odd_hop.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': model.lin1.parameters(), 'lr': args.lin_lr1, 'weight_decay': args.lin_weight_decay1},
            {'params': model.lin2.parameters(), 'lr': args.lin_lr2, 'weight_decay': args.lin_weight_decay2},
            {'params': model.lin5.parameters(), 'lr': args.alpha_lin_lr, 'weight_decay': args.alpha_lin_weight_decay},

        ])
    # 初始化早停相关变量
    patience = args.patience
    best_val_loss = float('inf')
    patience_counter = 0
    best_val_acc = 0
    best_model_state = None

    # 初始化存储指标
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    time_run = []

    # model.train()
    for epoch in range(args.epochs):
        # 训练阶段
        t_st = time.time()

        train(model, optimizer, data)
        time_epoch = time.time() - t_st  # each epoch train times
        time_run.append(time_epoch)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)



        # 记录指标
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 早停判断
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            patience_counter = 0
            # best_model_state = model.state_dict().copy()

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch}')
            break

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return test_acc, best_val_acc, time_run


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    choices=['Cora', 'Citeseer', 'Pubmed', 'Chameleon',
                             'Squirrel', 'Actor', 'Texas', 'Cornell', 'Wisconsin', 'Penn94', 'Genius',
                             'Chameleon_filtered','Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions'],
                    default='Squirrel')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')
parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
parser.add_argument('--Ko', type=int, default=10, help='odd propagation steps.')
parser.add_argument('--Ke', type=int, default=10, help='even propagation steps.')
parser.add_argument('--Gamma', type=float, default=0.1, help='alpha for even')
parser.add_argument('--Epsilon', type=float, default=0.1, help='Epsilon for odd')
parser.add_argument('--lin_lr1', type=float, default=0.01, help='lin_lr learning rate.')
parser.add_argument('--lin_weight_decay1', type=float, default=0.0005, help='lin_weight_decay.')
parser.add_argument('--lin_lr2', type=float, default=0.01, help='lin_lr learning rate.')
parser.add_argument('--lin_weight_decay2', type=float, default=0.0005, help='lin_weight_decay.')
parser.add_argument('--alpha_lin_lr', type=float, default=0.01, help='alpha_lin_lr learning rate.')
parser.add_argument('--alpha_lin_weight_decay', type=float, default=0.0005, help='alpha_lin_weight_decay.')

parser.add_argument('--optimizer_name', type=str, choices=['RMSprop', 'Adam', 'AdamW'], default='Adam')
parser.add_argument('--dropoutClassifier1', type=float, default=0.5, help='dropoutClassifier for neural networks.')
parser.add_argument('--dropoutClassifier2', type=float, default=0.5, help='dropoutClassifier for neural networks.')

parser.add_argument('--seed', type=int, default=60, help='seeds for random splits.')
parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
parser.add_argument('--split', type=int, default=0, help='dataset split')
parser.add_argument('--patience', type=int, default=200, help='patience.')
parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
parser.add_argument('--device', type=int, default=0, help='GPU device.')
parser.add_argument('--runs', type=int, default=10, help='number of runs.')

args = parser.parse_args()

SEEDS = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]
device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
# 加载数据集
if args.dataset in ['Chameleon_filtered', 'Squirrel_filtered']:
    ##处理原始数据_filtered.npz
    data = datasets.load_dataset_filtered(args.dataset.lower()).to(device)
elif args.dataset in ['Actor', 'Chameleon', 'Squirrel']:
    ##若已经有graph_edges等处理过的数据则不需要处理原始数据
    if args.dataset == 'Actor':
        data = datasets.load_dataset('film').to(device)
    else:
        data = datasets.load_dataset(args.dataset.lower()).to(device)
elif args.dataset in ['Penn94', 'Genius']:
    ##处理原始数据_filtered.npz
    # data = DataLoader(args.dataset.lower()).to(device)
    data = DataLoader(args.dataset.lower())
    # data.edge_index = to_sparse_tensor(data.edge_index, data.num_nodes).to(device)
    # data.edge_index = to_sparse_tensor(data)
else:
    data = DataLoader(args.dataset.lower()).to(device)




####################################
results = []
time_results = []
run_num = 10
for RP in tqdm(range(run_num)):
    args.seed = SEEDS[RP]
    set_seed(args.seed)
    args.runs = RP

    test_acc, best_val_acc, time_run = run(data, data.num_node_features, data.num_classes, args.Ko,
                                           args.Ke, args.dropoutClassifier1,args.dropoutClassifier2,
                                           args.hidden,
                                           args.Gamma, args.Epsilon)
    time_results.append(time_run)
    results.append([test_acc, best_val_acc])
    print(f'run_{str(RP + 1)} \t test_acc: {test_acc:.4f}')
    print(torch.cuda.is_available())
print(args)
run_sum = 0
epochsss = 0
for i in time_results:
    run_sum += sum(i)
    epochsss += len(i)

print("each run avg_time:", run_sum / 10, "s")
print("each epoch avg_time:", 1000 * run_sum / epochsss, "ms")

test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

values = np.asarray(results)[:, 0]
uncertainty = np.max(
    np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))

print(f'dataset {args.dataset}, in 10 repeated experiment:')
print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}  \t val acc mean = {val_acc_mean:.4f}')
