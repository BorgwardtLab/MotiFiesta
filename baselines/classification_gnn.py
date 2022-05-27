# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import copy
import pandas as pd
import torch
from collections import defaultdict
from timeit import default_timer as timer

from torch import nn
from torch_geometric import datasets
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import degree

from models import GNN, EdgePool


def load_args():
    parser = argparse.ArgumentParser(
        description='Graph classification with GNNs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="PROTEINS",
                        help='name of dataset')
    parser.add_argument('--fold-idx', type=int, default=1,
                        help='indices for the train/test datasets')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')

    # Model hyperparameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--embed-dim', type=int, default=64,
                        help='hidden dimensions')
    parser.add_argument('--gnn-type', type=str, default='gcn')
    parser.add_argument('--model-type', type=str, default='gnn')
    parser.add_argument('--global-pool', type=str, default='sum',
                        choices=['sum', 'mean', 'max'])
    parser.add_argument('--dropout', type=float, default=0.0)

    # Optimization hyperparameters
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-04)

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()

    args.save_logs = False
    if args.outdir != '':
        args.save_logs = True
        # outdir = args.outdir
        outdir = args.outdir + '/{}/{}_{}_{}_{}_{}_{}/fold-{}'
        outdir = outdir.format(
            args.dataset, args.gnn_type, args.num_layers, args.embed_dim,
            args.global_pool, args.lr, args.weight_decay,
            args.fold_idx
        )
        try:
            os.makedirs(outdir, exist_ok=True)
        except:
            pass
        args.outdir = outdir

    return args

def get_splits(dataset, fold_idx):
    idxpath = '../supervised_data/fold-idx/{}/{}_idx-{}.txt'

    test_idx = np.loadtxt(idxpath.format(dataset, 'test', fold_idx), dtype=int)
    train_idx = np.loadtxt(idxpath.format(dataset, 'train', fold_idx), dtype=int)
    size = len(train_idx)
    val_size = size // 9
    val_idx = train_idx[-val_size:]
    train_idx = train_idx[:-val_size]
    return train_idx, val_idx, test_idx


def train_epoch(model, loader, criterion, optimizer, use_cuda=False):
    model.train()

    running_loss = 0.0
    running_acc = 0.0

    tic = timer()
    for i, data in enumerate(loader):
        size = len(data.y)
        # data.y = data.y.float()

        if use_cuda:
            data = data.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        pred = output.data.argmax(dim=1)
        running_loss += loss.item() * size
        running_acc += torch.sum(pred == data.y).item()

    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_acc = running_acc / n_sample
    print('Train loss: {:.4f} Acc: {:.4f} time: {:.2f}s'.format(
          epoch_loss, epoch_acc, toc - tic))
    return epoch_loss


@torch.no_grad()
def eval_epoch(model, loader, criterion, use_cuda=False):
    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    tic = timer()
    for data in loader:
        if use_cuda:
            data = data.cuda()
        labels = data.y

        output = model(data)
        loss = criterion(output, labels)

        pred = output.data.argmax(dim=-1)
        running_acc += torch.sum(pred == labels).item()

        running_loss += loss.item() * len(data)
    toc = timer()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_acc = running_acc / n_sample
    print('Val loss: {:.4f} Acc: {:.4f} time: {:.2f}s'.format(
          epoch_loss, epoch_acc, toc - tic))
    return epoch_acc, epoch_loss

def main():
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    datapath = '../supervised_data/TUDataset'

    dset = datasets.TUDataset(datapath, args.dataset)
    in_dim = dset.num_node_labels
    num_class = dset.num_classes

    if dset.data.x is None:
        max_degree = 0
        degs = []
        for data in dset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        # if max_degree < 1000:
        dset.transform = T.OneHotDegree(max_degree)
        in_dim = max_degree + 1

    print(dset)
    train_idx, val_idx, test_idx = get_splits(args.dataset, args.fold_idx)
    # print(len(train_idx))
    # print(len(val_idx))
    # print(len(test_idx))
    # print(len(train_idx) + len(val_idx) + len(test_idx))

    # Data Loaders
    train_loader = DataLoader(dset[train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dset[val_idx], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dset[test_idx], batch_size=args.batch_size, shuffle=False)

    # Create model
    if args.model_type == 'edgepool':
        print("use EdgePool")
        model = EdgePool(
            in_dim=in_dim,
            num_class=num_class,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            # gnn_type=args.gnn_type,
            global_pool=args.global_pool,
            # dropout=args.dropout
        )
    else:
        model = GNN(
            in_dim=in_dim,
            num_class=num_class,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            gnn_type=args.gnn_type,
            global_pool=args.global_pool,
            dropout=args.dropout)

    print(model)
    if args.use_cuda:
        model.cuda()

    # Create optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Training
    print("Start training...")
    best_val_acc = 0
    best_model = None
    best_epoch = 0
    logs = defaultdict(list)
    start_time = timer()
    for epoch in range(args.epochs):
        print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.use_cuda)
        val_acc, val_loss = eval_epoch(model, val_loader, criterion, args.use_cuda)
        test_acc, test_loss = eval_epoch(model, test_loader, criterion, args.use_cuda)

        if lr_scheduler is not None:
            lr_scheduler.step()

        logs['train_loss'].append(train_loss)
        logs['val_acc'].append(val_acc)
        logs['val_loss'].append(val_loss)
        logs['test_acc'].append(test_acc)
        logs['test_loss'].append(test_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
    total_time = timer() - start_time
    print("best epoch: {} best val acc: {:.4f}".format(best_epoch, best_val_acc))
    model.load_state_dict(best_weights)

    print()
    print("Testing...")
    test_acc, test_loss = eval_epoch(model, test_loader, criterion, args.use_cuda)

    print("test Acc {:.4f}".format(test_acc))

    if args.save_logs:
        logs = pd.DataFrame.from_dict(logs)
        # logs_suffix = '_test' if args.test else ''
        logs.to_csv(args.outdir + '/logs.csv')
        results = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'total_time': total_time,
        }
        results = pd.DataFrame.from_dict(results, orient='index')
        results.to_csv(args.outdir + '/results.csv',
                       header=['value'], index_label='name')
        print('Saved!')

if __name__ == "__main__":
    main()
