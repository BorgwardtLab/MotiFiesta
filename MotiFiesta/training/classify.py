import os

import torch
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from torch_geometric.nn import global_add_pool
from torch_geometric.data import DataLoader

from MotiFiesta.utils.learning_utils import load_model
from MotiFiesta.src.loading import get_loader

class Classifier:
    def __init__(self):
        pass
    def decode():
        raise NotImplementedError
    def train():
        raise NotImplementedError
    def eval():
        raise NotImplementedError

    def eval_kfold(self, dataset, inds_path, n_splits=10):
        accs = []
        for split in range(1, n_splits+1):
            with open(os.path.join(inds_path, f"test_idx-{split}.txt"), 'r') as s:
                test_inds = [int(i) for i in s.readlines()]
            with open(os.path.join(inds_path, f"train_idx-{split}.txt"), 'r') as s:
                train_inds = [int(i) for i in s.readlines()]

            test_data = DataLoader([dataset[i] for i in test_inds])
            train_data = DataLoader([dataset[i] for i in train_inds])
            self.train(train_data)
            accs.append(self.eval(test_data))

        return np.mean(accs), np.std(accs)

class MotiFiestaClassifierSK(Classifier):
    def __init__(self,
                 encoder,
                 decoder='RF',
                 sigma_filter='all',
                 sigma_k=3,
                 largest=True,
                 layers=3,
                 dummy=False):
        self.encoder = load_model(encoder)['model']
        self.layers = layers
        self.dummy = dummy
    
        self.sigma_k = sigma_k
        self.largest = largest

        self.keep_all = False
        self.sigma_filter = sigma_filter
        if sigma_filter == 'all':
            self.keep_all = True

        if decoder == 'RF':
            self.classifier = RandomForestClassifier()
        super(MotiFiestaClassifierSK).__init__()

    def decode(self, loader):
        X = []
        y = []
        skipped_idx = set()
        for i, g in enumerate(loader):
            g = g['pos']
            with torch.no_grad():
                embs,sigmas,ee,_,merge_info,_  = self.encoder(g.x,
                                                              g.edge_index,
                                                              g.batch,
                                                              dummy=self.dummy
                                                              )
            tree = merge_info['tree']
            if len(embs) < self.layers:
                skipped_idx.add(i)
                continue
            x_layers = []
            for l in range(1, self.layers):
                if not self.keep_all:
                    X_l = embs[l]
                    S_l = torch.tensor([self.total_sigma(l, i, tree, sigmas, ee) for i in range(X_l.shape[0])], dtype=torch.float32)
                    if self.sigma_filter == 'topk':
                        X_l = self.sigma_filter_topk(X_l, S_l, k=self.sigma_k, largest=self.largest)
                    if self.sigma_filter == 'random':
                        X_l = self.sigma_filter_randomk(X_l, self.sigma_k)
                else:
                    X_l = embs[l]
                b = torch.zeros(len(X_l), dtype=torch.long)
                h = global_add_pool(X_l, b)
                x_layers.append(h)
            x_glob = torch.cat(x_layers, dim=1).squeeze()
            X.append(x_glob.squeeze())
            y.append(g.y)
        X = torch.stack(X)
        y = torch.stack(y)
        return X.detach().numpy(), y.detach().numpy()

    def train(self, train_loader):
        X, y = self.decode(train_loader)
        self.classifier.fit(X, y)

    def eval(self, test_loader):
        X_test, y_test = self.decode(test_loader)
        return self.classifier.score(X_test, y_test)

    @staticmethod
    def sigma_filter_topk(X, S, k, largest=True):
        _, keep_inds = torch.topk(S, min(k, X.shape[0]), largest=largest)
        return X[keep_inds]

    @staticmethod
    def sigma_filter_randomk(X, k):
        keep_inds = torch.sort(torch.randperm(X.size(0))[:k])[0]
        return X[keep_inds]

    @staticmethod
    def total_sigma(level, node, tree, sigmas, ee):
        """ Recursively compute total sigma score
        for a subgraph.
        """
        children = list(tree[level][node])
        n_children = len(children)

        if n_children == 0:
            return 0
        elif n_children == 1:
            return MotiFiestaClassifierSK.total_sigma(level-1, children[0], tree, sigmas, ee)
        else:
            eind = (ee[level-1][0] == children[0]) &\
                   (ee[level-1][1] == children[1])
            eind = eind.nonzero()[0][0].item()
            score = sigmas[level-1][eind]
            # get score for this node
            return score +\
                   MotiFiestaClassifierSK.total_sigma(level-1, children[0], tree, sigmas, ee) +\
                   MotiFiestaClassifierSK.total_sigma(level-1, children[1], tree, sigmas, ee)





class MotiFiestaClassifierFineTune:
    def __init__(self):
        super(MotiFiestaClassifierFineTune).__init__()
        pass

    def train(self, train_loader):
        for batch in train_loader:
            pass
        pass
    pass


if __name__ == "__main__":
    d = 'NCI1'
    f = 'NCI1'
    clf = MotiFiestaClassifierSK(d, dummy=False, sigma_filter='all')
    loaders = get_loader(d, batch_size=8)
    acc = clf.eval_kfold(loaders['dataset_whole'], f'supervised_data/fold-idx/{d}')
    print(acc)
    pass
