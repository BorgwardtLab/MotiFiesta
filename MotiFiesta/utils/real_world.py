"""
    Generate graphs containing synthetic motifs.
    The idea:
        1. A motif is a random graph
        2. For each instance, insert motif graph into larger random graph.
        4. Insertion means sampling a random node from parent graph and replacing it with motif.
        5. Links between motif and parent graph are made by sampling random nodes in both subgraphs with same probability
           as the edge probability which generated the graphs.
        6. Repeat for each motif.
"""
import os
import os.path as osp
import random
import itertools

import numpy as np
from numpy.random import normal
from scipy.stats import beta
from scipy.stats import uniform
from scipy.stats import expon
import networkx as nx
from networkx.generators.random_graphs import connected_watts_strogatz_graph
from networkx.generators.random_graphs import powerlaw_cluster_graph
from networkx.generators.random_graphs import extended_barabasi_albert_graph
from networkx.generators.random_graphs import erdos_renyi_graph
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.utils import to_networkx

random.seed(0)
np.random.seed(0)

def rewire(g_pyg, n_iter=100):
    """ Apply (u, v), (u', v') --> (u, v'), (v, u') to randomize graph.
    """
    has_features = g_pyg.x is not None
    if has_features:
        g_nx = to_networkx(g_pyg, node_attrs=['x'])
    else:
        g_nx = to_networkx(g_pyg)
    rewired_g = g_nx.copy()
    for n in range(n_iter):
        e1, e2 = random.sample(list(g_nx.edges()), 2)
        rewired_g.remove_edges_from([e1, e2])
        rewired_g.add_edges_from([(e1[0], e2[1]), (e1[1], e2[0])])

    rewired_g.remove_edges_from(list(nx.selfloop_edges(rewired_g)))
    if has_features:
        rewired_pyg = from_networkx(g_nx, group_node_attrs=['x'])
    else:
        rewired_pyg = from_networkx(g_nx)
    return rewired_pyg


class RealWorldDataset(Dataset):
    def __init__(self,
                 root="ENZYMES",
                 n_swap=100,
                 transform=None,
                 seed=0,
                 max_degree=100,
                 n_features=None):
        """ Builds the synthetic motif dataset. Motifs are built on the
        fly and stored to disk.

        Args:
        ---
        root (str): path to folder where graphs will be stores.
        n_graphs (int): number of graphs to generate
        n_motifs (int): number of motifs to inject in graphs
        """
        self.seed = seed
        self.n_swap = n_swap
        self.base_data = TUDataset(root='data', name=root)
        self.max_degree = max_degree
        self.n_features = n_features

        super(RealWorldDataset, self).__init__("data/"+root, transform)

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.base_data))]

    @property
    def num_features(self):
        if not self.n_features is None:
            return self.n_features
        return self.base_data.num_features

    def process(self):
        gs = []
        for g in self.base_data:
            # add dummy motif_id field
            try:
                g['x']
            except KeyError:
                T.OneHotDegree(self.max_degree)(g)
            g_neg = rewire(g, n_iter=self.n_swap)
            g.motif_id = torch.zeros(g.num_nodes)
            g_neg.motif_id = torch.zeros(g.num_nodes)
            gs.append({'pos': g, 'neg': g_neg})

        for i, g_pyg in enumerate(gs):
            torch.save(g_pyg, osp.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """ Returns dictionary where 'pos' key stores batch with
        graphs that contain the motif, and the 'neg' key has batches
        without the motif.
        """
        if idx > len(self) - 1:
            raise StopIteration
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

if __name__ == "__main__":
    # d = SyntheticMotifs(motif_type='barbell', distort_p=.02, motif_size=8, root="barbell-pair-s")
    d = RealWorldDataset(root='IMDB-BINARY', max_degree=300)

    # l = DataLoader(d, batch_size=1)
    # for batch in l:
        # print(batch)
        # print(batch['pos'].edge_index)
        # print(batch['pos'].x)
