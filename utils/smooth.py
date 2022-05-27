"""

Draw smoothness of embeddings.

"""
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

from MotiFiesta.utils.graph_utils import draw_one_instance
from MotiFiesta.utils.graph_utils import to_graphs
from MotiFiesta.utils.graph_utils import batch_to_node_indices
from MotiFiesta.utils.learning_utils import get_device
from MotiFiesta.utils.learning_utils import load_model
from MotiFiesta.utils.learning_utils import load_data
from MotiFiesta.src.loading import get_loader

def extract_subgraphs(dataset, model):
    subgraphs = defaultdict(list)
    for idx, g_pair in enumerate(dataset):
        g = g_pair['pos']

        batch = torch.zeros(len(g.x), dtype=torch.long)

        # return xx, pp, ee, batches, merge_info, internals
        out = model(g.x, g.edge_index, batch)
        embs, probas, merge_info = out[0], out[1], out[4]

        for t, x_t in enumerate(embs):
            for i,x in enumerate(x_t):
                # only include in hashtable if node got merged
                # i.e. node must have two children
                children = merge_info['tree'][t][i]
                if len(children) != 2 and t > 0:
                    continue
                # watch out for node ordering
                spotlight = list(merge_info['spotlights'][t][i])

                G = to_networkx(g)
                G = G.subgraph(spotlight).copy()

                subgraphs[t].append({'graph': G, 'x': x.detach().numpy()})

    return subgraphs

def kd_trees(subgraphs):
    trees = {}
    for t, sgs in subgraphs.items():
        X = np.array([sg['x'] for sg in sgs])
        tree = KDTree(X, leaf_size=2)
        trees[t] = tree
    return trees

def plot_one(size, subgraphs, trees):
    sgs = subgraphs[size]
    tree = trees[size]

    ind = random.randint(0, len(sgs))
    g = sgs[ind]['graph']
    x = sgs[ind]['x'].reshape(-1, 1).T


    dists, inds = tree.query(x, k=100, return_distance=True, sort_results=True)
    k = dists.shape[1]

    graphs = [g]
    distances = [0]
    for d, ind in zip(dists[0][::k//5], inds[0][::k//5]):
        graphs.append(sgs[ind]['graph'])
        distances.append(d)

    fig, ax = plt.subplots(1, len(graphs))
    for i,g in enumerate(graphs):
        nx.draw(g, ax=ax[i])
        if i == 0:
            ax[i].set_title("reference")
        else:
            ax[i].set_title(f"{distances[i]:.2f}")

    plt.show()
    pass

if __name__ == "__main__":
    run = 'ENZYMES'
    da = 'ENZYMES'
    model = load_model(run)
    data = get_loader(da, batch_size=2)

    subgraphs = extract_subgraphs(data['dataset_whole'], model)
    trees = kd_trees(subgraphs)
    for _ in range(10):
        plot_one(3, subgraphs, trees)
    pass
