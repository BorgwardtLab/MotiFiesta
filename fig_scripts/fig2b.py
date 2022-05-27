"""
Draw a spread of decoded motif subgraphs.
"""
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

from MotiFiesta.utils.learning_utils import load_model
from MotiFiesta.src.loading import get_loader
from MotiFiesta.src.decode import HashDecoder

run = 'PROTEINS'
dataset = 'PROTEINS'

tic = time.time()
decoder = HashDecoder(run, dataset, hash_dim=32, level=3)
gs = decoder.decode(n_graphs=-1)
toc = time.time()
print(toc-tic)
sys.exit()

# select the motif to draw
motifs_pred_all, true_motif_ids, sigma_all = HashDecoder.collect_output(gs)

print(motifs_pred_all)
motif_ids,counts = torch.unique(motifs_pred_all, return_counts=True)
sigma_avg = torch.zeros_like(motif_ids, dtype=torch.float32).scatter_add(0, motifs_pred_all, sigma_all)
print("ids ", motif_ids)
print("counts ", counts)
print("scores raw: ", sigma_avg)
sigma_avg /= counts
print("scores avg: ", sigma_avg)
top_motif = sigma_avg.argmax() + 1

print(f"motif scores: {sigma_avg}")
print(f"top motif: {top_motif}")

fig, ax = plt.subplots(4, 4)

subgraphs = []

for graph in gs:
    motif_nodes = torch.nonzero(graph.motif_pred == top_motif)
    true_nodes = set(torch.nonzero(graph.motif_id != 0).squeeze().numpy())
    spotlight_ids = graph.spotlight_ids[motif_nodes]
    todo_spotlights = torch.unique(spotlight_ids)
    g_nx = to_networkx(graph)
    for spotlight in todo_spotlights:
        motif_nodes = torch.nonzero(graph.spotlight_ids == spotlight)
        motif_nodes = set(motif_nodes.squeeze().numpy())
        context = set()
        for n in motif_nodes:
            context |= set(g_nx.neighbors(n))
        if len(motif_nodes) < 2:
            continue
        subg = g_nx.subgraph(motif_nodes | context).copy()
        # clean up induced edges
        remove_edges = []
        for (u, v) in subg.edges():
            if (u not in motif_nodes) and (v not in motif_nodes):
                remove_edges.append((u, v))
        subg.remove_edges_from(remove_edges)
        subgraphs.append((subg, true_nodes))

print("instances ", len(subgraphs))

c = 0
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        try:
            g, true_nodes = subgraphs[c]
            colors = ['blue' if n in true_nodes else 'grey' for n in g.nodes()]
            nx.draw(g.to_undirected(), node_color=colors, node_size=50, ax=ax[i][j])
            c += 1
        except:
            continue
plt.show()
