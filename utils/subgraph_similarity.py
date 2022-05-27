from itertools import combinations
from itertools import starmap
import numpy as np
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as wl
import torch
from igraph import Graph
from wwl import wwl


def build_K(subgraphs, cache=None):
    graph_pairs = ((*c, cache) for c in combinations(subgraphs, 2))
    d = list(starmap(subgraph_sim_dgl, graph_pairs))
    N = len(subgraphs)

    block = np.zeros((N, N))
    block[np.triu_indices(N, 1)] = d
    block += block.T
    block += np.eye(N)

    return torch.tensor(block, dtype=torch.float)

def build_wwl_K(graphs, node_features=None):
    graphs = [Graph.from_networkx(g) for g in graphs]
    kernel_matrix = wwl(graphs,
                        node_features,
                        num_iterations=4
                        )
    return torch.tensor(kernel_matrix, dtype=torch.float)

def subgraph_sim(sg1, sg2, timeout=1):
    return nx.algorithms.graph_edit_distance(sg1, sg2, timeout=timeout)

def subgraph_sim_dgl(sg1, sg2, cache, beta=.5, node_attr=None, edge_attr=None):
    G1 = dgl.from_networkx(sg1)
    G2 = dgl.from_networkx(sg2)

    node_sub, edge_sub = (None, None)

    if not node_attr is None:
        node_sub = build_sub_matrix(G1, G2, node_attr, mode='node')
    if not edge_attr is None:
        edge_sub = build_sub_matrix(G1, G2, edge_attr, mode='edge')

    if not cache is None:
        h_g1 = wl(sg1)
        h_g2 = wl(sg2)
        h_g1, h_g2 = sorted([h_g1, h_g2])
        try:
            distance = cache[h_g1][h_g2]
        except KeyError:
            try:
                distance,_,_ = graph_edit_distance(G1,
                                                   G2,
                                                   algorithm='hausdorff',
                                                   node_substitution_cost=node_sub,
                                                   edge_substitution_cost=edge_sub
                                                   )
            except Exception as e:
                print(e)
                print(G1.nodes(), G2.nodes())
                distance = float(abs(len(G1.nodes()) - len(G2.nodes())))
            cache[h_g1][h_g2] = distance
    else:
            distance,_,_ = graph_edit_distance(G1, G2, algorithm='hausdorff')

    sim = np.exp(-beta * distance)
    return sim
