import itertools
from collections import defaultdict

import torch
from torch_geometric.data import Data
import networkx as nx
from networkx.algorithms.swap import connected_double_edge_swap
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.utils import k_hop_subgraph


def induced_edge_filter_(G, roots, depth=1):
    """
        Remove edges in G introduced by the induced
        sugraph routine.
        Only keep edges which fall within a single
        node's neighbourhood.
        :param G: networkx subgraph
        :param roots: nodes to use for filtering
        :param depth: size of neighbourhood to take around each node.
        :returns clean_g: cleaned graph
    """
    # a depth of zero does not make sense for this operation as it would remove all edges
    if depth < 1:
        depth = 1
    neighbourhoods = []
    flat_neighbors = set()
    for root in roots:
        root_neighbors = bfs_expand(G, [root], hops=depth)
        neighbourhoods.append(root_neighbors)
        flat_neighbors = flat_neighbors.union(root_neighbors)

    flat_neighbors = list(flat_neighbors)
    subG = G.subgraph(flat_neighbors)
    subG = subG.copy()
    # G_new = G_new.subgraph(flat_neighbors)
    kill = []
    for (u, v) in subG.edges():
        for nei in neighbourhoods:
            if u in nei and v in nei:
                break
        else:
            kill.append((u, v))

    subG.remove_edges_from(kill)
    return subG

def induced_edge_filter(G, roots):
    kill = []
    for (u, v) in G.edges():
        if u not in roots and v not in roots:
            kill.append((u, v))
    G.remove_edges_from(kill)

def bfs_expand(G, initial_nodes, hops=2):
    """
        Extend motif graph starting with motif_nodes.
        Returns list of nodes.
    """

    total_nodes = [list(initial_nodes)]
    for d in range(hops):
        depth_ring = []
        for n in total_nodes[d]:
            for nei in G.neighbors(n):
                depth_ring.append(nei)
        else:
            total_nodes.append(depth_ring)
        # total_nodes.append(depth_ring)
    return set(itertools.chain(*total_nodes))


def bfs(G, initial_node, depth=2):
    """
        Generator for bfs given graph and initial node.
        Yields nodes at next hop at each call.
    """

    total_nodes = [[initial_node]]
    visited = []
    for d in range(depth):
        depth_ring = []
        for n in total_nodes[d]:
            visited.append(n)
            for nei in G.neighbors(n):
                if nei not in visited:
                    depth_ring.append(nei)
        total_nodes.append(depth_ring)
        yield depth_ring

def to_graphs(batch):
    """Convert batch to list of networkx subgraphs"""
    big_g = to_networkx(batch)
    nodelist = lambda i : list(np.argwhere(batch.batch==i).numpy()[0])
    graphs = [big_g.subgraph(nodelist(i)).copy() \
              for i in range(batch.num_graphs)]
    return graphs

def get_edge_subgraphs(edge_index, spotlights, level, graphs, x_base, batch, hop=False):
    """Return one spotlight subgraph per edge.
    """
    subgraphs = []
    X = []
    for u, v in edge_index.T:
        spotlight_u = spotlights[level][u.item()]
        spotlight_v = spotlights[level][v.item()]
        spotlight = spotlight_u.union(spotlight_v)

        graph = graphs[batch[list(spotlight)[0]]]
        subgraph = graph.subgraph(spotlight).copy()
        node_features = np.stack([x_base[n].cpu().numpy() for n in sorted(list(spotlight))])
        # nx.draw(subgraph)
        # plt.show()
        subgraphs.append(subgraph)
        X.append(node_features)

    return subgraphs, X

def get_subgraphs(node_ids, spotlights, level, graphs, x_base, batch, hop=False):
    """Return one spotlight subgraph per node_id
    """
    subgraphs = []
    X = []
    for node in node_ids:
        spotlight = spotlights[level][node]
        graph = graphs[batch[list(spotlight)[0]]]
        subgraph = graph.subgraph(spotlight).copy()
        node_features = np.stack([x_base[n].cpu().numpy() for n in sorted(list(spotlight))])
        # nx.draw(subgraph)
        # plt.show()
        subgraphs.append(subgraph)
        X.append(node_features)

    return subgraphs, X

def get_subgraph_edge(u, v, spotlights, level, graphs, batch, hop=False):
    """
        Get spotlight from contracting edge (u,v)
    """
    u,v = u.item(), v.item()
    spotlight_u = spotlights[level][u]
    spotlight_v = spotlights[level][v]
    spotlight_uv = spotlight_u | spotlight_v

    n = list(spotlight_uv)[0]
    graph = graphs[batch[n]]

    if hop:
        spotlight_uv = bfs_expand(graph, spotlight_uv, hops=1)

    return graph.subgraph(spotlight_uv).copy()

def expand_spotlights(spotlights, t, edge_index, k):
    """ Merge k-hop neighbhourhood spotlights.
    """
    if k < 1:
        return
    nodes = range(len(spotlights[t]))
    new_spotlights = defaultdict(set)
    for n in nodes:
        if len(edge_index[0]) == 0:
            continue
        nei = k_hop_subgraph(n, k, edge_index)
        neis = nei[0]
        new_nodes = set()
        for u in neis:
            new_nodes |= spotlights[t][u.item()]
        new_spotlights[n] = new_nodes | {n}

    for n, sp in new_spotlights.items():
        spotlights[t][n] = sp
    pass

def update_spotlights(spotlights, clusters, t):
    """ Keeps track of the spotlight of each node:

        spotlight(u^0) = u
        spotlight(u^t) = UNION(SPOTLIGHT(children(u)))

        >>> from collections import defaultdict
        >>> import torch
        >>> SL = {0: {0: {1, 2}, 1: {3, 4} }}
        >>> clusters = torch.tensor([0, 0], dtype=torch.long)
        >>> update_spotlights(SL, clusters, 1)
        >>> SL
        {0: {0: {1, 2}, 1: {3, 4}}, 1: defaultdict(<class 'set'>, {0: {1, 2, 3, 4}})}
    """

    spotlights[t] = defaultdict(set)
    for i,c in enumerate(clusters):
        spotlights[t][c.item()] |= spotlights[t-1][i]
    pass

def update_merge_graph(merge_graph, clusters, t):
    """ Keeps track of the children of each node."""
    merge_graph[t] = defaultdict(set)
    for i,c in enumerate(clusters):
        merge_graph[t][c.item()] |= {i}
    pass

def draw_one_instance(g_data, spotlight, show=False):
    G = to_networkx(g_data)
    nx.draw(G)
    if show:
        plt.show()

def ablate_graphs(graphs, method='swap', n_swaps=5):
    """ Take a batch of graphs and perform an ablation which is meant
    to be used as the 'configuration' model."""

    graphs_swap = []
    for g in graphs:
        graph_swap = g.copy().to_undirected()
        connected_double_edge_swap(graph_swap, nswap=n_swaps)
        # need to fix this to work with directed graphs
        graphs_swap.append(graph_swap.to_directed())

    return graphs_swap

def batch_to_node_indices(batch):
    """ Return node indices within each graph for a given batch.

        >>> import torch
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
        >>> batch_to_node_indices(batch)
        [0, 1, 2, 0, 1, 0]
    """
    assert bool((batch == torch.sort(batch)[0]).all()), "batch indices not sorted"
    indices = [0]
    current_batch = batch[0]
    ind = 1
    for b in batch[1:]:
        # start over if we are in a new batch
        if b != current_batch:
            ind = 0
            current_batch = b
        indices.append(ind)
        ind += 1
    assert len(indices) == len(batch)
    return indices

if __name__ == "__main__":
    import doctest
    doctest.testmod()
