import pickle
import argparse
import multiprocessing as mp
import numpy as np
from numpy import inf
import networkx as nx
from func_timeout import func_timeout, FunctionTimedOut



# def edge_match(d_g1, d_g2):
    # return d_g1['label'] == d_g2['label']

edge_match = None

# from subgraph_matching.config import parse_encoder
# from subgraph_matching.train import build_model
# from subgraph_matching.alignment import gen_alignment_matrix

### NEURAL STUFF
def load_model():
    args = pickle.load(open('neural_args.p', 'rb'))

    model = build_model(args)
    return model

def neural_subg_score(model, query, target):
    A = gen_alignment_matrix(model,
                             query.to_undirected(),
                             target.to_undirected())
    # some entries are -inf need to look into this
    # for now replace with 0
    A[A == -inf] = 0
    return np.mean(A)

#########

def make_line_graph(g, node_attr=None, edge_attr=None):
    lg = nx.line_graph(g)

    d = {}
    for u, v in g.edges():
        if node_attr is None and edge_attr is None:
            label = 1
        elif not (node_attr is None and edge_attr is None):
            label = (
                     g.nodes[u][node_attr],
                     g.edges[(u,v)][edge_attr],
                     g.nodes[v][node_attr]
                     )
        elif not edge_attr is None:
            label = (g.edges[(u,v)][edge_attr])
        else:
            label = (g.nodes[u][node_attr], g.nodes[v][node_attr])
        d[(u,v)] = label
    nx.set_node_attributes(lg, values=d, name='label')
    return lg

def has_subgraphs(g1, g2, node_attr=None, edge_attr=None):
    """G2 is included in G1 (all occurrences)"""
    g1_l = make_line_graph(g1, node_attr=node_attr, edge_attr=edge_attr)
    g2_l = make_line_graph(g2, node_attr=node_attr, edge_attr=edge_attr)

    M = nx.isomorphism.DiGraphMatcher(g1_l, g2_l, node_match=edge_match)
    return len(list(M.subgraph_isomorphisms_iter()))

def has_subgraph(g1, g2, node_attr=None, edge_attr=None, timeout=None):
    """G2 is included in G1"""
    g1_l = make_line_graph(g1, node_attr=node_attr, edge_attr=edge_attr)
    g2_l = make_line_graph(g2, node_attr=node_attr, edge_attr=edge_attr)
    M = nx.isomorphism.DiGraphMatcher(g1_l, g2_l, node_match=edge_match)

    return M.subgraph_is_isomorphic()

def subgraph_freq(subgraph,
                  graphs,
                  embedding=None,
                  norm='log',
                  mode='neural',
                  node_attr=None,
                  edge_attr=None,
                  timeout=None,
                  kde=False,
                  density_model=None
                  ):
    """ Estimate the density of subgraph in graphs."""
    if kde:
        return density_model.score_samples(embedding)[0]
    model = load_model()
    count = 0
    success = 0
    # import matplotlib.pyplot as plt
    for g in graphs:
        if mode == 'neural':
            count += neural_subg_score(model, subgraph, g)
            success += 1
        else:
            try:
                is_subg = func_timeout(3,
                                        has_subgraph,
                                        args=(g, subgraph),
                                        kwargs={'node_attr': node_attr,
                                                'edge_attr': edge_attr}
                                    )
            except FunctionTimedOut:
                continue
            else:
                count += is_subg
                success += 1

    if success < 3:
        raise ValueError

    N = success
    raw = count / (N + .001)

    # we want larger samples to have higher score
    if norm == 'log':
        return raw ** (1 / (np.log(N) + 1))
    if norm == 'sqrt':
        return raw ** (1 / np.sqrt(N) / 5)
    if norm == 'raw':
        return raw

