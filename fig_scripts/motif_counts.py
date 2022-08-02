import pandas as pd
from networkx.algorithms.clique import enumerate_all_cliques
from networkx.algorithms import isomorphism
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

from MotiFiesta.src.loading import get_loader
from MotiFiesta.utils.synthetic import SyntheticMotifs

def count_cliques(graphs, size=4):
    n_cliques = 0
    for g in graphs:
        cliques = enumerate_all_cliques(g.to_undirected())
        for c in cliques:
            if len(c) == size:
                n_cliques += 1
            if len(c) > size:
                break
    return n_cliques

def subg_matches(graphs, motif):
    """ Count the number of subgraphs of g isomorphic to motif
    """
    matches = 0
    for g in graphs:
        m = isomorphism.GraphMatcher(g, motif)
        matches += len(list(m.subgraph_isomorphisms_iter()))
    return matches

# m_type = 'clique'
# rows = []
# for motif_size in [4, 5, 6, 8, 10]:
    # for d in [0, .01, .02, .05]:
        # da = SyntheticMotifs(root=f'count-synth-distort-{m_type}-d{d:.2f}', seed=42, motif_size=motif_size, motif_type=m_type, parent_size=2*motif_size, distort_p=d)
        # graphs = [to_networkx(g['pos']) for g in da]
        # n_cliques = count_cliques(graphs, size=motif_size)
        # row = {"$\epsilon$": d, 'clique_size': motif_size, 'n_cliques': n_cliques}
        # print(row)
        # rows.append(row)

# df = pd.DataFrame(rows)
# table  = df.pivot(index=['clique_size'], columns=["$\epsilon$"], values=['n_cliques'])
# print(table.to_latex(escape=False, bold_rows=True))
# print(table.to_markdown())
# df.to_csv("cliques.csv")

for m_type in ['barbell', 'random', 'star']:
    rows = []
    for motif_size in [5, 8, 10]:
        for d in [0, .01, .02, .05, .1, .2]:
            da = SyntheticMotifs(root=f'synth-distort-{m_type}-d{d:.2f}-{motif_size}', seed=42, motif_size=motif_size, motif_type=m_type, parent_size=2*motif_size, distort_p=d)
            graphs = [to_networkx(g['pos'], node_attrs=['is_motif']) for g in da]
            g = graphs[0]
            motif = g.subgraph([n for n,d  in g.nodes(data=True) if d['is_motif']]).copy()
            n_matches = subg_matches(graphs, motif)
            row = {"$\epsilon$": d, 'motif size': motif_size, 'instances': n_matches}
            print(row)
            rows.append(row)


    df = pd.DataFrame(rows)
    table  = df.pivot(index=['motif size'], columns=["$\epsilon$"], values=['instances'])
    print(table.to_latex(escape=False, bold_rows=True))
    print(table.to_markdown())
