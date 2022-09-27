"""
Get table of sp miner results
"""
import os
import pickle
from itertools import permutations

import pandas as pd
import torch

from MotifPool.src.loading import get_loader

SP_PATH = os.path.join(os.path.dirname(__file__),
                       "..",
                       "neural-subgraph-learning-GNN",
                       "results"
                       )

def jaccard(run_id, dataset_id, top_k=5, n_motifs=1):
    out_path = os.path.join(SP_PATH,run_id +".p")
    graphs = pickle.load(open(out_path, 'rb'))
    dataset = get_loader(dataset_id)['dataset_whole']
    # dict with 'patterns', 'pattern_hashes', 'hashes_to_graph'
    h = graphs['pattern_hashes'][0]
    g = graphs['hashes_to_graphs'][h][0]
    g_idx_to_ind = dict()
    N_nodes = 0

    for i,g in enumerate(dataset):
        g = g['pos']
        n = len(g.x)
        g_idx_to_ind[i] = (N_nodes, N_nodes + n)
        N_nodes += n

    # build \hat{Y} matrix for all the graphs in the dataset 
    Y_hat = torch.zeros((N_nodes, top_k))
    for p, pattern_hash in enumerate(graphs['pattern_hashes'][:top_k]):
        for instance_nx in graphs['hashes_to_graphs'][pattern_hash]:
            start,_ = g_idx_to_ind[instance_nx.graph['graph_idx']]
            instance_inds = [n + start for n in instance_nx.nodes()]
            Y_hat[:,p][instance_inds] = 1.

    # build ground truth Y
    true_motifs = []
    for g in dataset:
        true_motifs.append(g['pos'].motif_id)
    Y = torch.cat(true_motifs).reshape(-1, 1)

    best_jaccard = 0
    for p in permutations(range(Y_hat.shape[1])):
        # apply permutation
        p = torch.tensor(p)
        pred_perm = Y_hat[:,p]

        # only keep as many motifs as true ones.
        pred_slice = pred_perm[:,:n_motifs]

        num = torch.min(pred_slice, Y).sum(dim=0)
        den = torch.max(pred_slice, Y).sum(dim=0)

        jaccard = (num / den).sum().item()

        if jaccard > best_jaccard:
            best_jaccard = jaccard

    return best_jaccard

def do_all():
    rows = []
    for m_type in ['barbell', 'star', 'clique', 'random']:
        for d in ['0.00', '0.01', '0.02', '0.05']:
            did = f"synth-distort-{m_type}-d{d}"
            j = jaccard(did, did)
            print(m_type, d, j)
            rows.append({'jaccard': j, 'd': d, 'dataset': m_type})
    df = pd.DataFrame(rows)
    table  = df.pivot(index=['dataset'], columns=['d'], values=['jaccard'])
    print(table.to_latex(escape=False, float_format="%.3f", bold_rows=True))
    pass

if __name__ == "__main__":
    do_all()
    pass
