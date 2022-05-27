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

def draw_hierarchy(g_pyg, e_inds, merge_info, edge_scores, steps=3, vpad=1.):
    g_nx = to_networkx(g_pyg)
    graph_nodes = set(g_nx.nodes())
    motif_nodes = {n.item() for n in torch.where(g_pyg.is_motif)[0]}

    big_g = nx.Graph()
    big_pos = {}

    score_data = []

    for level in range(steps):
        merges = merge_info['tree'][level]
        motif_nodes = []
        non_motif_nodes = []
        # merge {layer: {new_node : {children}, .. } }
        g_t_pyg = Data(edge_index=e_inds[level])
        g_t_nx = to_networkx(g_t_pyg)
        for n in g_t_nx.nodes():
            sl = merge_info['spotlights'][level][n]
            sl_in_motif = set([n for n in sl if g_pyg.is_motif[n].item()])
            is_motif = len(sl_in_motif) / len(sl) > .6
            big_g.add_node((level, n), is_motif=is_motif)

            if sl_in_motif:
                motif_nodes.append(n)
            else:
                non_motif_nodes.append(n)


        for eind, uv in enumerate(e_inds[level].T):
            u, v = uv[0].item(), uv[1].item()
            e_score = edge_scores[level][eind].item()
            big_g.add_edge((level, u), (level, v), score=e_score, merge=False)

            if u in motif_nodes and v in motif_nodes:
                score_data.append({'score': e_score, 'status': 'motif', 'level':level})
            else:
                score_data.append({'score': e_score, 'status': 'non-motif', 'level': level})


        if level == 0:
            pos = nx.spring_layout(g_t_nx)
            for node, coord in pos.items():
                big_pos[(level, node)] = coord
            continue
        # connect to children
        else:
            for node in g_t_nx.nodes():
                children_coords = np.stack([big_pos[(level-1, c)] for c in merges[node]])
                center = np.mean(children_coords, axis=0)
                center[1] += vpad
                big_pos[(level, node)] = center
                for child in merges[node]:
                    big_g.add_edge((level, node), (level-1, child), score=np.nan, merge=True)


    cm = matplotlib.cm.get_cmap('Blues')
    for level in range(steps):
        motif_nodes = [n for n,d in big_g.nodes(data=True) if d['is_motif'] == 1 and n[0] == level]

        nodes = nx.draw_networkx_nodes(big_g, big_pos, nodelist=motif_nodes,
                                        node_color=[level]*len(motif_nodes),
                                        cmap=cm,
                                        vmin=0,
                                        vmax=steps,
                                        node_shape="*",
                                        node_size=100
                                        )

        nodelist = [n for n in big_g.nodes() if n[0] == level and n not in motif_nodes]
        nodes.set_edgecolor('black')

        nodes = nx.draw_networkx_nodes(big_g, big_pos, nodelist=nodelist,
                                       node_color=[level]*len(nodelist),
                                       cmap=cm,
                                       vmin=0,
                                       node_size=50,
                                       vmax=steps)
        nodes.set_edgecolor('black')
    real_edges = [(u,v) for u,v,d in big_g.edges(data=True) if not d['merge']]
    edgewidths = [big_g.edges[e]['score'] for e in real_edges]
    merge_edges = [(u,v) for u,v,d in big_g.edges(data=True) if d['merge']]
    g_edges = nx.draw_networkx_edges(big_g, big_pos, edgelist=real_edges, width=edgewidths)
    m_edges = nx.draw_networkx_edges(big_g, big_pos, edgelist=merge_edges, style='dashed')
    plt.show()

run = 'barbell-d0.00'
model = load_model(run)['model']
data = get_loader('synth-distort-barbell-d0.00')
for idx in range(len(data['dataset_whole'])):
    g_pair = data['dataset_whole'][idx]
    batch = torch.zeros(len(g_pair['pos'].x), dtype=torch.long)
    embs,probas,ee,_,merge_info,_ = model(g_pair['pos'].x,
                                            g_pair['pos'].edge_index,
                                            batch,
                                            dummy=False)
    draw_hierarchy(g_pair['pos'], ee, merge_info, probas, steps=model.steps)
