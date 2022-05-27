"""
Generate datasets
"""
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import PPI
from torch_geometric.datasets import ZINC
from torch_geometric.datasets import Reddit
from torch_geometric.data import DataLoader

from MotiFiesta.utils.synthetic import SyntheticMotifs

### OPTIONS


"""
n_motifs: number of motifs to plant in the same graph
motif_type: motif type (if not given, random)
motif_size: number of nodes in motif
motif_density: probability of planting a motif
parent_density: edge probability for parent graph to motif graph
parent_size: number of nodes in parent graph
distort_p: probability of swapping edges in motif graph
"""

proteins = TUDataset(root='data', name='PROTEINS')
imdb = TUDataset(root='data', name='IMDB-BINARY')
cox = TUDataset(root='data', name='COX2')

print(">>> MOTIF TYPE")
for m_type in ['barbell', 'star', 'random', 'clique']:
    for d in [0, .01, .02, .05, .1, .2]:
        SyntheticMotifs(root=f'synth-mfinder-{m_type}-d{d:.2f}', seed=42, motif_size=4, motif_type=m_type,
                        parent_size=8, distort_p=d)

# Since mfinder does not work on motifs of size 6, we just do a smaller test case.
# print(">>> MOTIF TYPE")
# for m_type in ['barbell', 'star', 'random', 'clique']:
#     for d in [0, .01, .02, .05, .1, .2]:
#         SyntheticMotifs(root=f'synth-mfinder-{m_type}-d{d:.2f}', seed=42, motif_size=6, motif_type=m_type, parent_size=12, distort_p=d)
#
# # Multi-motif
# print(">>> MULTI MOTIF")
# for n_motifs in [3, 5, 10]:
#     for d in [0, .01, .02, .05, .1, .2]:
#         SyntheticMotifs(root=f'synth-mfinder-{n_motifs}motifs-d{d:.2f}', seed=42, motif_size=6,
#                         parent_size=(6 * n_motifs * 2), distort_p=d, n_motifs=n_motifs)
#
# # size
# print(f">>> SIZE")
# for motif_size in [5, 10, 20, 30, 40]:
#     for d in [0, .01, .02, .05, .1, .2]:
#         SyntheticMotifs(root=f'synth-mfinder-{motif_size}nodes-d{d:.2f}', motif_type='random', seed=42, motif_size=motif_size, parent_size=motif_size*2, distort_p=d)
