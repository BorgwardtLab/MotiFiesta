"""
Generate datasets
"""
import sys

sys.path.append("/Users/cgonzalez/Projects")
print(sys.path)
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import PPI
from torch_geometric.datasets import ZINC
from torch_geometric.datasets import Reddit
from torch_geometric.data import DataLoader

from MotiFiesta_ref.utils.synthetic import SyntheticMotifs

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
enzymes = TUDataset(root='data', name='ENZYMES')
imdb = TUDataset(root='data', name='IMDB-BINARY')
cox = TUDataset(root='data', name='COX2')


# Single motif
print(">>> MOTIF TYPE")
for m_type in ['barbell', 'star', 'random', 'clique']:
    for d in [0, .01, .02, .05, .1, .2]:
        SyntheticMotifs(root='data', name=f'synth-distort-{m_type}-d{d:.2f}', seed=42, motif_size=10, motif_type=m_type, parent_size=20, distort_p=d)

# Multi-motif
print(">>> MULTI MOTIF")
for n_motifs in [3, 5, 10]:
    for d in [0, .01, .02, .05, .1, .2]:
        SyntheticMotifs(root='data', name=f'synth-{n_motifs}motifs-d{d:.2f}', seed=42, motif_size=6, parent_size=(6* n_motifs * 2), distort_p=d, n_motifs=n_motifs)

# size
print(f">>> SIZE")
for motif_size in [5, 10, 20, 30, 40]:
    for d in [0, .01, .02, .05, .1, .2]:
        SyntheticMotifs(root='data', name=f'synth-{motif_size}nodes-d{d:.2f}', motif_type='random', seed=42, motif_size=motif_size, parent_size=motif_size*2, distort_p=d)

# density
print(">>> SPARSITY")
for sparsity in [.3, .5, 1]:
    for d in [0, .01, .02, .05, .1, .2]:
        SyntheticMotifs(root='data', name=f'synth-barbell-s{sparsity:.2f}', seed=42, motif_size=10, parent_size=20, distort_p=d, motif_type='random')
