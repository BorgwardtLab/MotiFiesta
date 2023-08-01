"""
Convert a dataset to format for MFinder
"""
import sys
import os
import numpy as np

# sys.path.append('../..')
sys.path.append('..')
from MotiFiesta.src.loading import get_loader

path = os.path.abspath(os.path.dirname(__file__))


def convert(name, overwrite=False):
    """
    Take a dataset in the form of a list of .pt files and turn into one large .txt file node1 node2 weight file
    the node1 node2 use a cumulative offset that grows with number of nodes in each graph
    Also dump a node_map such that outmapping[node_txt] = [graph_number, node_number]
    """
    dataset = get_loader(name)
    lines = []
    offset = 0
    outname = os.path.join(path, '..', f'data_mfinder/{name}.txt')
    outmapping_path = os.path.join(path, '..', f'data_mfinder/{name}.npy')
    outmapping = []
    if os.path.exists(outname) and os.path.exists(outmapping_path) and not overwrite:
        return
    # Stop after computing the outmapping if the data is already present
    if os.path.exists(outname) and not overwrite:
        compute_txt = False
    else:
        compute_txt = True
        outfile = open(outname, 'w')
    for i, g in enumerate(dataset['dataset_whole']):
        g = g['pos']
        outmapping.extend([[i, j] for j in range(g.num_nodes)])

        if compute_txt:
            e_ind = g.edge_index + offset
            done_edges = set()
            for pair in e_ind.t():
                u, v = tuple(pair.numpy())

                e_set = frozenset([u, v])
                if e_set in done_edges:
                    continue
                else:
                    done_edges.add(e_set)

                line = f"{u} {v} 1\n"
                outfile.write(line)
        offset += g.num_nodes
    if compute_txt:
        outfile.close()
    outmapping = np.asarray(outmapping)
    np.save(outmapping_path, outmapping)


def process_all(overwrite=False):
    data_path = os.path.join(path, '..', 'data')
    for dataset_name in os.listdir(data_path):
        convert(dataset_name, overwrite=overwrite)


if __name__ == "__main__":
    pass
    # convert('synth-mfinder-barbell-d0.00', overwrite=True)
    # convert('PROTEINS', overwrite=True)
    # convert(sys.argv[1])
    process_all(overwrite=False)
