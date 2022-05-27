import math

import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

from MotiFiesta.utils.synthetic import SyntheticMotifs
from MotiFiesta.utils.real_world import RealWorldDataset


def get_loader(root,
               batch_size=2,
               **kwargs
               ):
    if not root.lower().startswith('synth'):
        print(f"TU dataset {root}")
        if root == 'IMDB-BINARY':
            print("HELLO")
            dataset = RealWorldDataset(root=root, max_degree=300, n_features=301)
        else:
            dataset = RealWorldDataset(root=root)
    else:
        dataset = SyntheticMotifs(root=root, **kwargs)
    lengths = [math.floor(len(dataset) * .8), math.ceil(len(dataset) * .2)]
    train_data, test_data = random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
    loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return {'dataset_whole': dataset, 'loader_whole': loader, 'loader_train': loader_train, 'loader_test': loader_test}


if __name__ == "__main__":
    data = get_loader(root='barbell-pair')
    print(data)
