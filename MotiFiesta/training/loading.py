import os
import math

import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

from MotiFiesta.utils.synthetic import SyntheticMotifs
from MotiFiesta.utils.real_world import RealWorldDataset


def get_loader(root,
               name='synthetic',
               batch_size=2,
               **kwargs
               ):
    """
    Arguments
    ----------
    root: 
        path to folder for storing the dataset
    name: 
        ID of dataset (options: 'synthetic' generates synthetic motifs, else the string ID of a PyG dataset

    Returns
    -------
    
    dict:
        Dictionary with loaders and datasets for train/test 

    """
    if not name.startswith('synth'):
        if name == 'IMDB-BINARY':
            dataset = RealWorldDataset(root=root, max_degree=300, n_features=301)
        else:
            dataset = RealWorldDataset(root=root)
    else:
        dataset = SyntheticMotifs(root=root, name=name, **kwargs)
    lengths = [math.floor(len(dataset) * .8), math.ceil(len(dataset) * .2)]
    train_data, test_data = random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
    loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return {'dataset_whole': dataset, 'loader_whole': loader, 'loader_train': loader_train, 'loader_test': loader_test}


if __name__ == "__main__":
    data = get_loader(root='barbell-pair')
    print(data)
