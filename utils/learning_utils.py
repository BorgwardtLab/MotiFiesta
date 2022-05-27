import os
import json

import torch

device_cache = None
def get_device():
    global device_cache
    if device_cache is None:
        device_cache = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
        #device_cache = torch.device("cpu")
    return device_cache

def load_data(run, batch_size=2, background_only=False):
    with open(f'models/{run}/hparams.json', 'r') as j:
        json_params = json.load(j)
    data = dataset_from_json(json_params)
    return data

def load_model(run, permissive=False, verbose=True):
    """
    Input the name of a run
    :param run:
    :return:
    """
    with open(f'models/{run}/hparams.json', 'r') as j:
        json_params = json.load(j)

    model = model_from_json(json_params)

    try:
        model_dict = torch.load(f'models/{run}/{run}.pth',
                                map_location='cpu')
        state_dict = model_dict['model_state_dict']
        model.load_state_dict(state_dict)

        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_dict['optimizer_state_dict'])

    except FileNotFoundError:
        if not permissive:
            raise FileNotFoundError('There are no weights for this experiment...')
    return {'model': model,
            'epoch': model_dict['epoch'],
            'optimizer':optimizer,
            # 'controller_state_dict': model_dict['controller_state_dict']
            }

def dump_model_hparams(name, hparams):
    with open(f'models/{name}/hparams.json', 'w') as j:
        json.dump(hparams, j)
    pass

def model_from_json(params):
    from MotiFiesta.src.model import MotiFiesta
    model = MotiFiesta(**params['model'])
    return model

def dataset_from_json(params, background=False):
    from MotiFiesta.src.loading import get_loader
    data = get_loader(root=params['train']['dataset'],\
                                batch_size=params['train']['batch_size']\
                                )
    return data

def make_dirs(run):
    try:
        os.mkdir(f"models/{run}")
    except FileExistsError:
        pass

def one_hot_to_id(x):
    """ Create column vector with index where one-hot is 1. """
    return torch.nonzero(x, as_tuple=True)[1]
