import sys
import pickle
import argparse

import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)

from model import MotiFiesta
from loading import get_loader
from train import motif_train
from learning_utils import dump_model_hparams
from learning_utils import load_model
from learning_utils import load_data
from learning_utils import make_dirs


FUNCTIONS = ['train', 'test', 'build', 'browse', 'eval']

try:
    function = sys.argv[1]
except IndexError:
    raise ValueError("Specify a valid function please")


print(f"== {function} mode ==")

# train the motif pooling model
if function == 'train':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", type=str, default="default")
    parser.add_argument("--dataset", "-da", type=str, default='synthetic')
    parser.add_argument("--restart", "-r", action='store_true', default=False,  help='Restart model.')

    # training  loop
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    parser.add_argument("--max-batches", "-m", type=int, default=-1)
    parser.add_argument("--epochs", "-e", type=int, default=200)
    parser.add_argument("--stop-epochs", type=int, default=30, help="Number of epochs to train embeddings before doing motifs.")
    parser.add_argument("--attributed", default=False, action='store_true', help="Use attributed graphs. If False, use node degree as features.")

    # learning
    parser.add_argument("--lam",  type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--estimator", "-es", type=str, default='knn', help="Which density estimator to use for the motif agg. step")
    parser.add_argument("--n-neighbors", "-nn", type=int, default=30, help="Number of neighbors")
    parser.add_argument("--volume", "-vv", action='store_true', default=False, help="Use d-sphere volume to normalize density with kNN.")

    # model architecture
    parser.add_argument("--steps", "-s", type=int, default=5)
    parser.add_argument("--dim", "-d", type=int, default=8)
    parser.add_argument("--pool-dummy", action="store_true", default=False)
    parser.add_argument("--score-method", default="sigmoid", help="Edge soring method: (sigmoid, softmax-neighbors, softmax-all)")
    parser.add_argument("--merge-method", default="sum", help="Edge merge method: (sum, cat, set2set)")
    parser.add_argument("--hard-embed", action='store_true', help="Whether to use hard embedding using degree histogram. ")

    args, _ = parser.parse_known_args()
    print(args)

    make_dirs(args.name)

    hparams = {'model':{
                        'dim': args.dim,
                        'steps': args.steps,
                        'pool_dummy': args.pool_dummy,
                        'edge_score_method': args.score_method,
                        'merge_method': args.merge_method,
                        'hard_embed': args.hard_embed,
                         },
               'train': {
                        'epochs': args.epochs,
                        'batch_size': args.batch_size,
                        'max_batches': args.max_batches,
                        'dataset': args.dataset,
                        'lambda': args.lam,
                        'beta': args.beta,
                        'stop_epochs': args.stop_epochs,
                        'estimator': args.estimator,
                        'k': args.n_neighbors,
                        'volume': args.volume,
                        'attributed': args.attributed,
                        }
                }

    print(">>> loading data")
    data = get_loader(
                      root=hparams['train']['dataset'],
                      batch_size=hparams['train']['batch_size'],
                      attributed=hparams['train']['attributed']
                      )

    hparams['model']['n_features'] = data['dataset_whole'].num_features
    print(data['dataset_whole'].num_features)
    dump_model_hparams(args.name, hparams)

    print(">>> building model")

    if args.restart:
        print(f"Restarting training with ID: {args.name}")
        model_dict = load_model(args.name)
        model = model_dict['model']
        epoch_start = model_dict['epoch']
        optimizer = model_dict['optimizer']
        controller_state = model_dict['controller_state_dict']
    else:
        model = MotiFiesta(**hparams['model'])
        epoch_start, optimizer, controller_state = 0, None, None
    print(model)

    print(">>> training...")
    motif_train(model,
                train_loader=data['loader_train'],
                test_loader=data['loader_test'],
                model_name=args.name,
                epochs=args.epochs,
                max_batches=args.max_batches,
                stop_epochs=args.stop_epochs,
                estimator=args.estimator,
                volume=args.volume,
                n_neighbors=args.n_neighbors,
                hard_embed=args.hard_embed,
                beta=args.beta,
                lam=args.lam,
                epoch_start=epoch_start,
                optimizer=optimizer,
                controller_state=controller_state
                )

