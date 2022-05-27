import os
import multiprocessing

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from MotiFiesta.src.decode import HashDecoder

import warnings
warnings.filterwarnings("ignore")

NUM_WORKERS = 4

def do_one(decoder, n_motifs=1, n_runs=1, top_k=1, n_graphs=-1):
    jaccards = []
    for _ in range(n_runs):
        gs = decoder.decode(n_graphs=n_graphs)
        jacc = decoder.eval(gs, n_motifs=n_motifs, top_k=top_k)
        jaccards.append(jacc)
    return np.mean(jaccards), np.std(jaccards)

def eval_one(args):
    dataset  = args['dataset']
    run = args['run']
    params = args['params']
    d = args['d']

    n_motifs = 1
    if 'motifs' in dataset:
        n_motifs = int(dataset.split('-')[1][0])
        print(f"Switching to {n_motifs} motifs for {dataset}")

    decoder = HashDecoder(run,
                          f'{dataset}-d{d}',
                          **params
                          )
    j_mean, j_std = do_one(decoder,
                           n_runs=args['n_runs'],
                           top_k=args['top_k'],
                           n_graphs=args['n_graphs'],
                           n_motifs=n_motifs
                           )
    return {'jaccard_mean': j_mean,
            'jaccard_std': j_std,
            'hash_dim': params['hash_dim'],
            'level': params['level'],
            'd': d,
            'dataset': f'{dataset}-d{d}',
            'dummy': params['dummy']
            }

def launch(dump_path):

    datasets = {'synth-distort-barbell': 'barbell-d0.00',
                'synth-distort-star': 'star-d0.00',
                'synth-distort-random': 'random-d0.00',
                'synth-distort-clique': 'clique-d0.00',
                'synth-3motifs': 'synth-3-motifs',
                'synth-5motifs': 'synth-5-motifs',
                'synth-5nodes': 'synth-5-size',
                'synth-10nodes': 'synth-10-size',
                'synth-20nodes': 'synth-20-size',
                'synth-barbell-s0.30': 'sparsity-.3',
                'synth-barbell-s0.50': 'sparsity-.5',
                'synth-barbell-s1.00': 'sparsity-1'
                }

    # distortions = ['0.00', '0.01', '0.02', '0.05', '0.10', '0.20']
    distortions = ['0.00', '0.01', '0.02', '0.05']

    decode_params = {'hash_dim': [8, 16, 32],
                     'level': [2, 3, 4],
                     'dummy': [True, False]
                     }

    param_grid = ParameterGrid(decode_params)

    data_final = []

    todo = []
    for dataset, run in datasets.items():
        for para in param_grid:
            for d in distortions:
                todo.append({'dataset': dataset,
                             'run': run,
                             'params': para,
                             'top_k': 3,
                             'n_graphs': 200,
                             'n_runs': 3,
                             'd': d
                             })

    pool = multiprocessing.Pool(NUM_WORKERS)

    # open previous results dataframe to skip done jobs
    try:
        df_done = pd.read_csv(dump_path)
    except FileNotFoundError:
        pass
    else:
        todo_clean = []
        for job in todo:
            hit = df_done.loc[(df_done['dataset'] == f"{job['dataset']}-d{job['d']}") &\
                              (df_done['hash_dim'] == job['params']['hash_dim']) &\
                              (df_done['level'] == job['params']['level']) &\
                              (df_done['dummy'] == job['params']['dummy'])]
            if len(hit) == 0:
                todo_clean.append(job)
            else:
                print(f"skipping {job}")

        todo = todo_clean

    print(f">>> {len(todo)} jobs")
    n_done = 0
    for result in pool.imap_unordered(eval_one, todo):
        print(f"done {n_done} of {len(todo)} jobs.")
        df = pd.DataFrame([result])
        # df.to_csv("jaccards.csv")
        df.to_csv(dump_path, mode='a', header=not os.path.exists(dump_path))
        n_done += 1

    # df.to_csv("jaccards.csv")
    # df = df.pivot(index='dataset', columns='distortion', values='jaccard')
    # print(df.to_latex(escape=False, bold_rows=True))

def process_results(results_path):
    df = pd.read_csv(results_path)
    non_dummy = df.loc[df['dummy'] == False]
    best = df.iloc[non_dummy.groupby(['dataset'])['jaccard_mean'].idxmax()]
    best['root'] = [d[:-6] for d in best['dataset']]
    table = []
    for name, dataset in best.groupby('root'):
        for row in dataset.itertuples():
            dummy = df.loc[(df['dataset'] == row.dataset) &\
                           (df['hash_dim'] == row.hash_dim) &\
                           (df['level'] == row.level) &\
                           (df['dummy'] == True)
                           ]
            if len(dummy) == 0:
                dummy_mean = 0
                dummy_std = 0
            else:
                dummy_mean = dummy['jaccard_mean'].iloc[0]
                dummy_std = dummy['jaccard_std'].iloc[0]

            j = f"{row.jaccard_mean:.2f} $\pm$ {row.jaccard_std:.2f}" + " {\small " + f"({dummy_mean:.2f} $\pm$ {dummy_std:.2f})" +"}"
            print(j)
            table.append({'dataset': name,
                          'jaccard':  j,
                          'd': row.d
                          }
                         )
    table = pd.DataFrame(table)
    table  = table.pivot(index=['dataset'], columns=['d'], values=['jaccard'])
    print(table.to_latex(escape=False, bold_rows=True))

if __name__ == "__main__":
    # launch("jaccards_fast2.csv")
    # process_results('jaccards_fast2.csv')
    process_results('jaccards_fast_slurm.csv')
