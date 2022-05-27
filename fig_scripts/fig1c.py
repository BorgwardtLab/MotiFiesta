"""
Heatmap of train vs test dataset accuracy.
"""
import itertools

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from decode import HashDecoder

def do_one(decoder, n_runs=1, n_graphs=-1):
    jaccards = []
    for _ in range(n_runs):
        gs = decoder.decode(n_graphs=n_graphs)
        jacc = decoder.eval(gs)
        jaccards.append(jacc)
    return np.mean(jaccards), np.std(jaccards)


reps = 1
n_graphs = -1

# one run per dataset (i.e. train on each dataset)
run_base = 'barbell-borg-15'

dataset = 'synth-distort-barbell'

# distortions = ['0.00', '0.01', '0.02', '0.05', '0.10', '0.20']
distortions = ['0.00', '0.01', '0.02']
distortion_pairs = itertools.combinations(distortions, 2)

runs = [f"{run_base}-d{d}" for d in distortions]


fig, ax = plt.subplots()

jaccards = []

for train, test in distortion_pairs:
    decoder = HashDecoder(f"{run_base}-d{train}",
                          f'{dataset}-d{test}',
                          dummy=False
                          )
    jaccards.append(do_one(decoder, n_runs=reps, n_graphs=n_graphs))

jaccards = np.array(jaccards)
jaccards = jaccards.reshape((len(distortions), len(distortions)))
sns.heatmap(jaccards)
plt.show()

