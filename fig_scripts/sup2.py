import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from MotiFiesta.src.decode import HashDecoder

def do_one(decoder, top_k=1, n_runs=1, n_graphs=-1):
    jaccards = []
    for _ in range(n_runs):
        gs = decoder.decode(n_graphs=n_graphs)
        print("yo ", len(gs))
        jacc = decoder.eval(gs, top_k=top_k)
        jaccards.append(jacc)
    return np.mean(jaccards), np.std(jaccards)


reps = 5
levels = [1, 2, 3, 4, 5]
hash_dim = 32
n_graphs = 100
top_k = 3

run = 'barbell-d0.00'
dataset = 'synth-distort-barbell'

# distortions = ['0.00', '0.01', '0.02', '0.05', '0.10', '0.20']
distortions = ['0.00', '0.01', '0.02', '0.05', '0.10']

fig, ax = plt.subplots()

totals_mean = []
totals_std = []
for level in levels:
    times = []
    print(f" level {level}")
    for i in range(reps):
        print(i)
        tic = time.time()
        decoder = HashDecoder(run,
                              dataset,
                              dummy=False,
                              level=level,
                              hash_dim=hash_dim
                              )
        gs = decoder.decode()
        toc = time.time() - tic
        times.append(toc)

        # estimate subgraph size
        sizes = []
        for g in gs:
            sls = torch.unique(g.spotlight_ids)
            sls_map = {s.item():i for i, s in enumerate(sls)}
            sls_reindex = torch.zeros_like(sls)
            for i, s in enumerate(sls):
                sls_reindex[i] = sls_map[s.item()]
            sl_sizes = torch.zeros_like(sls).scatter_add(0, sls_reindex, torch.ones_like(sls_reindex))
            print(sl_sizes)

    totals_mean.append(np.mean(times))
    totals_std.append(np.std(times))

print(totals_mean)
ax.errorbar(range(len(levels)), totals_mean , yerr=totals_std, marker='o')

ax.set_xticks(range(len(levels)))
ax.set_xticklabels(levels)
plt.ylabel("Runtime (s)")
plt.xlabel("Level")
plt.show()

