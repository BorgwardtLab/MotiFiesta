import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from MotiFiesta.src.decode import HashDecoder

run = 'barbell-d0.00'
dataset = 'synth-distort-barbell-d0.00'


fig, ax = plt.subplots()

decoder = HashDecoder(run,
                      dataset,
                      dummy=False,
                      level=3
                      )

gs = decoder.decode(n_graphs=1000)
motifs_pred, true_motif_ids, sigmas = HashDecoder.collect_output(gs)

points = []
for status, sigma in zip(true_motif_ids, sigmas):
    points.append({'status': 'motif' if status.item() else 'non-motif', 'score': sigma.item()})
df = pd.DataFrame(points)

sns.boxplot(data=df, x='status', y='score')
plt.xlabel('')
plt.savefig("figs/fig1b.pdf", format="pdf")
plt.show()

