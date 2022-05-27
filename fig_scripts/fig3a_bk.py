import pandas as pd
import matplotlib.pyplot as plt

from MotiFiesta.src.classify import MotiFiestaClassifierSK
from MotiFiesta.src.loading import get_loader

import warnings
warnings.filterwarnings("ignore")

points_mean = []
points_std = []


df = pd.read_csv("PROTEINS_ablate.csv")
ks = df['k']
points_mean = df['acc_mean']
points_std = df['acc_std']
ctl_mean = df['random_mean']
ctl_std = df['random_std']


fig, ax = plt.subplots()
ax.errorbar(range(len(ks)), points_mean, yerr=points_std, color='blue', marker='o', label="top k")
ax.errorbar(range(len(ks)), ctl_mean, yerr=ctl_std, color='red', marker='o', linestyle='--', label="random")

ax.set_xticks(range(len(ks)))
ax.set_xticklabels(ks)
plt.xlabel("Number of subgraphs retained")
plt.ylabel("Test accuarcy")
plt.legend()
plt.show()
