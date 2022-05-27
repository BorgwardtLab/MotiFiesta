import pandas as pd

from MotiFiesta.mfinder_benchmark import eval

shapes = ['barbell', 'star', 'random', 'clique']
distortions = ['0.00', '0.01', '0.02', '0.05']

rows = []
for shape in shapes:
    for d in distortions:
        dataset = f"synth-mfinder-{shape}-d{d}"
        print(dataset)
        # j = eval(dataset)
        # rows.append({'dataset': dataset,
                     # 'd': d,
                     # 'jaccard': j
                     # }
                     # )

df = pd.DataFrame(rows)
print(df)


