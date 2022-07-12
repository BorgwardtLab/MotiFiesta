http://img.shields.io/badge/cs.LG-arXiv%3A2206.01008-B31B1B.svg

# MotiFiesta: Neural Approximate Motif Mining

## Setup

```
$ pip install -r requirements.txt
$ source setup.sh
```

## Build datasets

```
$ python utils/build_data.py
```

## Training a model

```
$ python src/motifiesta.py train -h
$ python src/motifiesta.py train -da <dataset_id> -n test
```

## Making motif predictions


```python
from MotiFiesta.src.decode import HashDecoder

model_id = 'barbell-d0.00'
data_id = 'synth-distort-barbell-d0.00'
level = 3

decoder = HashDecoder(model_id, data_id, level)

decoded_graphs = decoder.decode()

for graph in decoded_graphs:
	print(f"Motif assignment for each node: {g.motif_pred}")
```

Scripts for generating figures in the paper are in `fig_scripts/`

Output from running [mfinder](https://www.weizmann.ac.il/mcb/UriAlon/sites/mcb.UriAlon/files/uploads/NetworkMotifsSW/mfinder/mfindermanual.pdf) are in `data_mfinder` and `out_mfinder`, the script `minder_benchmark.py` parses this output.


Citation

```
@article{oliver2022approximate,
  title={Approximate Network Motif Mining Via Graph Learning},
  author={Oliver, Carlos and Chen, Dexiong and Mallet, Vincent and Philippopoulos, Pericles and Borgwardt, Karsten},
  journal={arXiv preprint arXiv:2206.01008},
  year={2022}
}
```
