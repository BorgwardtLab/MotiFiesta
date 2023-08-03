from collections import Counter
from itertools import permutations

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

import torch
import torch.nn.functional as F
from lshashpy3 import LSHash

from MotiFiesta.utils.learning_utils import load_model
from MotiFiesta.training.loading import get_loader

class Decoder:
    def __init__(self, model_id, dataset_id):
        self.model_id = model_id
        self.dataset_id = dataset_id

        self.model = load_model(model_id)['model']
        print(self.model)
        self.dataset = get_loader(dataset_id)
        pass

    def decode(self):
        """ Assigns a motif ID vector to each node in the graph.
        Returns a Dataset object with an additional feature `motif_pred`
        """
        raise NotImplementedError
    def eval(self):
        """ Computes jaccard score for a model."""
        raise NotImplementedError

class HashDecoder(Decoder):
    def __init__(self, model_id, dataset_id, hash_dim=4, dummy=False, level=2):
        self.level = level
        self.dummy = dummy
        self.hash_dim = hash_dim

        super().__init__(model_id=model_id,
                         dataset_id=dataset_id,
                         )

    @staticmethod
    def total_sigma(level, node, tree, sigmas, ee):
        """ Recursively compute total sigma score
        for a subgraph.
        """
        children = list(tree[level][node])
        n_children = len(children)

        if n_children == 0:
            return 0
        elif n_children == 1:
            return HashDecoder.total_sigma(level-1, children[0], tree, sigmas, ee)
        else:
            eind = (ee[level-1][0] == children[0]) &\
                   (ee[level-1][1] == children[1])
            eind = eind.nonzero()[0][0].item()
            score = sigmas[level-1][eind]
            # get score for this node
            return score +\
                   HashDecoder.total_sigma(level-1, children[0], tree, sigmas, ee) +\
                   HashDecoder.total_sigma(level-1, children[1], tree, sigmas, ee)


    def decode(self, n_graphs=-1):
        # one hash table for each coarsening level
        hash_table = LSHash(self.hash_dim, self.model.hidden_dim)

        hash_to_int = Counter()
        hash_set = set()

        self.model.eval()

        nb = 0
        all_hashes = []
        all_scores = []
        all_spotlights = []

        skipped_idx = set()

        spot_count = 0
        for idx, g_pair in enumerate(self.dataset['dataset_whole']):
            if idx > n_graphs and n_graphs > -1:
                break
            g = g_pair['pos']
            g_hashes = [''] * len(g.x)

            batch = torch.zeros(len(g.x), dtype=torch.long)
            motif_scores = torch.zeros_like(batch, dtype=torch.float32)
            spotlight_ids = torch.zeros_like(batch, dtype=torch.long)

            with torch.no_grad():
                embs,probas,ee,_,merge_info,_  = self.model(g.x,
                                                          g.edge_index,
                                                          batch,
                                                          dummy=self.dummy)

            if self.level > len(embs)-1:
                skipped_idx.add(idx)
                all_scores.append(None)
                all_hashes.append(None)
                all_spotlights.append(None)
                continue
            g = to_networkx(g_pair['pos'])
            for i,x in enumerate(embs[self.level]):
                h = hash_table.index(x.detach().numpy())[0]
                # def total_sigma(self, level, node, tree, sigmas, ee):
                spotlight = list(merge_info['spotlights'][self.level][i])
                score = self.total_sigma(self.level, i, merge_info['tree'], probas, ee)
                hash_set.add(h)
                for node in spotlight:
                    motif_scores[node] = score
                    g_hashes[node] = h
                    spotlight_ids[node] = spot_count
                spot_count += 1
            all_scores.append(motif_scores)
            all_hashes.append(g_hashes)
            all_spotlights.append(spotlight_ids)

        hash_idx = {h:i+1 for i, h in enumerate(sorted(hash_set))}

        decoded_graphs = []
        for idx, g_pair in enumerate(self.dataset['dataset_whole']):
            if idx > n_graphs and n_graphs > -1:
                break
            if idx in skipped_idx:
                continue
            motif_inds = torch.tensor([hash_idx[h] for h in all_hashes[idx]])
            g_pair['pos'].motif_pred = motif_inds
            g_pair['pos'].cum_scores = all_scores[idx]
            g_pair['pos'].spotlight_ids = all_spotlights[idx]
            decoded_graphs.append(g_pair['pos'])

        return decoded_graphs


    @staticmethod
    def collect_output(decoded_graphs):
        sigma_all, motifs_pred_all, true_motif_ids = [], [], []
        for g in decoded_graphs:
            sigma_all.append(g.cum_scores)
            motifs_pred_all.append(g.motif_pred)
            true_motif_ids.append(g.motif_id)

        sigma_all = torch.cat(sigma_all)
        motifs_pred_all = torch.cat(motifs_pred_all)
        true_motif_ids = torch.cat(true_motif_ids)


        # reindex motifs from 0 to 1
        motifs_input = torch.unique(motifs_pred_all)
        motif_indices = {m.item():i for i,m in enumerate(motifs_input)}
        for i in range(len(motifs_pred_all)):
            item = motifs_pred_all[i]
            motifs_pred_all[i] = motif_indices[item.item()]

        return motifs_pred_all, true_motif_ids, sigma_all

    def motif_sigma(self, decoded_graphs):
        _, true_motif_ids, sigma_all = HashDecoder.collect_output(decoded_graphs)
        sig_mot = torch.tensor([0., 0.]).scatter_add(0, true_motif_ids, sigma_all)
        vals, counts = torch.unique(true_motif_ids, return_counts=True)
        return sig_mot / counts

    def eval(self, decoded_graphs, n_motifs=1, top_k=1):
        """ Keep top k motifs and match them to the true motif annotation usin
        permutations and jaccard.

        >>> from torch_geometric.data import Data
        >>> scores = torch.tensor([.5, .9, .9, 1])
        >>> true = torch.tensor([0, 0, 0, 1])
        >>> pred = torch.tensor([1, 1, 1, 3])
        >>> g = Data(cum_scores=scores, motif_pred=pred, motif_id=true)
        >>> eval([g])
        1.0
        """

        # collect all the graphs into one big tensor
        motifs_pred_all, true_motif_ids, sigma_all = HashDecoder.collect_output(decoded_graphs)

        # compute average sigma by motif ID
        motif_ids,counts = torch.unique(motifs_pred_all, return_counts=True)
        sigma_avg = torch.zeros_like(motif_ids, dtype=torch.float32).scatter_add(0, motifs_pred_all, sigma_all) / counts

        # rank motifs
        motifs_sorted = torch.argsort(sigma_avg, descending=True)
        ranks = torch.zeros_like(motifs_sorted)
        for ind, val in enumerate(motifs_sorted):
            ranks[val] = ind

        # print(sigma_avg)
        # print(ranks)
        # kill motifs below top_k (by setting motif id to 0), elimiate last index which is a dummy
        motifs_pred_all = torch.where(ranks[motifs_pred_all] < top_k, motifs_pred_all+1, 0)
        motif_ids = torch.unique(motifs_pred_all)

        # tensor for motif IDs: tensor[i] = 1 if assigned to motif i, all zeros if no motif assigned
        # skip the zero motif (no motif)
        # print(motifs_pred_all)
        pred = F.one_hot(motifs_pred_all)
        non_empty_mask = pred.abs().sum(dim=0).bool()

        pred = pred[:,non_empty_mask]
        true = F.one_hot(true_motif_ids)[:,1:]
        # print(pred)
        # print(true)

        # try all permutations of predicted motif IDs
        best_jaccard = 0
        for p in permutations(range(pred.shape[1])):
            # apply permutation
            p = torch.tensor(p)
            pred_perm = pred[:,p]

            # only keep as many motifs as true ones.
            pred_slice = pred_perm[:,:n_motifs]

            num = torch.min(pred_slice, true).sum(dim=0)
            den = torch.max(pred_slice, true).sum(dim=0)

            jaccard = (num / den).sum().item()

            if jaccard > best_jaccard:
                best_jaccard = jaccard
        return best_jaccard

if __name__ == "__main__":
    # import doctest
    # doctest.testmod()
    # from torch_geometric.data import Data
    # scores = torch.tensor([.5, .9, .9, 1])
    # true = torch.tensor([0, 0, 0, 1])
    # pred = torch.tensor([1, 1, 1, 3])
    # g = Data(cum_scores=scores, motif_pred=pred, motif_id=true)
    # _eval([g], top_k=2)

    decoder = HashDecoder('barbell-borg-15',
                          'synth-distort-barbell-d0.00',
                          dummy=False,
                          level=2)
    graphs = decoder.decode(n_graphs=5)
    score = decoder.eval(graphs, top_k=5)
    mot_sigma = decoder.motif_sigma(graphs)
    print(mot_sigma)
    print(score)
    pass
