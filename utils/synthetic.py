"""
    Generate graphs containing synthetic motifs.
    The idea:
        1. A motif is a random graph
        2. For each instance, insert motif graph into larger random graph.
        4. Insertion means sampling a random node from parent graph and replacing it with motif.
        5. Links between motif and parent graph are made by sampling random nodes in both subgraphs with same probability
           as the edge probability which generated the graphs.
        6. Repeat for each motif.
"""
import os
import os.path as osp
import random
import itertools

import numpy as np
from numpy.random import normal
from scipy.stats import beta
from scipy.stats import uniform
from scipy.stats import expon
import networkx as nx
from networkx.generators.random_graphs import connected_watts_strogatz_graph
from networkx.generators.random_graphs import powerlaw_cluster_graph
from networkx.generators.random_graphs import extended_barabasi_albert_graph
from networkx.generators.random_graphs import erdos_renyi_graph
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Dataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.utils import to_networkx

random.seed(0)
np.random.seed(0)

def rewire(g, n_iter=100):
    """ Apply (u, v), (u', v') --> (u, v'), (v, u') to randomize graph.
    """
    rewired_g = g.copy()
    for n in range(n_iter):
        e1, e2 = random.sample(list(g.edges()), 2)
        rewired_g.remove_edges_from([e1, e2])
        rewired_g.add_edges_from([(e1[0], e2[1]), (e1[1], e2[0])])

    rewired_g.remove_edges_from(list(nx.selfloop_edges(rewired_g)))
    return rewired_g


def motif_distort(g, motif_nodes, p=0.1, n_classes=2):
    """ Distort motif subgraph according to probability p
        We iterate over all nodes of edges and with probability p
        we remove the edge if it is present, or create it if it is absent.
        Returns new motif graph.
    """
    motif_new = g.copy()
    node_pairs = itertools.combinations(motif_nodes, 2)
    add_edges = []
    remove_edges = []
    for u, v in node_pairs:
        if p > random.random():
            if (u, v) in g.edges():
                remove_edges.append((u, v))
            else:
                add_edges.append((u, v))
    motif_new.remove_edges_from(remove_edges)
    motif_new.add_edges_from(add_edges)

    for n in g.nodes():
        if p > random.random():
            motif_new.nodes[n]['class'] = random.randint(0, n_classes-1)
        else:
            motif_new.nodes[n]['class'] = g.nodes[n]['class']

    #fig, ax = plt.subplots(1, 2)
    #nx.draw(g, ax=ax[0])
    #nx.draw(motif_new, ax=ax[1])
    #plt.show()
    return motif_new

def generate_parent_erdos(n, p, max_degree=25, seed=0):
    # if not seed is None:
        # np.random.seed(seed)
    offset = 0
    while True:
        g = erdos_renyi_graph(n, p, seed=seed+offset)
        connected = nx.is_connected(g)
        deg = max((g.degree(n) for n in g.nodes())) < max_degree
        if deg and connected:
            return g
        else:
            offset+=1

def generate_parent(n, seed=None, show=False, sparse_factor=1):
    if not seed is None:
        np.random.seed(seed)
        random.seed(seed)

    generators = ['erdos', 'power', 'barbasi', 'watts']
    generator = random.choice(generators)

    if generator == 'erdos':
        p = beta(1.3, (1.3 * n) / np.log2(n) - 1.3).rvs()
        if show:
            print(f"erdos: n={n} p={p}")
        g = erdos_renyi_graph(n, p/sparse_factor, seed=seed)
    if generator == 'power':
        m = int(uniform(1, 2 * np.log2(n)).rvs())
        p = uniform(0, .5).rvs()
        if show:
            print(f"power: m={m} p={p}")
        g = powerlaw_cluster_graph(n, max(1, m//sparse_factor), p/sparse_factor, seed=seed)
    if generator == 'barbasi':
        m = int(uniform(1, 2 * np.log2(n)).rvs())
        p = min(expon(20).rvs(), .2)
        q = min(expon(20).rvs(), .2)

        if show:
            print(f"barbasi : n={n} m={m} p={p} q={q}")
        g = extended_barabasi_albert_graph(n, max(1, m//sparse_factor), p/sparse_factor, q/sparse_factor, seed=seed)
    if generator == 'watts':
        k = int(max(2, n * beta(1.3, 1.3 * ((n/ np.log2(n)) - 1.3)).rvs()))
        p = beta(2, 2).rvs()

        if show:
            print(f"watts: k={k} p={p}")
        g = connected_watts_strogatz_graph(n, k, p/sparse_factor, seed=seed)
    return g

def motif_embed(motifs,
                parent_size=20,
                parent_e_prob=.05,
                distort_p=0.0, embed_prob=1,
                n_classes=2,
                ):
    """
        Embeds motifs into a larger parent graph.
        Create random larger graph G.
        Pick random node from G and replace with motif instance m.
        Reconnect a random sample of nodes in m to nodes in G.
    """
    parent_graph = generate_parent_erdos(parent_size, p=parent_e_prob)

    for n in parent_graph.nodes():
        parent_graph.nodes[n]['is_motif'] = 0
        parent_graph.nodes[n]['motif_id'] = 0
        parent_graph.nodes[n]['class'] = random.randint(0, n_classes-1)

    original = parent_graph.copy()

    anchor_nodes = sorted(random.sample(parent_graph.nodes(), len(motifs)))

    offset = 0
    ind = 0
    for motif_id, motif in motifs.items():
        # embed only if passes concentration probability
        if random.random() > embed_prob:
            continue
        N = len(motif.nodes())

        motif = motif_distort(motif, motif.nodes(), p=distort_p)
        #pick a random node in the parent graph to substitute
        sub_node = anchor_nodes[ind] + offset
        parent_graph.remove_node(sub_node)

        motif_node_ids = [i + sub_node for i in motif.nodes()]

        # relabel motif nodes to start from chosen parent node
        motif = nx.relabel_nodes(motif,
                                 {i:i+sub_node for i in motif.nodes()}
                                 )

        parent_graph = nx.relabel_nodes(parent_graph,
                                        {i:i if i < sub_node else i+N for i in parent_graph.nodes()}
                                        )

        link_nodes = [n for n in motif.nodes() if random.random() < parent_e_prob]
        parent_links = [n for n in parent_graph.nodes() if random.random() < parent_e_prob]

        # add edges between motif and parent graph
        # we can maybe adjust this by choosing nodes at the border of the motif to not
        # mess up the motif too much.
        parent_graph.add_nodes_from(motif.nodes(data=True))
        parent_graph.add_edges_from(motif.edges())
        parent_graph.add_edges_from(list(zip(parent_links, link_nodes)))

        for n in motif.nodes():
            parent_graph.nodes[n]['is_motif'] = 1
            parent_graph.nodes[n]['motif_id'] = motif_id

        offset += N
        ind += 1

    parent_graph_fresh = nx.Graph()
    parent_graph_fresh.add_nodes_from(sorted(parent_graph.nodes(data=True)))
    parent_graph_fresh.add_edges_from(sorted(parent_graph.edges(data=True)))
    parent_graph = parent_graph_fresh
    # generate randomized graph
    randomized = rewire(parent_graph)

    nx.set_node_attributes(randomized, 0, 'is_motif')
    nx.set_node_attributes(randomized, 0, 'motif_id')

    nx.set_node_attributes(original, 0, 'is_motif')
    nx.set_node_attributes(original, 0, 'motif_id')

    nx.set_node_attributes(original, {n:int(original.degree(n)) for n in original.nodes()}, 'deg')
    nx.set_node_attributes(parent_graph, {n:int(parent_graph.degree(n)) for n in parent_graph.nodes()}, 'deg')
    nx.set_node_attributes(randomized, {n:int(randomized.degree(n)) for n in randomized.nodes()}, 'deg')

    return parent_graph, original, randomized

def draw_motif(graph):
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=20, node_color=['blue' if d['is_motif'] else 'grey' for n,d in graph.nodes(data=True)])
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    motif_edges = [(u, v) for u,v in graph.edges() if graph.nodes[u]['is_motif'] and graph.nodes[v]['is_motif']]
    nx.draw_networkx_edges(graph, pos, width=3, edgelist=motif_edges)
    plt.show()

def generate_instances(
                       n_motifs=1,
                       n_graphs=1000,
                       edge_prob=.2,
                       draw_instance=False,
                       motif_type=None,
                       motif_size=10,
                       distort_p=-1.,
                       parent_e_prob=.05,
                       parent_size=10,
                       concentration=1,
                       n_classes=2,
                       attributed=True,
                       max_degree=25.,
                       seed=0
                       ):
    """
        Generates dataset of synthetic motifs and yields one instance
        at a time.

        Arguments:
            num_motifs (int): number of motifs to use (default=100)
            num_instances (tuple): mean and std to sample for each motif which determines the number of instances we generate
            edge_prob (tuple): mean and std to sample for the edge creation probability of each motif.
            draw_instance (bool): draw motif after it has been inserted in the parent graph.
        Returns:
            list: 2D list with list of motif instances for each motif.  """
    # create motifs

    motif_menu = {'star': nx.star_graph(motif_size),
                  'barbell': nx.barbell_graph(motif_size//2, 2),
                  'wheel': nx.wheel_graph(motif_size),
                  'random': generate_parent_erdos(motif_size, p=parent_e_prob, seed=seed),
                  'clique': nx.complete_graph(motif_size),
                  'lollipop': nx.lollipop_graph(motif_size//2, motif_size)
                  }

    if not motif_type is None:
        motifs = {motif_type: motif_menu[motif_type]}
    else:
        motifs = {n: generate_parent_erdos(motif_size, p=parent_e_prob, seed=seed+ n) for n in range(n_motifs)}

    motif_ids = {m:i+1 for i, m in enumerate(sorted(motifs.keys()))}

    # add node labels to motifs
    for _,m in motifs.items():
        for n in m.nodes():
            m.nodes[n]['class'] = random.randint(0, n_classes-1)

    def graph_accept(g):
        deg = max((g.degree(n) for n in g.nodes())) < max_degree
        connected = nx.is_connected(g)
        return deg and connected

    pygs = []
    for i in range(n_graphs):
        tries = 0
        while True:
            to_plant = {motif_ids[m]:motifs[m] for m in motifs.keys()}
            graphs = motif_embed(to_plant,
                                 parent_size=parent_size,
                                 distort_p=distort_p,
                                 parent_e_prob=parent_e_prob,
                                 embed_prob=concentration
                                 )

            if sum(map(graph_accept,graphs)) == 3:
                break
            else:
                tries += 1

        planted, original, wired = graphs
        node_attrs = ['class'] if attributed else ['deg']

        original_pyg = from_networkx(original, group_node_attrs=node_attrs)
        motif_pyg = from_networkx(planted, group_node_attrs=node_attrs)
        random_pyg = from_networkx(wired, group_node_attrs=node_attrs)

        if attributed:
            original_pyg.x = F.one_hot(original_pyg.x.squeeze(), num_classes=n_classes).float()
            motif_pyg.x = F.one_hot(motif_pyg.x.squeeze(), num_classes=n_classes).float()
            random_pyg.x = F.one_hot(random_pyg.x.squeeze(), num_classes=n_classes).float()
        else:
            original_pyg.x = F.one_hot(original_pyg.x.squeeze(), num_classes=max_degree).float()
            motif_pyg.x = F.one_hot(motif_pyg.x.squeeze(), num_classes=max_degree).float()
            random_pyg.x = F.one_hot(random_pyg.x.squeeze(), num_classes=max_degree).float()

        original_pyg.graph_idx = i
        motif_pyg.graph_idx = i
        random_pyg.graph_idx = i

        pygs.append({'pos': motif_pyg, 'neg': original_pyg, 'rand': random_pyg})
        print(f"{i+1} of {n_graphs}", end='\r')

    return pygs

class SyntheticMotifs(Dataset):
    def __init__(self,
                 n_motifs=1,
                 root="data",
                 name='synth',
                 n_instances=1000,
                 n_graphs=1000,
                 motif_size=5,
                 motif_type='star',
                 distort_p=-.1,
                 transform=None,
                 parent_e_prob=0.10,
                 parent_size=20,
                 pre_transform=None,
                 deg_bins=5,
                 attributed=False,
                 n_classes=2,
                 max_degree=25,
                 seed=0):
        """ Builds the synthetic motif dataset. Motifs are built on the
        fly and stored to disk.

        Args:
        ---
        root (str): path to folder where graphs will be stores.
        n_graphs (int): number of graphs to generate
        n_motifs (int): number of motifs to inject in graphs
        """
        self.n_motifs = n_motifs
        self.n_graphs = n_graphs
        self.n_instances = n_instances
        self.motif_size = motif_size
        self.motif_type = motif_type
        self.parent_e_prob = parent_e_prob
        self.parent_size = parent_size
        self.distort_p = distort_p
        self.n_classes = n_classes
        self.deg_bins = deg_bins
        self.attributed = attributed
        self.max_degree = max_degree
        self.seed = seed

        super(SyntheticMotifs, self).__init__(osp.join(root, name), transform, pre_transform)


    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.n_graphs)]

    @property
    def num_features(self):
        print("is attributed: ", self.attributed)
        return self.n_classes if self.attributed else self.max_degree

    def process(self):
        print(f">> generating {self.motif_type} synthetic graphs")
        print(self.__dict__)
        gs = generate_instances(
                                n_motifs=self.n_motifs,
                                motif_type=self.motif_type,
                                motif_size=self.motif_size,
                                parent_e_prob=self.parent_e_prob,
                                parent_size=self.parent_size,
                                distort_p=self.distort_p,
                                n_classes=self.n_classes,
                                attributed=self.attributed,
                                max_degree=self.max_degree,
                                seed=self.seed
                                )

        assert len(gs) == self.n_graphs

        for i, g_pyg in enumerate(gs):
            torch.save(g_pyg, osp.join(self.processed_dir, f'data_{i}.pt'))

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, idx):
        """ Returns dictionary where 'pos' key stores batch with
        graphs that contain the motif, and the 'neg' key has batches
        without the motif.
        """
        if idx > len(self) - 1:
            raise StopIteration
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

if __name__ == "__main__":
    # d = SyntheticMotifs(motif_type='barbell', distort_p=.02, motif_size=8, root="barbell-pair-s")
    d = SyntheticMotifs(motif_type='random', parent_e_prob=.1, attributed=False, parent_size=100, distort_p=.15, motif_size=10, root="hhot-randos", seed=1)

    # l = DataLoader(d, batch_size=1)
    # for batch in l:
        # print(batch)
        # print(batch['pos'].edge_index)
        # print(batch['pos'].x)
