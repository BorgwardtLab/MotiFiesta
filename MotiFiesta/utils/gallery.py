import os
import pickle
import sys
import json
import uuid
from collections import defaultdict, Counter
import torch

import networkx as nx

from flask import Flask
from flask import render_template
from flask import request
from flask import flash
from flask import redirect
from flask import url_for
from flask import session
from flask import jsonify
import numpy as np

from torch_geometric.datasets import QM9
from torch_geometric.utils import to_networkx
from graph_utils import bfs_expand
from graph_utils import induced_edge_filter

# logging.basicConfig(filename='main.log',level=logging.DEBUG)

app = Flask(__name__)

def launch_server(run, data):
    global motif_tables
    global dataset
    global motif_to_level

    motif_to_level = {}
    dataset = data
    motif_tables = pickle.load(open(f"databases/{run}/motif_db.p", "rb"))

    # store which level each motif ID belongs to
    for t, table in motif_tables.items():
        for k in table.hash_tables[0].keys():
            motif_to_level[k] = t

    app.run(debug=True)

def build_graphs(motif_id):
    graphs = []
    t = motif_to_level[motif_id]
    table = motif_tables[t].hash_tables[0]
    instances = table.get_list(motif_id)
    print(f"N instances {len(instances)}")
    i = 0
    for emb, data in instances[:30]:
        # g_data = dataset.get(data['graph_index'])
        # G = to_networkx(g_data)
        # assert len(G.nodes()) == data['graph_size']
        # spotlight = data['spotlight']
        G = nx.Graph()
        G.add_nodes_from(data['nodes'])
        G.add_edges_from(data['edges'])

        nx.set_node_attributes(G, data['is_motif'], 'is_motif')

        print(G.nodes(data=True), G.edges())
        # spotlight_expand = bfs_expand(G, spotlight, hops=1)
        # G = G.subgraph(spotlight_expand).copy()
        # G = G.subgraph(spotlight).copy()


        # induced_edge_filter(G, spotlight)
        degs = Counter((G.degree(n) for n in G.nodes()))
        deg_hist = [degs[ind] for ind in range(32)]
        print(deg_hist)
        print(data['x'])

        G_js = nx.readwrite.json_graph.node_link_data(G)
        js = json.dumps(G_js)
        # gid = json.dumps(int(g_data.name[0]))
        # graphs.append({'graph': js, 'graph_id': gid})
        graphs.append({'graph': js, 'graph_id': i})
        i += 1

    return graphs

@app.route("/")
def home():
    n_steps = len(motif_tables.keys())
    motif_ids = {}
    for s in range(n_steps):
        table = motif_tables[s].hash_tables[0]
        motif_ids[s] = {k: len(table.get_list(k)) for k in table.keys()}
        motif_ids[s] = dict(sorted(motif_ids[s].items(), reverse=True, key=lambda x: x[1]))
    return render_template("home.html", motif_ids=motif_ids)


@app.route('/motif_page/<motif_id>')
def motif_page(motif_id):
    graphs = build_graphs(motif_id)
    return render_template("motif_page.html", graphs=graphs)
