# -*- coding: utf-8 -*-

import torch
import random
import os
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

from itertools import islice, combinations
from time import time
from tqdm import tqdm
import pickle

# Define base directory for data (relative to this file)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

def generate_graph(n, d=None, p=None, graph_type='reg', random_seed=0):
    """random graph nx objectを生成"""
    if graph_type == 'reg':
        print(f'Generating d-regular graph with n={n}, d={d}, seed={random_seed}')
        nx_random_graph = nx.random_regular_graph(d=d, n=n, seed=random_seed)
    elif graph_type == 'prob':
        print(f'Generating p-probabilistic graph with n={n}, p={p}, seed={random_seed}')
        nx_random_graph = nx.fast_gnp_random_graph(n, p, seed=random_seed)
    elif graph_type == 'erdos':
        print(f'Generating erdos-renyi graph with n={n}, p={p}, seed={random_seed}')
        nx_random_graph = nx.erdos_renyi_graph(n, p, seed=random_seed)
    else:
        raise NotImplementedError(f'!! Graph type {graph_type} not handled !!')
    return nx_random_graph


def gen_combinations(combs, chunk_size):
    yield from iter(lambda: list(islice(combs, chunk_size)), [])

def is_symmetric(matrix):
    if matrix.size(0) != matrix.size(1):
        return False
    diff = torch.abs(matrix - matrix.t())
    return torch.max(diff).item() < 1e-6

def run_mis_solver(nx_graph):
    t_start = time()
    ind_set = nx.algorithms.approximation.clique.maximum_independent_set(nx_graph)
    t_solve = time() - t_start
    ind_set_size = len(ind_set)
    nx_bitstring = [1 if (node in ind_set) else 0 for node in sorted(list(nx_graph.nodes))]
    edge_set = set(list(nx_graph.edges))
    number_violations = 0
    for ind_set_chunk in gen_combinations(combinations(ind_set, 2), 100000):
        number_violations += len(set(ind_set_chunk).intersection(edge_set))
    return nx_bitstring, ind_set_size, number_violations, t_solve

def postprocess_gnn_mis(best_bit_string, nx_graph):
    bitstring_list = list(best_bit_string)
    size_mis = sum(bitstring_list)
    ind_set = set([node for node, entry in enumerate(bitstring_list) if entry == 1])
    edge_set = set(list(nx_graph.edges))
    number_violations = 0
    for ind_set_chunk in gen_combinations(combinations(ind_set, 2), 100000):
        number_violations += len(set(ind_set_chunk).intersection(edge_set))
    return size_mis, ind_set, number_violations

def postprocess_gnn_max_cut(best_bit_string, nx_graph):
    bitstring_list = list(best_bit_string)
    S0 = [node for node in nx_graph.nodes if not bitstring_list[node]]
    S1 = [node for node in nx_graph.nodes if bitstring_list[node]]
    cut_edges = [(u, v) for u, v in nx_graph.edges if bitstring_list[u] != bitstring_list[v]]
    uncut_edges = [(u, v) for u, v in nx_graph.edges if bitstring_list[u] == bitstring_list[v]]
    size_max_cut = len(cut_edges)
    return size_max_cut, [S0, S1], cut_edges, uncut_edges

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_best_Twitter():
    data_path = "/proj/dsc/user_work/arai/test/cra/data/maxclique/twitter"
    files = os.listdir(data_path)
    best_list = []
    for file in files:
        file_path = os.path.join(data_path, file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        for best in data[1]:
            best_list.append(best)
    return np.array(best_list)

def get_best_RB():
    data_path = "/proj/dsc/user_work/arai/test/cra/data/maxclique/RB_test"
    files = os.listdir(data_path)
    best_list = []
    for file in files:
        file_path = os.path.join(data_path, file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        for best in data:
            best_list.append(best[0])
    return np.array(best_list)

def get_best_ba(case):
    all_data_path = "/proj/dsc/user_work/arai/test/cra/data/maxcut/maxcut-ba"
    data_path_list = os.listdir(all_data_path)
    data_path = os.path.join(all_data_path, data_path_list[case])
    files = os.listdir(data_path)
    best_list = []
    for file in files:
        file_path = os.path.join(data_path, file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        best_list.append(data[1][0])
    return np.array(best_list)

import torch
import random
import os
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

from itertools import islice, combinations
from time import time
from tqdm import tqdm
import pickle

# Define base directory for data (relative to this file)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

def get_best_Twitter():
    data_path = os.path.join(DATA_DIR, 'maxclique', 'twitter')
    files = os.listdir(data_path)
    best_list = []
    for file in files:
        file_path = os.path.join(data_path, file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        for best in data[1]:
            best_list.append(best)
    return np.array(best_list)

def get_best_RB():
    data_path = os.path.join(DATA_DIR, 'maxclique', 'RB_test')
    files = os.listdir(data_path)
    best_list = []
    for file in files:
        file_path = os.path.join(data_path, file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        for best in data:
            best_list.append(best[0])
    return np.array(best_list)

def get_best_ba(case):
    all_data_path = os.path.join(DATA_DIR, 'maxcut', 'maxcut-ba')
    data_path_list = os.listdir(all_data_path)
    data_path = os.path.join(all_data_path, data_path_list[case])
    files = os.listdir(data_path)
    best_list = []
    for file in files:
        file_path = os.path.join(data_path, file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        best_list.append(data[1][0])
    return np.array(best_list)

def get_best_er(case):
    all_data_path = os.path.join(DATA_DIR, 'maxcut', 'maxcut-er')
    data_path_list = os.listdir(all_data_path)
    data_path = os.path.join(all_data_path, data_path_list[case])
    files = os.listdir(data_path)
    best_list = []
    for file in files:
        file_path = os.path.join(data_path, file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        best_list.append(data[1][0])
    return np.array(best_list)

def get_best_opt():
    data_path = os.path.join(DATA_DIR, 'maxcut', 'optsicom')
    files = os.listdir(data_path)
    best_list = []
    for file in files:
        file_path = os.path.join(data_path, file)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        best_list.append(data[1][0])
    return np.array(best_list)
