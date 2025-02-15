import torch
import random
import numpy as np
from itertools import islice, combinations

def qubo_dict_to_torch(nx_G, Q, torch_dtype=torch.float32, device="cpu"):
    n_nodes = len(nx_G.nodes)
    Q_mat = torch.zeros(n_nodes, n_nodes)
    for (x_coord, y_coord), val in Q.items():
        Q_mat[x_coord][y_coord] = val
    return Q_mat.type(torch_dtype).to(device)

def gen_combinations(combs, chunk_size):
    yield from iter(lambda: list(islice(combs, chunk_size)), [])

def postprocess_mis(best_bit_string, nx_graph):
    bitstring_list = list(best_bit_string)
    size_mis = sum(bitstring_list)

    ind_set = set([node for node, entry in enumerate(bitstring_list) if entry == 1])
    edge_set = set(list(nx_graph.edges))

    number_violations = 0
    for ind_set_chunk in gen_combinations(combinations(ind_set, 2), 100000):
        number_violations += len(set(ind_set_chunk).intersection(edge_set))

    return size_mis, ind_set, number_violations

def postprocess_coloring(best_string, num_color, nx_graph):
    string_list = list(best_string)

    color_set_list = [set([node for node, entry in enumerate(string_list) if entry == i]) for i in range(num_color)]
    edge_set = set(list(nx_graph.edges))

    number_violations = 0
    for color_set in color_set_list:
        for color_set_chunk in gen_combinations(combinations(color_set, 2), 100000):
            number_violations += len(set(color_set_chunk).intersection(edge_set))

    return number_violations

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
