import torch
import os
import numpy as np
import networkx as nx
import pickle
from abc import ABC, abstractmethod

# Define base directory for data (relative to this file)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')


class COProblem(ABC):
    @abstractmethod
    def loss_fn(self, x):
        pass

class QUBOProblem(ABC):
    @abstractmethod
    def generate_qubo_matrix(self):
        pass

    @abstractmethod
    def loss_fn(self, x):
        pass

class MaximumIndependentSet(QUBOProblem):
    def __init__(self, nx_graph, penalty=3, device="cpu"):
        super().__init__()
        self.nx_graph = nx_graph
        self.penalty = penalty
        self.device = device
        self.num_nodes = nx_graph.number_of_nodes()
        self.Q_mat = self.generate_qubo_matrix()

    def generate_qubo_matrix(self):
        Q = torch.full((self.num_nodes, self.num_nodes), 0.0)
        for (u, v) in self.nx_graph.edges:
            Q[u][v] = self.penalty
            Q[v][u] = self.penalty
        for u in self.nx_graph.nodes:
            Q[u][u] = -1
        return Q.to(self.device)

    def loss_fn(self, x):
        return torch.einsum('bi,ij,bj->b', x, self.Q_mat, x)

class MaximumIndependentSetInstance:
    def __init__(self, nx_graph_list, max_node, penalty=3, device="cpu"):
        Q_list = []
        for nx_graph in nx_graph_list:
            Q = torch.full((max_node, max_node), 0.0)
            for (u, v) in nx_graph.edges:
                Q[u][v] = penalty
                Q[v][u] = penalty
            for u in nx_graph.nodes:
                Q[u][u] = -1
            Q_list.append(Q)
        self.Q_tensor = torch.stack(Q_list).to(device)
        self.num_instance = len(nx_graph_list)
        self.max_node = max_node

    def loss_fn(self, x):
        return torch.einsum('bci,cij,bcj->bc', x, self.Q_tensor, x)

class MaxClique(QUBOProblem):
    def __init__(self, nx_graph, penalty=3, device="cpu"):
        super().__init__()
        self.nx_graph = nx_graph
        self.penalty = penalty
        self.device = device
        self.num_nodes = nx_graph.number_of_nodes()
        self.Q_mat = self.generate_qubo_matrix()

    def generate_qubo_matrix(self):
        Q = torch.full((self.num_nodes, self.num_nodes), self.penalty)
        for (u, v) in self.nx_graph.edges:
            Q[u][v] = 0
            Q[v][u] = 0
        for u in self.nx_graph.nodes:
            Q[u][u] = -1
        return Q.to(self.device)

    def loss_fn(self, x):
        return torch.einsum('bi,ij,bj->b', x, self.Q_mat, x)

class MaxCliqueInstance:
    def __init__(self, nx_graph_list, max_node, penalty=3, device="cpu"):
        Q_list = []
        for nx_graph in nx_graph_list:
            Q = torch.full((max_node, max_node), penalty)
            for (u, v) in nx_graph.edges:
                Q[u][v] = 0
                Q[v][u] = 0
            for u in nx_graph.nodes:
                Q[u][u] = -1
            Q_list.append(Q)
        self.Q_tensor = torch.stack(Q_list).to(device)
        self.num_instance = len(nx_graph_list)
        self.max_node = max_node

    def loss_fn(self, x):
        return torch.einsum('bci,cij,bcj->bc', x, self.Q_tensor, x)

class MaxCut(QUBOProblem):
    def __init__(self, nx_graph, device="cpu"):
        super().__init__()
        self.nx_graph = nx_graph
        self.device = device
        self.num_nodes = nx_graph.number_of_nodes()
        self.Q_mat = self.generate_qubo_matrix()

    def generate_qubo_matrix(self):
        Q = torch.full((self.num_nodes, self.num_nodes), 0.0)
        for (u, v, data) in self.nx_graph.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            Q[u][v] = w
            Q[v][u] = w
        wsum = Q.sum(dim=1)
        for u in self.nx_graph.nodes:
            Q[u][u] = -wsum[u].item()
        return Q.to(self.device)

    def loss_fn(self, x):
        return torch.einsum('bi,ij,bj->b', x, self.Q_mat, x)

class MaxCutInstance:
    def __init__(self, nx_graph_list, max_node, device="cpu"):
        Q_list = []
        for nx_graph in nx_graph_list:
            Q = torch.full((max_node, max_node), 0.0)
            for (u, v, data) in nx_graph.edges(data=True):
                w = data['weight'] if 'weight' in data else 1.0
                Q[u][v] = w
                Q[v][u] = w
            wsum = Q.sum(dim=1)
            for u in nx_graph.nodes:
                Q[u][u] = -wsum[u].item()
            Q_list.append(Q)
        self.Q_tensor = torch.stack(Q_list).to(device)
        self.num_instance = len(nx_graph_list)
        self.max_node = max_node

    def loss_fn(self, x):
        return torch.einsum('bci,cij,bcj->bc', x, self.Q_tensor, x)

class BalancedGraphPartition(QUBOProblem):
    def __init__(self, nx_graph, num_category=3, device='cpu', penalty=0.0005):
        super().__init__()
        self.nx_graph = nx_graph
        self.adj = torch.tensor(nx.adjacency_matrix(nx_graph).toarray(),
                                device=device, dtype=torch.float32)
        self.num_node = nx_graph.number_of_nodes()
        self.num_edge = nx_graph.number_of_edges()
        self.num_category = num_category
        self.penalty = penalty
        self.device = device
        self.Q_mat = self.generate_qubo_matrix()

    def generate_qubo_matrix(self):
        return None

    def loss_fn(self, x):
        edge_cut = self.num_edge - torch.sum(torch.einsum('bis,ij,bjs->bs', x, self.adj, x) / 2, dim=1)
        bal = torch.sum((self.num_node / self.num_category - torch.sum(x, dim=1)) ** 2, dim=1)
        return edge_cut + bal * self.penalty

    def cut_ratio(self, x):
        return (self.num_edge - torch.sum(torch.einsum('bis,ij,bjs->bs', x, self.adj, x) / 2, dim=1)) / self.num_edge

    def balanceness(self, x):
        return 1 - torch.mean((1 - torch.sum(x, dim=1) / (self.num_node / self.num_category)) ** 2, dim=1)

    def eval(self, x):
        return [self.cut_ratio(x).item(), self.balanceness(x).item()]

class Coloring(QUBOProblem):
    def __init__(self, nx_graph, num_category=3, device="cpu"):
        super().__init__()
        self.nx_graph = nx_graph
        self.adj = torch.tensor(nx.adjacency_matrix(nx_graph).toarray(),
                                device=device, dtype=torch.float32)
        self.num_node = nx_graph.number_of_nodes()
        self.num_edge = nx_graph.number_of_edges()
        self.num_category = num_category
        self.device = device
        self.Q_mat = self.generate_qubo_matrix()

    def generate_qubo_matrix(self):
        return None

    def loss_fn(self, x):
        return torch.sum(torch.einsum('bis,ij,bjs->bs', x, self.adj, x) / 2, dim=1)

    def eval(self, x):
        return [self.loss_fn(x).item()]

# Load Data
def mis_er_small(penalty=3, problem_type='list', device='cpu'):
    # Load small ER graphs from ../data/mis/er_test
    path = os.path.join(DATA_DIR, 'mis', 'er_test')
    files = os.listdir(path)
    graphs = []
    for f in files:
        fp = os.path.join(path, f)
        with open(fp, 'rb') as fh:
            g = pickle.load(fh)
        graphs.append(g)
    if problem_type == 'all':
        return MaximumIndependentSetInstance(graphs, 800, penalty=penalty, device=device)
    else:
        return [MaximumIndependentSet(g, penalty=penalty, device=device) for g in graphs]

def mis_er_large(penalty=3, problem_type='list', device='cpu'):
    # Load large ER graphs from ../data/mis/er_large_test
    path = os.path.join(DATA_DIR, 'mis', 'er_large_test')
    files = os.listdir(path)
    graphs = []
    for f in files:
        fp = os.path.join(path, f)
        with open(fp, 'rb') as fh:
            g = pickle.load(fh)
        graphs.append(g)
    if problem_type == 'all':
        return MaximumIndependentSetInstance(graphs, 10915, penalty=penalty, device=device)
    else:
        return [MaximumIndependentSet(g, penalty=penalty, device=device) for g in graphs]

def mis_sat(group='all', problem_type='list', device='cpu'):
    # Load SAT-based MIS graphs from ../data/mis/SAT_graphs_ver2
    path = os.path.join(DATA_DIR, 'mis', 'SAT_graphs_ver2')
    files = os.listdir(path)
    graphs = []
    if group == 'first':
        subset = files[:len(files)//2]
    elif group == 'second':
        subset = files[len(files)//2:]
    else:
        subset = files
    for f in subset:
        fp = os.path.join(path, f)
        mat = np.load(fp)
        g = nx.from_numpy_array(mat)
        graphs.append(g)
    if problem_type == 'all':
        return MaximumIndependentSetInstance(graphs, 1347, penalty=1, device=device)
    else:
        return [MaximumIndependentSet(g, penalty=1, device=device) for g in graphs]

def mcq_twitter(problem_type='list', device='cpu'):
    # Load data from ../data/maxclique/twitter (adjust if needed)
    path = os.path.join(DATA_DIR, 'maxclique', 'twitter')
    files = os.listdir(path)
    graphs = []
    for f in files:
        fp = os.path.join(path, f)
        with open(fp, 'rb') as fh:
            data = pickle.load(fh)
        for g in data[0]:
            graphs.append(g)
    if problem_type == 'all':
        return MaxCliqueInstance(graphs, 247, device=device)
    else:
        return [MaxClique(g, device=device) for g in graphs]

def mcq_RB(group='all', problem_type='list', device='cpu'):
    # Load data from ../data/maxclique/RB_test (adjust if needed)
    path = os.path.join(DATA_DIR, 'maxclique', 'RB_test')
    files = os.listdir(path)
    graphs = []
    if group == 'first':
        subset = files[:len(files)//2]
    elif group == 'second':
        subset = files[len(files)//2:]
    else:
        subset = files
    for f in subset:
        fp = os.path.join(path, f)
        with open(fp, 'rb') as fh:
            data = pickle.load(fh)
        for g in data:
            graphs.append(g[1])
    if problem_type == 'all':
        return MaxCliqueInstance(graphs, 475, device=device)
    else:
        return [MaxClique(g, device=device) for g in graphs]

def mct_ba(case, problem_type='list', device='cpu'):
    # Load from ../data/maxcut/maxcut-ba (adjust if needed)
    base = os.path.join(DATA_DIR, 'maxcut', 'maxcut-ba')
    max_nodes_list = [1100, 150, 20, 300, 40, 600, 75]
    all_folders = os.listdir(base)
    path = os.path.join(base, all_folders[case])
    files = os.listdir(path)
    graphs = []
    for f in files:
        fp = os.path.join(path, f)
        with open(fp, 'rb') as fh:
            data = pickle.load(fh)
        graphs.append(data[0])
    if problem_type == 'all':
        return MaxCutInstance(graphs, max_nodes_list[case], device=device)
    else:
        return [MaxCut(g, device=device) for g in graphs]

def mct_er(case, problem_type='list', device='cpu'):
    # Load from ../data/maxcut/maxcut-er (adjust if needed)
    base = os.path.join(DATA_DIR, 'maxcut', 'maxcut-er')
    max_nodes_list = [1100, 150, 20, 300, 40, 600, 75]
    all_folders = os.listdir(base)
    path = os.path.join(base, all_folders[case])
    files = os.listdir(path)
    graphs = []
    for f in files:
        fp = os.path.join(path, f)
        with open(fp, 'rb') as fh:
            data = pickle.load(fh)
        graphs.append(data[0])
    if problem_type == 'all':
        return MaxCutInstance(graphs, max_nodes_list[case], device=device)
    else:
        return [MaxCut(g, device=device) for g in graphs]

def mct_opt(problem_type='list', device='cpu'):
    # Load from ../data/maxcut/optsicom (adjust if needed)
    path = os.path.join(DATA_DIR, 'maxcut', 'optsicom')
    files = os.listdir(path)
    graphs = []
    for f in files:
        fp = os.path.join(path, f)
        with open(fp, 'rb') as fh:
            data = pickle.load(fh)
        graphs.append(data[0])
    if problem_type == 'all':
        return MaxCutInstance(graphs, 125, device=device)
    else:
        return [MaxCut(g, device=device) for g in graphs]

def balanced_graph_partition(file_path, device='cpu', penalty=0.0005):
    # Load a single graph pickle
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    return BalancedGraphPartition(graph, device=device, penalty=penalty)

