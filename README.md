# ⚡ Quasi-Quantum Annealing (QQA)

A **PyTorch implementation** of the [ICLR2025](https://iclr.cc/) paper:
**[Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling](https://openreview.net/forum?id=9EfBeXaXf0)**.

---
## Abstract
Learning-based methods have gained attention as general-purpose solvers due to their ability to automatically learn problem-specific heuristics, reducing the need for manually crafted heuristics. However, these methods often face scalability challenges. To address these issues, the improved Sampling algorithm for Combinatorial Optimization (iSCO), using discrete Langevin dynamics, has been proposed, demonstrating better performance than several learning-based solvers. This study proposes a different approach that integrates gradient-based update through continuous relaxation, combined with **Q**uasi-**Q**uantum **A**nnealing (**QQA**). QQA smoothly transitions the objective function, starting from a simple convex function, minimized at half-integral values, to the original objective function, where the relaxed variables are minimized only in the discrete space. Furthermore, we incorporate parallel run communication leveraging GPUs to enhance exploration capabilities and accelerate convergence. Numerical experiments demonstrate that our method is a competitive general-purpose solver, achieving performance comparable to iSCO and learning-based solvers across various benchmark problems. Notably, our method exhibits superior speed-quality trade-offs for large-scale instances compared to iSCO, learning-based solvers, commercial solvers, and specialized algorithms.

## Demo: Annealing Process
<p align="center">
  <img src="data/fig/demo.gif" width="180px">
</p>

---

## Installation

We recommend using **Python 3.9+**. Install the required packages via:

```bash
pip install -r requirements.txt
```

### **Required Packages & Versions**
- `torch==2.5.1`
- `numpy==1.26.4`
- `matplotlib==3.9.4`
- `networkx==3.2.1`
- `tqdm==4.67.1`
- `scipy==1.13.1`

---

## 🚀 Usage Guide

Below is an example of how to use QQA for a **Maximum Independent Set** (MIS) problem. We define a custom problem class, generate a random regular graph, and run `qqa.batch_annealing` with specified hyperparameters. 

### **Step 1: Define a Custom Problem Class**

```python
# Example: ProblemClass (Maximum Independent Set)
class MaximumIndependentSet(COProblem):
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
```

### **Step 2: Construct a Problem Instance**

```python
import random
import torch
import numpy as np

# Fix seed for reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Graph Parameters
N, d, p = 50, 3, None
nx_graph = nx.random_regular_graph(d=d, n=N, seed=SEED)

# Create the MIS problem instance
problem = MaximumIndependentSet(nx_graph, penalty=2, device=device)
```

### **Step 3: Run Quasi-Quantum Annealing**

```python
from tqdm import tqdm
import qqa  # Assume qqa.py (or a QQA module) is provided

# QQA Parameters
cfg = {
    'sol_size': 100,
    'learning_rate': 1.0,
    'temp': 0.001,
    'min_bg': -3,
    'max_bg': 0.1,
    'curve_rate': 4,
    'div_param': 0.2
}

best_sol, best_obj, runtime = qqa.batch_annealing(
    problem,
    sol_size=cfg['sol_size'],
    learning_rate=cfg['learning_rate'],
    temp=cfg['temp'],
    min_bg=cfg['min_bg'],
    max_bg=cfg['max_bg'],
    curve_rate=cfg['curve_rate'],
    div_param=cfg['div_param'],
    num_epochs=int(3e3),
    check_interval=500,
    device=device,
    plot_dynamics=True
)

print("Best Solution:", best_sol)
print("Best Objective Value:", best_obj.item())
print("Runtime (sec):", runtime)
```

---

## 📚 Citation

If you use our approach in your work, please cite:

```bibtex
@inproceedings{
ichikawa2025optimization,
title={Optimization by Parallel Quasi-Quantum Annealing with Gradient-Based Sampling},
author={Yuma Ichikawa and Yamato Arai},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=9EfBeXaXf0}
}
```

---

## 📜 License
This project is licensed under the **BSD 3-Clause License**. See [LICENSE](LICENSE.txt) for details.

---


### 🎉 Enjoy Exploring Quasi-Quantum Annealing!  
Feel free to raise issues or submit pull requests for further improvements. Happy Researching and Optimizing!
