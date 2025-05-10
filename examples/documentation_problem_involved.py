import cvxpy as cp
import numpy as np
import torch
from scipy.sparse import csr_array

from algorithm.BeersCpu import BeersCpu

n = 1000
b = 100

# Load the data
deezer = np.load("deezer_adj_direct.npy")[:n, :n]
opinions = np.load("deezer_opinions.npy")[:n]

# Augment the data to add influencer
column_to_infl = np.full((n, 1), 1e-3)
row_from_infl = np.full((1, n + 1), 0)
deezer = np.hstack((deezer, column_to_infl))
deezer = np.vstack((deezer, row_from_infl))
opinions = np.append(opinions, 0)

# Get sparse representation of the adjacency matrix
sparse_deezer_direct = csr_array(deezer)

# Get indices of mutable data
mutable_rows = np.arange(n)
mutable_cols = np.full(n, n)

# Get the initial weights of the mutable connections
w_0 = np.zeros(mutable_rows.shape[0])


# Upper-level objective
def phi(w: torch.tensor, y: torch.tensor):
    yTy = torch.dot(y, y)
    n = y.shape[0]
    return yTy / n


# We have the constraints w >= 0 and sum w <= b
w = cp.Variable(mutable_rows.size)
constraints = [cp.sum(w) <= b, w >= 0]

# Define the problem
problem = BeersCpu(
    weights=sparse_deezer_direct,
    opinions=opinions,
    mutable_rows=mutable_rows,
    mutable_cols=mutable_cols,
    phi=phi,
    w_0=w_0,
    w=w,
    constraints=constraints,
)

problem.solve(step_size=10, tol=1e-3, momentum_parameter=0.95)
print("Min cost: ", problem.min_cost)
