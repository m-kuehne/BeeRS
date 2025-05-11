"""
   Copyright 2025 ETH Zürich, Marino Kühne

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import cvxpy as cp
import numpy as np
import torch
from scipy.sparse import csr_array

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../algorithm"))
)

from BeersCpu import BeersCpu

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
