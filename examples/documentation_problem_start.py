"""
   Copyright 2025 ETH Zurich, Marino Kühne

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

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../algorithm"))
)

from BeersCpu import BeersCpu

# Specify the network
matrix = np.array([[0, 1], [1, 0]])
opinions = np.array([1, 0])

# Number of users
n = 2

# The square of the frobenius norm of the initial adjacency matrix
frob_square_initial = np.sum(np.square(matrix))

# Define mutable rows and cols
mutable_rows = np.array([0, 1])
mutable_cols = np.array([1, 0])

# Get initial values of w
w_0 = np.array([1, 1])


# Upper-level objective
def phi(w: torch.tensor, y: torch.tensor):
    disagreement = 1 / 2 * (w[0] * (y[0] - y[1]) ** 2 + w[1] * (y[1] - y[0]) ** 2)
    return disagreement


delta = 0.2

# We define the constraints
w = cp.Variable(2)

constraint_non_negative = w >= 0
constraint_undirect = w[0] == w[1]
constraint_frob = cp.sum_squares(w - w_0) <= (delta**2) * frob_square_initial

# Assemble constraints
constraints = [
    constraint_non_negative,
    constraint_frob,
    constraint_undirect,
]

# Define the problem
problem = BeersCpu(
    weights=matrix,
    opinions=opinions,
    mutable_rows=mutable_rows,
    mutable_cols=mutable_cols,
    phi=phi,
    w_0=w_0,
    w=w,
    constraints=constraints,
    dense=True,
)

problem.solve()
print("Minimum cost: ", problem.min_cost)
print("Minimum weight: ", problem.min_mutable_weights)
