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
import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy.sparse import csr_array

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../algorithm"))
)

from BeersGpu import BeersGpu
from BeersCpu import BeersCpu

# We test if Beers finds the correct solution
# We brute-force a simple example and compare the solution to Beers


# Define upper-level optimization function
def phi(x: torch.tensor, y: torch.tensor):
    return y.t() @ y / 3


@jax.jit
def phi_gpu(x: jnp.array, y: jnp.array):
    return jnp.dot(y, y) / 3


dense_adj_matrix = np.array([[0, 1.4, 0], [1, 0, 1.1], [1, 0, 0]])
sparse_adj_matrix = csr_array(dense_adj_matrix)
opinions = np.array([-1, 1, 0.5])
mutable_rows = np.array([0, 1])
mutable_cols = np.array([1, 2])
w_0 = np.array([1.4, 1.1])

# Define cvxpy constraints and variables
z = cp.Variable(2)
constraints = [z >= 0, z <= 2]

# Brute-force problem
x = np.linspace(0, 2, num=100)
y = np.linspace(0, 2, num=100)

# Create the grid
xx, yy = np.meshgrid(x, y)

min_pol = None
min_x = None
min_y = None
# Iterate over the grid
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        A = np.array(
            [
                [1 + xx[i, j], -xx[i, j], 0],
                [-1, 1 + yy[i, j] + 1, -yy[i, j]],
                [-1, 0, 2],
            ]
        )
        y_star = np.linalg.solve(A, opinions)
        pol = np.dot(y_star, y_star) / 3
        if min_pol is None or pol < min_pol:
            min_pol = pol
            min_x = xx[i, j]
            min_y = yy[i, j]


def test_simple_example_cpu_dense() -> None:
    problem_dense = BeersCpu(
        weights=dense_adj_matrix,
        opinions=opinions,
        mutable_rows=mutable_rows,
        mutable_cols=mutable_cols,
        phi=phi,
        w_0=w_0,
        w=z,
        constraints=constraints,
        dense=True,
    )
    problem_dense.solve(step_size=1)

    assert (
        np.allclose(min_pol, problem_dense.min_cost, atol=1e-6)
        and np.allclose(min_x, problem_dense.min_mutable_weights[0], atol=7e-3)
        and np.allclose(min_y, problem_dense.min_mutable_weights[1], atol=7e-3)
    )


def test_simple_example_cpu_sparse() -> None:
    problem_sparse = BeersCpu(
        weights=sparse_adj_matrix,
        opinions=opinions,
        mutable_rows=mutable_rows,
        mutable_cols=mutable_cols,
        phi=phi,
        w_0=w_0,
        w=z,
        constraints=constraints,
    )
    problem_sparse.solve(step_size=1)

    assert (
        np.allclose(min_pol, problem_sparse.min_cost, atol=1e-6)
        and np.allclose(min_x, problem_sparse.min_mutable_weights[0], atol=7e-3)
        and np.allclose(min_y, problem_sparse.min_mutable_weights[1], atol=7e-3)
    )


def test_simple_example_gpu() -> None:
    problem_gpu = BeersGpu(
        weights=sparse_adj_matrix,
        opinions=opinions,
        mutable_rows=mutable_rows,
        mutable_cols=mutable_cols,
        phi=phi_gpu,
        w_0=w_0,
        ub=np.array([2, 2]),
    )
    problem_gpu.solve(step_size=1)

    assert (
        np.allclose(min_pol, problem_gpu.min_cost, atol=1e-6)
        and np.allclose(min_x, problem_gpu.min_mutable_weights[0], atol=8e-3)
        and np.allclose(min_y, problem_gpu.min_mutable_weights[1], atol=8e-3)
    )
