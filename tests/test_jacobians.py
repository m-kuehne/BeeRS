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

# We test if the correct Jacobians are computed
# We consider an easy example, where we can compute the Jacobians by hand and
# compare to the resulting Jacobians from Beers


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


def test_jacobians_cpu_dense() -> None:
    problem_dense = BeersCpu(
        weights=dense_adj_matrix,
        opinions=opinions,
        mutable_rows=mutable_rows,
        mutable_cols=mutable_cols,
        phi=phi,
        w_0=w_0,
        dense=True,
    )
    problem_dense.solve(max_iter=1)

    # we use the initial values of w because we only consider the 1st iteration
    j2f = 2 * np.array(
        [[1 + w_0[0], -w_0[0], 0], [-1, 1 + w_0[1] + 1, -w_0[1]], [-1, 0, 2]]
    )

    equilibrium_opinions = problem_dense.initial_y_star
    j1f = 2 * np.array(
        [
            [equilibrium_opinions[0] - equilibrium_opinions[1], 0],
            [0, equilibrium_opinions[1] - equilibrium_opinions[2]],
            [0, 0],
        ]
    )

    assert np.allclose(problem_dense.J2F, j2f) and np.allclose(problem_dense.J1F, j1f)


def test_jacobians_cpu_sparse() -> None:
    problem_sparse = BeersCpu(
        weights=sparse_adj_matrix,
        opinions=opinions,
        mutable_rows=mutable_rows,
        mutable_cols=mutable_cols,
        phi=phi,
        w_0=w_0,
    )
    problem_sparse.solve(max_iter=1)

    # we use the initial values of w because we only consider the 1st iteration
    j2f = 2 * np.array(
        [[1 + w_0[0], -w_0[0], 0], [-1, 1 + w_0[1] + 1, -w_0[1]], [-1, 0, 2]]
    )

    equilibrium_opinions = problem_sparse.initial_y_star
    j1f = 2 * np.array(
        [
            [equilibrium_opinions[0] - equilibrium_opinions[1], 0],
            [0, equilibrium_opinions[1] - equilibrium_opinions[2]],
            [0, 0],
        ]
    )

    assert np.allclose(problem_sparse.J2F.todense(), j2f) and np.allclose(
        problem_sparse.J1F.todense(), j1f
    )


def test_jacobians_gpu() -> None:
    problem_gpu = BeersGpu(
        weights=sparse_adj_matrix,
        opinions=opinions,
        mutable_rows=mutable_rows,
        mutable_cols=mutable_cols,
        phi=phi_gpu,
        w_0=w_0,
    )
    problem_gpu.solve(max_iter=1)

    # we use the initial values of w because we only consider the 1st iteration
    j2f = 2 * np.array(
        [[1 + w_0[0], -w_0[0], 0], [-1, 1 + w_0[1] + 1, -w_0[1]], [-1, 0, 2]]
    )

    equilibrium_opinions = problem_gpu.y_star
    j1f = 2 * np.array(
        [
            [equilibrium_opinions[0] - equilibrium_opinions[1], 0],
            [0, equilibrium_opinions[1] - equilibrium_opinions[2]],
            [0, 0],
        ]
    )

    assert np.allclose(problem_gpu.J2F.todense(), j2f) and np.allclose(
        problem_gpu.J1F.todense(), j1f
    )
