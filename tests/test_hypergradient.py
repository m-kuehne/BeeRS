import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy.sparse import csr_array

from algorithm.BeersGpu import BeersGpu
from algorithm.BeersCpu import BeersCpu

# We test if the hypergradient is correct
# We consider an easy example, where we can compute the Jacobians and thus
# the hypergradient by hand


# Define upper-level optimization function
def phi(x: torch.tensor, y: torch.tensor):
    return y.t() @ y / 3


def grad_y(x, y):
    return 2 / 3 * y


@jax.jit
def phi_gpu(x: jnp.array, y: jnp.array):
    return jnp.dot(y, y) / 3


dense_adj_matrix = np.array([[0, 1.4, 0], [1, 0, 1.1], [1, 0, 0]])
sparse_adj_matrix = csr_array(dense_adj_matrix)
opinions = np.array([-1, 1, 0.5])
mutable_rows = np.array([0, 1])
mutable_cols = np.array([1, 2])
w_0 = np.array([1.4, 1.1])


def test_hypergrad_cpu_dense() -> None:
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

    sensitivity = -np.linalg.solve(j2f, j1f)
    hypergradient = sensitivity.T @ grad_y(w_0, equilibrium_opinions)

    assert np.allclose(problem_dense.hypergrad, hypergradient)


def test_hypergrad_cpu_sparse() -> None:
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

    sensitivity = -np.linalg.solve(j2f, j1f)
    hypergradient = sensitivity.T @ grad_y(w_0, equilibrium_opinions)

    assert np.allclose(problem_sparse.hypergrad, hypergradient)


def test_hypergrad_gpu() -> None:
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

    sensitivity = -np.linalg.solve(j2f, j1f)
    hypergradient = sensitivity.T @ grad_y(w_0, equilibrium_opinions)

    assert np.allclose(problem_gpu.hypergrad, hypergradient)
