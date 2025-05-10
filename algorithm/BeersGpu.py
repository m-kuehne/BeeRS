import os
import time
from functools import partial

import jax
import jax.experimental.sparse as sp
import jax.numpy as jnp
import jaxopt
import numpy as np
import scipy as sc
from scipy.sparse import csr_array
from tqdm import tqdm

# Flag significantly improves projection and gmres time, see
# https://github.com/jax-ml/jax/issues/9259
os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"


class BeersGpu:
    """Solve a recommender system optimization.

    Parameters
    ----------
    weights: csr_array
        The adjacency matrix of the initial weights of the network
    opinions: np.ndarray
        The internal opinions of the users
    mutable_rows: np.ndarray
        The row indices corresponding to mutable weights
    mutable_cols: np.ndarray
        The column indices corresponding to mutable weights. Note that,
        (1) (mutable_rows[i], mutable_cols[j]) corresponds to w_ij, and
        (2) (mutable_rows[i], mutable_cols[j]) must be in "weights", even
        if w_ij is initially 0.
    phi: object
        The upper-level objective
    A: csr_array
        Defines equality constraints
        (see https://jaxopt.github.io/stable/_autosummary/jaxopt.BoxOSQP.html)
    lb: np.ndarray
        Lower-bound constraints
    ub: np.ndarray
        Upper-bound constraints
    w_0: np.ndarray
        Initialization of mutable weights

    Returns
    -------
    None
    """

    def __init__(
        self,
        weights: csr_array,
        opinions: np.ndarray,
        mutable_rows: np.ndarray,
        mutable_cols: np.ndarray,
        phi: object,
        A: csr_array = None,
        lb: np.ndarray = None,
        ub: np.ndarray = None,
        w_0: np.ndarray = None,
        project: object = None
    ) -> None:
        self.data_weights = jnp.array(weights.data).astype(jnp.float32)
        self.indices_weights = jnp.array(weights.indices).astype(jnp.int32)
        self.indptr_weights = jnp.array(weights.indptr).astype(jnp.int32)
        self.s = jnp.array(opinions).astype(jnp.float32)
        self.mutable_rows = jnp.array(mutable_rows).astype(jnp.int32)
        self.mutable_cols = jnp.array(mutable_cols).astype(jnp.int32)
        self.phi = phi
        self.w_0 = w_0
        self.project = project

        # Number of users
        self.n = opinions.shape[0]

        # Number of UL-variables
        self.m = self.mutable_rows.shape[0]

        # Get data required to construct J1F
        self.indptr_J1F = _get_1jf_indptr(self.mutable_rows, self.n)

        # Get default constraints
        if A is None and self.project is None:
            data = jnp.ones(self.m)
            indices = jnp.arange(self.m)
            indptr = jnp.arange(self.m + 1)
            self.A = sp.CSR((data, indices, indptr), shape=(self.m, self.m))
        elif self.project is None:
            self.A = sp.CSR((A.data, A.indices, A.indptr), shape=A.shape)

        if lb is None and self.project is None:
            self.lb = jnp.zeros(self.m)
        elif self.project is None:
            self.lb = jnp.array(lb)

        if ub is None and self.project is None:
            self.ub = jnp.full(self.m, jnp.inf)
        elif self.project is None:
            self.ub = jnp.array(ub)

        # We assume a fixed sparsity structure of J2F, so we only get the indices,
        # indptr once
        self.indptr_J2F = self.indptr_weights + jnp.arange(self.n + 1, dtype=jnp.int32)
        (
            self.indices_J2F,
            self.indices_J2F_diag,
            self.indices_J2F_data,
        ) = self._get_j2f_indices(indptr_augment=self.indptr_J2F)

        # Get the value of the mutable weights
        if w_0 is None:
            # No initialization given, we take the values provided by weights
            # self.initial_mutable_weights = self._get_mutable_weights(
            #     weights, self.data_weights, mutable_rows, mutable_cols
            # )
            self.initial_mutable_weights = self._get_mutable_weights_jax(
                weights, self.data_weights, mutable_rows, mutable_cols
            )
        else:
            # We use the specific initialization
            self.initial_mutable_weights = jnp.array(w_0)

        # Get the gradients of the UL-objective
        self.grads = jax.jit(jax.grad(phi, argnums=(0, 1)))

        # Get indices of mutable_weights within data
        self.mutable_weights_indices = jnp.array(
            self._get_mutable_weights_indices(
                csr_array=weights, mutable_rows=mutable_rows, mutable_cols=mutable_cols
            )
        )

        # The following code is pure JAX but extremely slow
        # self.mutable_weights_indices = self._get_mutable_weights_indices_jax(
        #     csr_array=weights, mutable_rows=mutable_rows, mutable_cols=mutable_cols
        # )
        # self.mutable_weights_indices.block_until_ready()

        # Update weights if w_0 is given
        if self.w_0 is not None:
            self.data_weights = _update_data_weights(
                old_data=self.data_weights,
                new_data=self.w_0,
                indices=self.mutable_weights_indices,
            )

    def solve(
        self,
        max_iter=100,
        tol=1e-5,
        step_size=1,
        momentum_parameter=0.95,
        variant="Momentum",
        epsilon=1e-5,
        init_with_prev=100,
    ):
        """Solve a recommender system optimization.

        Parameters
        ----------
        max_iter: float
            The maximum number of iterations
        tol: float
            The tolerance used as termination criterion of the iteration
        step_size: float
            Step size used for the first-order method
        momentum_parameter: float
            Parameter used for gradient-descent with momentum
        variant: str
            Selects first-order method
        epsilon: float
            Parameter used for AdaGrad
        init_with_prev: int
            Initialize GMRES to solve for v and y star with result from previous
            iteration at iteration init_with_prev


        Returns
        -------
        None
        """
        mutable_weights = self.initial_mutable_weights

        self.costs = []

        self.min_cost = None

        # The weights of the entire network leading to minimum cost
        self.min_weights = self.data_weights

        # The mutable weights leading to minimum cost
        self.min_mutable_weights = mutable_weights

        # Used for gradient descent with momentum
        self.last_momentum_term = jnp.zeros(self.m)

        # Used for Nesterov accelerated gradient descent (NAG)
        # https://stanford.edu/~boyd/papers/pdf/ode_nest_grad.pdf
        self.x_k = mutable_weights

        # Used for AdaGrad
        # https://optimization.cbe.cornell.edu/index.php?title=AdaGrad
        self.G_t = jnp.zeros(mutable_weights.shape[0])

        # As we want to evaluate the result from the previous iteration
        # if the maximum iteration is reached, we start an additional one
        # and break after evaluation
        # for k in tqdm(range(max_iter + 1)):
        for k in range(max_iter):
            # 1) Get J2F
            self.J2F = _get_j2f(
                self.indices_weights,
                self.data_weights,
                self.indptr_weights,
                self.indices_J2F,
                self.indptr_J2F,
                self.indices_J2F_diag,
                self.indices_J2F_data,
                self.n,
            )

            # 2) Get equilibrium opinions
            if k < init_with_prev:
                init = jnp.zeros(self.s.shape[0])
            else:
                init = self.y_star
            self.y_star, info = _gmres(self.J2F, 2 * self.s, init)
            if info == 0:
                self.y_star = jnp.array(self.y_star)
            else:
                raise ValueError("Could not compute y_star.")

            # 3) Evaluate cost function and check for convergence
            cost = self.phi(self.data_weights, self.y_star)
            cost.block_until_ready()
            tqdm.write("Cost at current iteration: " + str(cost))
            self.costs.append(cost.item())
            if k > 0:
                if cost < self.min_cost:
                    self.min_cost = cost
                    self.min_weights = self.data_weights
                    self.min_mutable_weights = mutable_weights
                    self.y_star_min = self.y_star
                if _converged(cost, self.costs[k - 1], tol):
                    break
                if k == max_iter:
                    break
            else:
                self.min_cost = cost
                self.y_star_min = self.y_star

            # 4) Get J1F
            # Get J1F
            self.J1F = _get_j1f(
                self.y_star, self.mutable_rows, self.mutable_cols, self.indptr_J1F
            )

            # 5) Get the gradients
            grad_x, grad_y = self.grads(mutable_weights, self.y_star)

            # 6) Get v
            if k < init_with_prev:
                init = jnp.zeros(self.s.shape[0])
            else:
                init = self.v  # Use previous v as initialization
            self.v, info = _gmres(self.J2F.T, grad_y, init)

            if info == 0:
                self.v = jnp.array(self.v)
            else:
                raise ValueError("Could not compute v.")

            # 7) Get hypergradient
            self.hypergrad = _get_hypergrad(grad_x, self.J1F, self.v)

            # 8) Make step with first-order method
            if callable(step_size):
                step_size_k = step_size(k)
            else:
                step_size_k = step_size
            if variant == "Momentum":
                new_weights, self.last_momentum_term = _momentum_step(
                    mutable_weights,
                    self.last_momentum_term,
                    self.hypergrad,
                    momentum_parameter,
                    step_size_k,
                )
            elif variant == "NAG":
                new_weights, self.x_k = _nag_step(
                    mutable_weights, self.x_k, self.hypergrad, step_size_k, k
                )
            elif variant == "AdaGrad":
                new_weights, self.G_t = _adagrad_step(
                    mutable_weights, self.G_t, self.hypergrad, step_size_k, epsilon
                )

            if self.project is None:
                projected_weights = _project(new_weights, self.A, self.lb, self.ub)
            else:
                projected_weights = self.project(new_weights)

            # 10) Update value of weights
            self.data_weights = _update_data_weights(
                old_data=self.data_weights,
                new_data=projected_weights,
                indices=self.mutable_weights_indices,
            )
            mutable_weights = projected_weights

    def _get_mutable_weights_indices(self, csr_array, mutable_rows, mutable_cols):
        indices = csr_array.indices
        indptr = csr_array.indptr
        data_indices = np.zeros(mutable_cols.shape[0], dtype=int)
        for i, (row, col) in enumerate(zip(mutable_rows, mutable_cols)):
            start = indptr[row]
            end = indptr[row + 1]
            col_index = np.where(indices[start:end] == col)[0][0]
            data_indices[i] = start + col_index
        return jnp.array(data_indices)

    # This function is pure JAX but extremely slow compared to the numpy version above
    def _get_mutable_weights_indices_jax(self, csr_array, mutable_rows, mutable_cols):
        indices = csr_array.indices
        indptr = csr_array.indptr
        row_starts = indptr[mutable_rows]
        row_ends = indptr[mutable_rows + 1]
        column_indices = jnp.array(
            [
                jnp.searchsorted(indices[start:end], col)
                for start, end, col in zip(row_starts, row_ends, mutable_cols)
            ]
        )
        data_indices = row_starts + column_indices
        return data_indices

    def _get_mutable_weights_jax(self, csr_array, data, mutable_rows, mutable_cols):
        data = csr_array.data
        indices = csr_array.indices
        indptr = csr_array.indptr
        extracted_values = jnp.zeros(mutable_rows.shape[0])
        row_start = indptr[mutable_rows]
        row_end = indptr[mutable_rows + 1]
        for i in range(len(mutable_rows)):
            row_indices = indices[row_start[i] : row_end[i]]
            row_data = data[row_start[i] : row_end[i]]
            mask = row_indices == mutable_cols[i]
            row_values = row_data[mask][0]
            extracted_values = extracted_values.at[i].set(row_values)
        return extracted_values

    def _get_j2f_indices(self, indptr_augment):
        mask = jnp.zeros(indptr_augment[-1], dtype=jnp.int32)
        indices_to_set = indptr_augment[1:] - 1
        mask = mask.at[indices_to_set].set(1)
        mask = mask.astype(jnp.bool_)
        mask_invert = ~mask
        full_indices = jnp.arange(self.n + self.data_weights.shape[0])
        diag_indices = full_indices[mask]
        data_indices = full_indices[mask_invert]
        indices_augment = jnp.zeros(
            self.data_weights.shape[0] + self.n, dtype=jnp.int32
        )
        indices_augment = indices_augment.at[diag_indices].set(jnp.arange(self.n))
        indices_augment = indices_augment.at[data_indices].set(self.indices_weights)
        return indices_augment, diag_indices, data_indices


@jax.jit
def _matvec_q(Q, x):
    return x


@jax.jit
def _matvec_a(A, x):
    return sp.csr_matvec(A, x)


@jax.jit
def _project(z, A, lb, ub):
    solver = jaxopt.BoxOSQP(
        check_primal_dual_infeasability=False, matvec_A=_matvec_a, matvec_Q=_matvec_q
    )
    sol = solver.run(params_obj=(None, -z), params_eq=A, params_ineq=(lb, ub))
    # state = sol.state.status
    solution = sol.params.primal[0]
    return solution


@jax.jit
def _gmres(A, b, x0):
    x, info = jax.scipy.sparse.linalg.gmres(A=A, b=b, x0=x0)
    return x, info


@partial(jax.jit, static_argnums=(3, 4))
def _adagrad_step(x_t, G_t_minus_1, hypergrad, alpha, epsilon):
    G_t = G_t_minus_1 + jnp.square(hypergrad)
    x_t_plus_1 = x_t - (alpha / jnp.sqrt(epsilon + G_t)) * hypergrad
    return x_t_plus_1, G_t


@partial(jax.jit, static_argnums=(3))
def _nag_step(y_k_minus_1, x_k_minus_1, hypergrad, alpha, k_minus_1):
    x_k = y_k_minus_1 - alpha * hypergrad
    y_k = x_k + k_minus_1 / (k_minus_1 + 3) * (x_k - x_k_minus_1)
    return y_k, x_k


@partial(jax.jit, static_argnums=(3, 4))
def _momentum_step(last_x, last_m, hypergrad, gamma, alpha):
    next_m = gamma * last_m + hypergrad
    next_x = last_x - alpha * next_m
    return next_x, next_m


@partial(jax.jit, static_argnums=(2))
def _converged(current_cost, last_cost, tol):
    return jnp.abs(current_cost - last_cost) / jnp.abs(last_cost) < tol


@jax.jit
def _update_data_weights(old_data, new_data, indices):
    return old_data.at[indices].set(new_data)


@jax.jit
def _get_hypergrad(grad_x, J1F, v):
    return grad_x - J1F.T @ v


@jax.jit
def _get_j1f_data(y, mutable_rows, mutable_cols):
    return 2 * (y[mutable_rows] - y[mutable_cols])


@partial(jax.jit, static_argnums=(1))
def _get_1jf_indptr(mutable_rows, n):
    counts = jnp.bincount(mutable_rows, length=n)
    indptr = jnp.zeros(n + 1, dtype=int)
    return indptr.at[1:].set(jnp.cumsum(counts))


@jax.jit
def _get_j1f(y, mutable_rows, mutable_cols, indptr_J1F):
    values = _get_j1f_data(y=y, mutable_rows=mutable_rows, mutable_cols=mutable_cols)
    J1F = sp.CSR(
        (values, jnp.arange(mutable_rows.shape[0]), indptr_J1F),
        shape=(y.shape[0], mutable_rows.shape[0]),
    )
    return J1F


@partial(jax.jit, static_argnums=(7))
def _get_j2f(
    indices,
    data,
    indptr,
    indices_augment,
    indptr_augment,
    diag_indices,
    data_indices,
    n,
):
    diagonals = 1 + sp.csr_matvec(
        sp.CSR((data, indices, indptr), shape=(n, n)), jnp.ones(n)
    )
    data_augment = jnp.zeros(data.shape[0] + n)

    data_augment = data_augment.at[diag_indices].set(diagonals)
    data_augment = data_augment.at[data_indices].set(-data)

    J2F = sp.CSR((2 * data_augment, indices_augment, indptr_augment), shape=(n, n))

    return J2F
