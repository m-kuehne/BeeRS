import time

import cvxpy as cp
import numpy as np
import scipy as sc
import torch
from scipy.sparse import csr_array
from tqdm import tqdm


class BeersCpu:
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
        (1) (mutable_rows[i], mutable_cols[j]) corresponds to w_ij
        (2) (mutable_rows[i], mutable_cols[j]) must be in "weights", even
        if w_ij is initially 0.
    phi: object
        The upper-level objective
    w_0: np.ndarray
        Initialization of mutable weights
    w: cp.Variable
        cvxpy variable of the mutable weights
    constraints: list
        cvxpy constraints
    project: object
        Custom projection function
    allow_dpp: bool
        Flag to formulate projection as a Disciplined Parametrized Program
    dense: bool
        Flag to indicate that weight matrix is dense

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
        w_0: np.ndarray,
        w: cp.Variable = None,
        constraints: list = None,
        project: object = None,
        allow_dpp: bool = True,
        dense: bool = False,
    ) -> None:
        # Check if weights has correct format
        if dense and not isinstance(weights, np.ndarray):
            raise TypeError("Dense adjacency matrix expected as dense == True.")
        elif not dense and not isinstance(weights, csr_array):
            raise TypeError("Adjacency matrix is expected to be csr_array.")
        self.weights_matrix = weights.astype(np.float32)
        self.s = opinions.astype(np.float32)
        self.mutable_rows = mutable_rows.astype(np.int32)
        self.mutable_cols = mutable_cols.astype(np.int32)
        self.phi = phi
        self.w_0 = w_0.astype(np.float32)
        self.w_cvxpy = w
        self.dense = dense

        # Number of users
        self.n = opinions.shape[0]

        # Number of UL-variables
        self.m = self.mutable_rows.shape[0]

        if not self.dense:
            # Get indptr for J1F
            self.indptr_J1F = self._get_1jf_indptr()

            # Get indptr and indices for J2F
            self.indptr_J2F = self.weights_matrix.indptr + np.arange(
                self.n + 1, dtype=np.int32
            )
            (
                self.indices_J2F,
                self.indices_J2F_diag,
                self.indices_J2F_data,
            ) = self._get_j2f_indices()

            # Get indices of mutable_weights within data of csr_array
            self.mutable_weights_indices = self._get_mutable_weights_indices()

        # Pass w_0 to the weight matrix data
        self._update_data_weights(new_data=self.w_0)

        # Create projection problem
        self.project = project
        if self.project is None:
            # Get default constraints
            if self.w_cvxpy is None:
                self.w_cvxpy = cp.Variable(self.m)
            if constraints is None:
                self.constraints = [self.w_cvxpy >= 0]
            else:
                self.constraints = constraints

            self.projection_param = cp.Parameter(self.m)
            cost = cp.sum_squares(self.w_cvxpy - self.projection_param)
            objective = cp.Minimize(cost)
            self.prob = cp.Problem(objective=objective, constraints=constraints)
            self.allow_dpp = allow_dpp

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
        # The current value of the mutable weights
        self.mutable_weights = self.w_0

        self.costs = []

        self.min_cost = None

        # The weights of the entire network leading to minimum cost
        if self.dense:
            self.min_weights = self._extract_flat_weights()
        else:
            self.min_weights = self.weights_matrix.data

        # The mutable weights leading to minimum cost
        self.min_mutable_weights = self.mutable_weights

        # Used for gradient descent with momentum
        if variant == "Momentum":
            self.last_momentum_term = np.zeros(self.m)
        # Used for Nesterov accelerated gradient descent (NAG)
        # https://stanford.edu/~boyd/papers/pdf/ode_nest_grad.pdf
        elif variant == "NAG":
            self.x_k = self.mutable_weights
        # Used for AdaGrad
        # https://optimization.cbe.cornell.edu/index.php?title=AdaGrad
        elif variant == "AdaGrad":
            self.G_t = np.zeros(self.m)

        # As we want to evaluate the result from the previous iteration
        # if the maximum iteration is reached, we start an additional one
        # and break after evaluation
        # for k in tqdm(range(max_iter + 1)):
        for self.k in range(max_iter):
            # 1) Get J2F
            self.J2F = self._get_j2f()

            # 2) Get equilibrium opinions
            if self.dense:
                self.y_star = np.linalg.solve(self.J2F, 2 * self.s)
            else:
                if self.k < init_with_prev:
                    init = np.zeros(self.n)
                else:
                    init = self.y_star
                self.y_star, info = sc.sparse.linalg.gmres(self.J2F, 2 * self.s, init)
                if info != 0:
                    raise ValueError("Could not compute y_star.")

            # 3) Evaluate cost function and check for convergence
            if self.dense:
                x_torch_eval = torch.tensor(self._extract_flat_weights())
            else:
                x_torch_eval = torch.tensor(self.weights_matrix.data)
            y_torch_eval = torch.tensor(self.y_star)

            cost = self.phi(x_torch_eval, y_torch_eval).detach().item()
            tqdm.write("Cost at current iteration: " + str(cost))
            self.costs.append(cost)

            if self.k > 0:
                if cost < self.min_cost:
                    self.min_cost = cost
                    if self.dense:
                        self.min_weight = self._extract_flat_weights()
                        self.min_mutable_weights = self.weights_matrix[
                            self.mutable_rows, self.mutable_cols
                        ]
                    else:
                        self.min_weights = self.weights_matrix.data
                        self.min_mutable_weights = self.mutable_weights
                    self.y_star_min = self.y_star
                change_over_iteration = np.abs(cost - self.costs[-2]) / np.abs(
                    self.costs[-2]
                )
                if change_over_iteration < tol:
                    break
                if self.k == max_iter:
                    break
            else:
                self.initial_y_star = self.y_star
                self.min_cost = cost
                self.y_star_min = self.y_star

            # 4) Get J1F
            self.J1F = self._get_j1f()

            # 5) Get the gradients
            x_torch = torch.tensor(self.mutable_weights, requires_grad=True)
            y_torch = torch.tensor(self.y_star, requires_grad=True)
            grad_x, grad_y = torch.autograd.grad(
                outputs=self.phi(x_torch, y_torch),
                inputs=[x_torch, y_torch],
                create_graph=False,
                allow_unused=True,
            )
            if grad_x is None:
                grad_x = np.zeros(self.m)
            else:
                grad_x = grad_x.detach().numpy()
            if grad_y is None:
                grad_y = np.zeros(self.n)
            else:
                grad_y = grad_y.detach().numpy()

            # 6) Get v
            if self.dense:
                self.v = np.linalg.solve(np.transpose(self.J2F), grad_y)
            else:
                if self.k < init_with_prev:
                    init = np.zeros(self.n)
                else:
                    init = self.v  # Use previous v as initialization
                self.v, info = sc.sparse.linalg.gmres(self.J2F.T, grad_y, init)

                if info != 0:
                    raise ValueError("Could not compute v.")

            # 7) Get hypergradient
            if self.dense:
                self.hypergrad = grad_x - np.transpose(self.J1F) @ self.v
            else:
                self.hypergrad = grad_x - self.J1F.T @ self.v

            # 8) Make step with first-order method
            if callable(step_size):
                step_size_k = step_size(self.k)
            else:
                step_size_k = step_size
            if variant == "Momentum":
                self.last_momentum_term = (
                    momentum_parameter * self.last_momentum_term + self.hypergrad
                )
                new_weights = (
                    self.mutable_weights - step_size_k * self.last_momentum_term
                )

            elif variant == "NAG":
                x_k_plus_1 = self.mutable_weights - step_size_k * self.hypergrad
                new_weights = x_k_plus_1 + self.k / (self.k + 3) * (
                    x_k_plus_1 - self.x_k
                )
                self.x_k = x_k_plus_1

            elif variant == "AdaGrad":
                self.G_t = self.G_t + np.square(self.hypergrad)
                new_weights = (
                    self.mutable_weights
                    - (step_size_k / np.sqrt(epsilon + self.G_t)) * self.hypergrad
                )

            # 9) Project onto constraints
            if self.project is None:
                projected_weights = self._project(new_weights)
            else:
                projected_weights = self.project(new_weights)

            # 10) Update value of weights
            if self.dense:
                self.weights_matrix[
                    self.mutable_rows, self.mutable_cols
                ] = projected_weights
            else:
                self._update_data_weights(new_data=projected_weights)
            self.mutable_weights = projected_weights

    def _extract_flat_weights(self):
        mask = ~np.eye(self.n, dtype=bool)  # True for off-diagonal elements
        return self.weights_matrix[mask]

    def _get_mutable_weights_indices(self):
        data_indices = np.zeros(self.mutable_cols.shape[0], dtype=np.int32)
        for i, (row, col) in enumerate(zip(self.mutable_rows, self.mutable_cols)):
            start = self.weights_matrix.indptr[row]
            end = self.weights_matrix.indptr[row + 1]
            col_index = np.where(self.weights_matrix.indices[start:end] == col)[0][0]
            data_indices[i] = start + col_index
        return data_indices

    def _get_j2f_indices(self):
        mask = np.zeros(self.indptr_J2F[-1], dtype=np.int32)
        indices_to_set = self.indptr_J2F[1:] - 1
        mask[indices_to_set] = 1
        mask = mask.astype(np.bool_)
        mask_invert = ~mask
        full_indices = np.arange(self.n + self.weights_matrix.data.shape[0])
        diag_indices = full_indices[mask]
        data_indices = full_indices[mask_invert]
        indices_augment = np.zeros(full_indices.shape[0], dtype=np.int32)
        indices_augment[diag_indices] = np.arange(self.n)
        indices_augment[data_indices] = self.weights_matrix.indices
        return indices_augment, diag_indices, data_indices

    def _project(self, x):
        self.projection_param.value = x
        if self.prob.is_qp():
            if self.allow_dpp:
                self.prob.solve(solver=cp.CLARABEL)
            else:
                self.prob.solve(solver=cp.CLARABEL, ignore_dpp=True)
        else:
            if self.allow_dpp:
                self.prob.solve(solver=cp.SCS)
            else:
                self.prob.solve(solver=cp.SCS, ignore_dpp=True)
        if self.prob.status not in ["infeasible", "unbounded"]:
            return np.array(self.w_cvxpy.value, dtype=np.float32)
        else:
            raise ValueError(
                "The projection problem is either infeasible or unbounded. Value: ",
                self.prob.value,
            )

    def _update_data_weights(self, new_data):
        if self.dense:
            self.weights_matrix[self.mutable_rows, self.mutable_cols] = new_data
        else:
            self.weights_matrix.data[self.mutable_weights_indices] = new_data

    def _get_1jf_indptr(self):
        counts = np.bincount(self.mutable_rows, minlength=self.n)
        indptr = np.zeros(self.n + 1, dtype=int)
        indptr[1:] = np.cumsum(counts)
        return indptr

    def _get_j1f_data(self, y):
        return 2 * (y[self.mutable_rows] - y[self.mutable_cols])

    def _get_j1f(self):
        values = self._get_j1f_data(y=self.y_star)
        if self.dense:
            J1F = np.zeros((self.n, self.m))
            mutable_counts = np.bincount(self.mutable_rows, minlength=self.n)
            # Starting index for placing data into matrix
            current_index = 0
            # Fill the matrix row by row
            for i in range(self.n):
                row_data_length = mutable_counts[
                    i
                ]  # How many entries to place in this row
                J1F[i, current_index : current_index + row_data_length] = values[
                    current_index : current_index + row_data_length
                ]
                current_index += (
                    row_data_length  # Update the index to the next segment in data
                )
        else:
            J1F = csr_array(
                (values, np.arange(self.m), self.indptr_J1F),
                shape=(self.n, self.m),
            )
        return J1F

    def _get_j2f(self):
        if self.dense:
            J2F = 2 * (
                -self.weights_matrix
                + np.eye(self.n)
                + np.diag(np.sum(self.weights_matrix, axis=1))
            )
        else:
            diagonals = 1 + np.sum(self.weights_matrix, axis=1)
            data_augment = np.zeros(self.weights_matrix.data.size + self.n)
            data_augment[self.indices_J2F_diag] = diagonals
            data_augment[self.indices_J2F_data] = -self.weights_matrix.data
            J2F = csr_array(
                (2 * data_augment, self.indices_J2F, self.indptr_J2F),
                shape=(self.n, self.n),
            )
        return J2F
