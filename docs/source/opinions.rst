Advanced Example
================

In the following, we solve a more involved example on the CPU using
:ref:`BeersCpu <beers-cpu-class>`. The problem could also be solved with slight modifications
:ref:`on the GPU <beers-gpu-class>`, and we highlight the differences when applicable.

Problem Formulation
-------------------

We solve the problem

.. math::
   \DeclareMathOperator*{\argmin}{\arg\!\min\,}
   \begin{aligned}
   & \min_{w_{i \zeta}} && \frac{1}{n} \sum_{i=1}^n y_i^2 \\
   & \text{s.t.} && A(w)y = s \\
   & && w_{i \zeta} \geq 0, \forall i \in \mathcal{V} \setminus \zeta \\
   & && \sum_{i=1}^n w_{i \zeta} \leq b,
   \end{aligned}

on an extension to (a subset) the deezer dataset :cite:`deezer`. In particular, we add a connection
from every pre-existing user :math:`i\in \mathcal{V}` to a newly introduced influencer :math:`\zeta`,
and our goal is to find the best weight :math:`w_{i \zeta}` for all :math:`i \in \mathcal{V}`, such that
polarization in the social network is minimized. Internal opinions are drawn
uniformly at random from the set :math:`\{-1,+1\}` for pre-existing users and set to 0 for the influencer.

.. admonition:: Interpretation

    A practical interpretation of this problem formulation is as
    follows. The influencer represents a neutral news agency
    that seeks to distribute its content in a social network. The
    network operator, corresponding to the leader, has to decide
    how to deliver its content, i.e., news articles, to the users
    of the network. The weights :math:`w_{i \zeta} \in \mathbb{R}_{\geq 0}` on the edges from
    user i to the news company :math:`\zeta` are proportional to the share of
    :math:`\zeta`'s news items appearing in i's feed. In the interest of peaceful
    interactions, the leader tries to assign the weights such
    that polarization is minimized. The news agency incurs an
    advertising fee for the distribution of its content, which we
    assume to be proportional to
    :math:`\sum_{i=1}^n w_{i \zeta}` , and upper-bounded
    by :math:`b`, the agency's advertising budget.

Solving the Problem
-------------------

Loading the Data
^^^^^^^^^^^^^^^^

First, we load the (dense) adjacency matrix of the dataset and randomly generated internal opinions.
Additionally, we specify the number of users (without the influencer) and the budget of the influencer.

.. code-block:: python

    n = 1000
    b = 100

    # Load the data
    deezer = np.load("deezer_adj_direct.npy")[:n, :n]
    opinions = np.load("deezer_opinions.npy")[:n]

Augmenting the Data and Transformation into Sparse Representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next we add the influencer to the dataset. Since every user is influenced by the influencer, 
we add a column with connections to the influencer. The influencer is not influenced by 
anyone, so we add a row of 0's. Note that, all weights we want to optimize must be present
in the sparse representation of the matrix. Therefore, we add small weights of :code:`1e-3` to the 
last column, because :code:`csr_array` removes them otherwise. 
Keep in mind that these values do not necessarily correspond to the initial weights of 
these connections, as we can specify them seperately.

.. code-block:: python
    
    # Augment the data to add influencer
    column_to_infl = np.full((n, 1), 1e-3)
    row_from_infl = np.full((1, n + 1), 0)
    deezer = np.hstack((deezer, column_to_infl))
    deezer = np.vstack((deezer, row_from_infl))
    opinions = np.append(opinions, 0)

    # Get sparse representation of the adjacency matrix
    sparse_deezer_direct = csr_array(deezer)

Define Mutable Weights and Initial values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As we want to optimize over all connections to the influencer, the mutable weights are simply
the last column of the augmented adjacency matrix. Additionally, we initialize all connections
to the influencer as 0 (therefore, setting the weights to :code:`1e-3` in the previous step does 
not have an effect).

.. code-block:: python
    
    # Get indices of mutable data
    mutable_rows = np.arange(n)
    mutable_cols = np.full(n, n)

    # Get the initial weights of the mutable connections
    w_0 = np.zeros(mutable_rows.shape[0])

Defining the Objective
^^^^^^^^^^^^^^^^^^^^^^

We use a standard polarization metric (see, e.g., :cite:`pol`).

.. code-block:: python

    # Upper-level objective
    def phi(w: torch.tensor, y: torch.tensor):
        yTy = torch.dot(y, y)
        n = y.shape[0]
        return yTy / n

.. admonition:: Difference to GPU
    
    :ref:`On the GPU <beers-gpu-class>`, the objective is defined with JAX.

    .. code-block:: python

        # Upper-level objective
        @jax.jit
        def phi(w, y):
            yTy = jnp.dot(y, y)
            n = y.shape[0]
            return yTy / n

Defining the Constraints
^^^^^^^^^^^^^^^^^^^^^^^^

We define the constraints with CVXpy. If they are not passed to :code:`BeersCpu`, the constraint
:math:`w \geq 0` is added by default.

.. code-block:: python

    # We have the constraints w >= 0 and sum w <= b
    w = cp.Variable(mutable_rows.size)
    constraints = [cp.sum(w) <= b, w >= 0]

.. admonition:: Custom projection on CPU

    On the CPU, we can specify a custom projection of the form

    .. code-block:: python

        def project(to_be_projected):
            projected = ...
            ...
            return projected

    Then, :code:`w` and :code:`constraints` are not necessary.


.. admonition:: Difference to GPU
    
    :ref:`On the GPU <beers-gpu-class>`, the projection is done with 
    `jaxopt.BoxOSQP <https://jaxopt.github.io/stable/_autosummary/jaxopt.BoxOSQP.html>`_.
    We specify :code:`A`, :code:`lb`, and :code:`ub`. Note that, :code:`A` is a sparse :code:`csr_array`.

    .. code-block:: python

        # We have the constraints w >= 0 and sum w <= b
        # We have to specify the A, l, b as in
        # https://jaxopt.github.io/stable/_autosummary/jaxopt.BoxOSQP.html
        lb = np.full(n + 1, 0)
        ub = np.full(n + 1, jnp.inf)
        ub[-1] = b
        A = np.eye(n, n)
        A = np.vstack((A, np.ones((1, n))))
        A = csr_array(A)

Defining the Problem
^^^^^^^^^^^^^^^^^^^^

We define the problem.

.. code-block:: python

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

.. admonition:: Custom projection on CPU

    If we specify a custom projection :code:`project`, then we define

    .. code-block:: python

        # Define the problem
        problem = BeersCpu(
            weights=sparse_deezer_direct,
            opinions=opinions,
            mutable_rows=mutable_rows,
            mutable_cols=mutable_cols,
            phi=phi,
            w_0=w_0,
            project=project,
        )

Solving the Problem
^^^^^^^^^^^^^^^^^^^

We solve it and print the results.

.. code-block:: python

    problem.solve(step_size=10, tol=1e-3, momentum_parameter=0.95)
    print("Min cost: ", problem.min_cost)

.. code-block:: console

    Cost at current iteration: 0.6552165450135484                                                                             
    Cost at current iteration: 0.6257607973540302                                                                             
    Cost at current iteration: 0.5768141724928133                                                                             
    Cost at current iteration: 0.519722031427104                                                                              
    Cost at current iteration: 0.4754177256153373                                                                             
    Cost at current iteration: 0.4711191726284613                                                                             
    Cost at current iteration: 0.4688381978000241                                                                             
    Cost at current iteration: 0.46831414619373873                                                                            
    Cost at current iteration: 0.4683502774751571                                                                             
    8%|██████▊                                      | 8/100 [00:00<00:02, 39.01it/s]
    Min cost:  0.46831414619373873