Problem Formulation & Solution
==============================

Basics
------
We consider a social network consisting of :math:`n := \lvert \mathcal{V} \rvert` users, where :math:`\mathcal{V}`
denotes the set of users. Every user :math:`i \in \mathcal{V}` hold a constant internal opinion 
:math:`s_i \in \mathbb{R}` and a time-varying external opinion :math:`y_i \in \mathbb{R}`. The external 
opinions are updated according to the FJ dynamics as

.. math::
      
      y_i^{(k+1)} = \frac{s_i + \sum_{j \in \mathcal{N}_i} w_{ij} y_j^{(k)}}{1 + \sum_{j \in \mathcal{N}_i} w_{ij}},

where :math:`w_{ij} \in \mathbb{R}_{\geq 0}` represent the weight of the connection from user :math:`i` to user
:math:`j`. Intuitively, the weight indicates how susceptible user :math:`i` is to the opinion of user :math:`j`.
The set :math:`\mathcal{N}_i` 
represents the neighborhood of user :math:`i`, i.e., 
:math:`\mathcal{N}_i = \{j \mid w_{ij} > 0, \, \forall j \in \mathcal{V} \setminus i \}`.

We consider networks without self loops and we define :math:`w` as the flattened row-major form of 
the network's adjacency matrix :math:`W` with omitted diagonal entries :math:`W[i,i]` (since we do not 
allow self loops, these weights are 0 anyways). Further, we define 
:math:`y := [y_1, \ldots, y_n]^\top \in \mathbb{R}^n` and :math:`s := [s_1, \ldots, s_n]^\top \in \mathbb{R}^n`.
It can be shown that the FJ update leads to a unique equilibrium :math:`y^\star(w)`, and it can be easily seen
that this equilibrium is implicitly given by the equation

.. math::

   \underbrace{
   \begin{bmatrix}
       1 + \sum_{j \in \mathcal{N}_1} w_{1j} & \cdots & -w_{1n} \\
       \vdots & \ddots & \vdots \\
       -w_{n1} & \cdots & 1 + \sum_{j \in \mathcal{N}_n} w_{nj}
   \end{bmatrix}
   }_{A(w)} y = s.

Then, we can write the problem of finding the best network weight as

.. math::
   \DeclareMathOperator*{\argmin}{\arg\!\min\,}
   \begin{aligned}
   & \min_{w,y} && \varphi(w,y) \\
   & \text{s.t.} && A(w) y = s \\
   & && w \in \mathcal{W},
   \end{aligned}

where :math:`\varphi` is the leaders' continuously differentiable objective function, 
and :math:`\mathcal{W}` is a convex, closed, and non-empty set, restricting the possible interventions.
The feasible set of this problem is non-convex due to the constraint constraint :math:`A(w)y = s`. To address
this issue, we reformulate the problem as

.. math::
   \DeclareMathOperator*{\argmin}{\arg\!\min\,}
   \begin{aligned}
   & \min_{w} && \varphi(w,y^\star(w)) \\
   & \text{s.t.} && w \in \mathcal{W},
   \end{aligned}

where :math:`y^\star(w)` solves :math:`A(w)y = s`.

Difficulty
----------

Although the constraints are now convex, the problem is
still challenging to solve. The overall problem is still non-convex since :math:`y^\star(w)` is an implicit mapping,
for which a closed-form expression is, in general, not available. Even
though solvers exists for such optimization problems, they
might perform poorly even for small-scale problems. Nevertheless, the continuous
differentiability of the objective is an appealing property,
considering that first-order methods have been used successfully
for large-scale non-convex problems.

Solution Idea
-------------

To solve the problem (locally), we run projected gradient descent. The projection ensures feasibility of the
iterates :math:`w^{(k)}`.
To get the hypergradient :math:`\nabla_w \varphi(w, y^\star(w))`, the gradient of :math:`\varphi` with 
respect to :math:`w`, we apply the chain rule

.. math::

    \nabla_w \varphi(w, y^\star(w)) = \nabla_1 \varphi(w, y^\star(w)) +
    Jy^\star(w)^\top \nabla_2 \varphi(w,y^\star(w)),

where :math:`\nabla_1 \varphi(w, y^\star(w))` and :math:`\nabla_2 \varphi(w, y^\star(w))`
denote the partial gradients of :math:`\varphi(w, y^\star(w))` with respect to the first and second
argument, respectively.
Intuitively, the sensitivity :math:`Jy^\star(w)` tells us how the equilibrium opinions :math:`y^\star` 
change when the network weights :math:`w` change.

Obtaining the Sensitivity
^^^^^^^^^^^^^^^^^^^^^^^^^

To obtain the sensitivity, we consider the equation

.. math::

    F(w,y) := 2(A(w)y - s) = 0.

By taking the derivative and invoking the implicit function theorem :cite:p:`Dontchev2009{Th.1B.1}`,
we find

.. math::

    Jy^\star(w) = -J_2 F(w, y^\star(w))^{-1} J_1 F(w,y^\star(w)),

where :math:`J_1F(w,y)` and :math:`J_2F(w,y)` are the partial Jacobians of :math:`F(w,y)` with respect
to the first and second argument, respectively. They are given by

.. math::

    J_2 F(w,y^\star(w)) = 2A(w).

and 

.. math::
    \small
    J_1F(w,y) = 
    2\begin{bmatrix}
        y_1 - y_2 & y_1 - y_3 & \cdots & y_1 - y_n & 0 & \cdots & 0 &  0 & 0 & \cdots & 0 \\
            & & & & & \vdots & & & & &  \\
        0 & 0 & \cdots &  0 & 0 & \cdots & 0 & y_n - y_1 & y_n - y_2 & \cdots & y_n - y_{n-1}
    \end{bmatrix}.

We could now get the hypergradient by plugging the pieces into the equation above. However, at
this point, it is worth mentioning that computing the sensitivity with the matrix equation above might
be computationally infeasible for large problems.

Efficiently Obtaining the Hypergradient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using a trick shown in :cite:`Grazzi2020OnTI`, we can avoid this issue. Observe that
we do not actally need the sensitivity :math:`Jy^\star(w)` but the product

.. math::

    Jy^\star(w)^\top \nabla_2 \varphi(w,y^\star(w)).

By inserting the analytical solution for the sensitivity, we get

.. math::

    - J_1 F(w,y^\star(w))^\top (J_2 F(w, y^\star(w))^\top)^{-1} \nabla_2 \varphi(w,y^\star(w)).

This expression is equivalent to

.. math::

    - J_1 F(w,y^\star(w))^\top v,

where `v` is given by

.. math::

    J_2 F(w, y^\star(w))^\top v = \nabla_2 \varphi(w,y^\star(w)).

This is significantly easier to solve than the original system. The resulting hypergradient is
given by

.. math::

    \nabla_w \varphi(w, y^\star(w)) = \nabla_1 \varphi(w, y^\star(w)) - J_1 F(w,y^\star(w))^\top v.

Running Projected Gradient Descent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, we are able to run projected gradient descent. Our software allows three distinct variations.
First, we have projected gradient descent with momentum with the update

.. math::

    m^{(k+1)} = \gamma m^{(k)} + \nabla_w \varphi(w^{(k)}, y(w^{(k)})) \\
    w^{(k+1)} = \Pi_\mathcal{W} [w^{(k)} - \alpha^{(k)} m^{(k+1)}],

with a (possibly iteration dependant) step size :math:`\alpha^{(k)}` and momentum parameter :math:`\gamma`.
For details, consider `this link <https://distill.pub/2017/momentum/>`_.

Second, we have projected Nesterov Accelerated Gradient Descent with the update

.. math::

    x^{(k)} = w^{(k-1)} - \alpha^{(k)} \nabla_w \varphi(w^{(k-1)}, y(w^{(k-1)})) \\
    w^{(k)} = \Pi_\mathcal{W} [x^{(k)} + \frac{k - 1}{k + 2} (x^{(k)} - x^{(k-1)})],

again with step size :math:`\alpha^{(k)}`.
For details, consider `this link <https://stanford.edu/~boyd/papers/pdf/ode_nest_grad.pdf>`_.

Third, we have projected AdaGrad with the update

.. math::

    G^{(k+1)}[i] = G^{(k)}[i] + ( \nabla_w \varphi(w^{(k)}, y(w^{(k)}))[i] )^2 \\
    w^{(k+1)}[i] = \Pi_\mathcal{W} [w^{(k)}[i] - \alpha^{(k)} \frac{1}{\sqrt{\epsilon + G^{(k+1)}[i]}} \nabla_w \varphi(w^{(k)}, y(w^{(k)}))[i]],

where :math:`[i]` indicates that the updates are carried our component wise.
For details, consider `this link <https://optimization.cbe.cornell.edu/index.php?title=AdaGrad>`_.