.. BeeRS documentation master file, created by
   sphinx-quickstart on Tue Jul 16 15:39:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BeeRS documentation
===================

**BeeRS** (**Be**\ st Int\ **e**\ rvention for **R**\ ecommender **S**\ ystems) is a Python library to optimize opinions in social networks under the 
Friedkin-Johnsen (FJ) opinion dynamics :cite:`FJ` with regard to a leaders's
objective. Specifically, it computes (locally) optimal network weights under convex constraints.
Formally, this problem is stated as

.. math::
   \DeclareMathOperator*{\argmin}{\arg\!\min\,}
   \begin{aligned}
   & \min_{w,y} && \varphi(w,y) \\
   & \text{s.t.} && A(w) y = s \\
   & && w \in \mathcal{W},
   \end{aligned}

where :math:`w` corresponds to the network weights, :math:`y` denotes the opinions of the users,
:math:`s` contains the internal opinions of the users, :math:`\varphi` is the leaders' continuously differentiable
objective function, 
and :math:`\mathcal{W}` is a convex, closed, and non-empty set, restricting the possible interventions.

The matrix :math:`A(w) \in \mathbb{R}^{n \times n}` is given by

.. math::

   A(w) =
   \begin{bmatrix}
       1 + \sum_{j \in \mathcal{N}_1} w_{1j} & \cdots & -w_{1n} \\
       \vdots & \ddots & \vdots \\
       -w_{n1} & \cdots & 1 + \sum_{j \in \mathcal{N}_n} w_{nj}
   \end{bmatrix}

and captures the FJ dynamics.

.. admonition:: FJ Opinion Dynamics Model

   In the FJ model :cite:`FJ`, every user :math:`i \in \mathcal{V}` has an internal opinions :math:`s_i \in \mathbb{R}`, which is kept
   private, and a time-varying external opinion :math:`y_i \in \mathbb{R}`, which is shared with peers. The set 
   :math:`\mathcal{V}` contains the users of the social network. The external
   opinion evolves according to the update 

   .. math::
      
      y_i^{(k+1)} = \frac{s_i + \sum_{j \in \mathcal{N}_i} w_{ij} y_j^{(k)}}{1 + \sum_{j \in \mathcal{N}_i} w_{ij}},

   where :math:`w_{ij}` represents the weight of the connection from user :math:`i` to user :math:`j`. The weight 
   represents how susceptible user :math:`i` is to the opinion of user :math:`j`. The set :math:`\mathcal{N}_i` 
   represents the neighborhood of user :math:`i`, i.e., 
   :math:`\mathcal{N}_i = \{j \mid w_{ij} > 0, \, \forall j \in \mathcal{V} \setminus i \}`.
   There are various slightly different formulations of the FJ dynamics, we employ one used in :cite:`BINDEL2015248`.
   It can be shown that this update converges to a unique equilibrium if :math:`w_{ij} \geq 0, \forall i,j \in \mathcal{V}`
   with :math:`i \neq j`.

Due to the constraint :math:`A(w)y = s`, this problem is non-convex and inherently difficult to solve. While
traditional solvers can find solutions, the running time is prohibitive even for
small-to-medium-scale problems. **BeeRS** bridges this gap by providing a flexible, user-friendly, and
highly scalable approach.

A quick guide to getting started with a minimal, running example can be found in :doc:`start`.
A more in-depth analysis of the problem is given in :doc:`problem`, followed by a more
sophisticated example in :doc:`opinions`. Lastly, a complete documentation of the code is given in :doc:`documentation`.

.. admonition:: Key Features of **BeeRS**

   - **Flexibility**: customize :math:`\varphi` to your liking - as long as PyTorch can take gradients you are good to go!

   - **Scalability**: solve networks with thousands of nodes in seconds!

   - **User friendliness**: out-of-the-box solution to your problem!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   start
   problem
   opinions
   documentation
   references
   license
