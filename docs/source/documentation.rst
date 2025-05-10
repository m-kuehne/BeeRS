Documentation
=============

BeeRS Module
------------

**BeeRS** consists of two classes. The first class, :ref:`BeersCpu <beers-cpu-class>`, 
is based on SciPy, NumPy, PyTorch, and CVXpy, and performs computations on the CPU. It 
supports both sparse and dense adjacency matrices. The projection is done with CVXpy and
therefore extremely flexible. Additionally, a custom projection can be used.

The second calss, :ref:`BeersGpu <beers-gpu-class>`, is mainly based on JAX and jaxopt,
with some NumPy dependencies. It carries out (most) computations on the GPU (if available).
It only supports sparse adjacency matrices. The projection is done with 
`jaxopt.BoxOSQP <https://jaxopt.github.io/stable/_autosummary/jaxopt.BoxOSQP.html>`_, thus
more restrictive than with the CPU version. Further, no custom projection is supported.

We made the observation that the GPU version is generally slower than the CPU version due 
to the large overhead of JAX. However, we do not rule out that the GPU version might be
faster for very large problems.

Classes
-------

.. _beers-cpu-class:

.. autoclass:: BeersCpu.BeersCpu
    :members:

.. _beers-gpu-class:

.. autoclass:: BeersGpu.BeersGpu
    :members:
