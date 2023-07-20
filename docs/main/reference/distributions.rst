Distributions
====================================================

This are implemented distributions used for sampling.
The :meth:`logsim.distributions.Distribution` is an abstract class that defines the interface for all distributions and states the required methods.
It is preferred that new distribution implementations inherit from this class and use the frozen distribution to store the distribution type from the :mod:`scipy.stats` module.
These distributions are implemented manually in this module in order to provide additional functionality that enables the distributions to be stored in the database and to be used in the simulation.

.. autoclass:: logsim.distributions.Random
    :members:

.. autoclass:: logsim.distributions.Distribution
    :members:

.. autoclass:: logsim.distributions.Uniform
    :members:

.. autoclass:: logsim.distributions.PERT
    :members:

.. autoclass:: logsim.distributions.PertmGen
    :members:
    :undoc-members:
    :private-members: