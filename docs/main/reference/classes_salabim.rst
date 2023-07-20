Salabim Classes
====================================================

This are overwritten salabim classes and self created classes mainly used to run the simulation
and set correct parameters for the simulation.
The salabim environment class is based on the sim.Environment class and is used to run the simulation.
It contains the required stores and data storage for each simulation run.
It also provides methods that can be used to let the simulation interact with the weather module.
The Component class adds some additional monitoring functionality in order to store results correctly.
The SimulationEnvironment class is used to store the current run experiment, it contains a salabim environment
which is overwritten at each run (to reset all resources and the clock to 0).

.. autoclass:: logsim.classes_salabim.SalabimEnvironment
    :members:

.. autoclass:: logsim.classes_salabim.SimulationEnvironment
    :members:

.. autoclass:: logsim.classes_salabim.Component
    :members:

