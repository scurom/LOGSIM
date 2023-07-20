Simulation Process
====================================================

This module contains the process flow that is used by the simulation environment to simulate the installation and supply process.
The flows contain the operational durations, random draws from distributions and the logic that is used to determine next activities.
Each component is a class and contains the main process flow of that component in the process method.
According to the vessel parameters and the supply chain configuration that is considered in the experiment the correct flows are called.
This module also enables interaction and communication between the different components.
All common processes that are used by both the installation vessel and the supply vessel are defined in the :class:`logsim.simulation_process.SharedProcesses` class in order to reduce code duplication.


.. autoclass:: logsim.simulation_process.FoundationGenerator
    :members:

.. autoclass:: logsim.simulation_process.SharedProcesses
    :members:

.. autoclass:: logsim.simulation_process.InstallationVesselComponent
    :members:

.. autoclass:: logsim.simulation_process.SupplyVesselComponent
    :members:


