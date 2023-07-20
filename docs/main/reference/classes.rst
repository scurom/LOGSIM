Classes
====================================================

This are overwritten salabim classes and self created classes mainly used to run the simulation
and set correct parameters for the simulation.

The classes Vessel, InstallationVessel and SupplyVessel have the goal of providing a structured way to store and enter parameters for the vessels.
The most important class is the Experiment class as it provides the main simulation run and contains methods to transform and clean the results of the runs before storing it to the database.
The main method used is the :meth:`logsim.classes.Experiment.run_experiment` method.
This method will go over each of the weather samples and create a seperate salabim environment for each sample.
It will generate the vessel components and run the appropriate supply chain configuration.
It will then save the sample in the experiment class with the :meth:`logsim.classes.Experiment.save_sample_result` method.
After each sample is run it will call the :meth:`logsim.classes.Experiment.calculate_kpis` method to clean the results, calculate the overall KPI's and store it in the database by calling the :meth:`logsim.db.Database.insert` method.
Each entry in the database will be one run with the corresponding configuration and parameters, it contains the durations of each sample and the overall KPI's.
It is also possible to write the results to a JSON file instead of a database by setting the save\_json parameter to True in the logsim.classes.Experiment.run\_experiment() method.

.. autoclass:: logsim.classes.Vessel
    :members:

.. autoclass:: logsim.classes.InstallationVessel
    :members:

.. autoclass:: logsim.classes.SupplyVessel
    :members:

.. autoclass:: logsim.classes.Experiment
    :members:


