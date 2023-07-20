Introduction
====================================================

Main purpose of this package is to experiment with different logistical configurations
and vessel parameters to analyze KPI's impact when installing offshore wind farm foundations.
The package is set up to work in according to the main configurations analyzed for the
thesis this package was developed for. The inner workings of the model however can be easily adjusted to suit
particular needs. The main processes within the simulation are modelled in classes
:py:class:`logsim.simulation_process.InstallationVesselComponent` and :py:class:`logsim.simulation_process.SupplyVesselComponent`.
The first class models the installation vessel with its parameters and processes and the second class
models the offshore barge with its parameters and processes. The two vessels interact with each other
by using stores and resources. The stores are used to store the foundations and the resources are used to model the
quays. When transfering components ship-to-ship the barge places itself in a store which can be retrieved by the
installation vessel.

In order to run a simulation first an experiment needs to be setup. First import the required classes as shown below.

.. code:: python

    from logsim.classes import Experiment, Vessel, Distances, WeatherData
    from logsim.enumerations import VesselType, LogisticType

Then the vessels can be modeled one by one by using the Vessel class.

.. code:: python

    # Set seed before creating experiment
    Random.set_seed(2000)

    installation_vessel = InstallationVessel(
        foundation_capacity=3,
        sailing_speed=PERT(5.02, 8.99, 8.11, 'knots'),
        within_port_speed=PERT(4.51, 8.13, 5.53, 'knots'),
        day_rate=(150000, 'euro/day'),
        mobilization_rate=(600000, 'euro'),
        sailing_cost=(796.12, 'euro/hour'),
        dp_fuel_cost=(358.25, 'euro/hour'),
        installation_time=PERT(5.53, 13.13, 7.07, 'h'),
        installation_limit_wave=2.5,
        installation_limit_period=8,
        sailing_limit_period=8,
        sailing_limit_wave=3.5,
        alongside_limit_wave=1.5,
        alongside_limit_period=8,
        load_time=PERT(1.72, 4.28, 2.45, 'h'),
        mooring_time=PERT(45, 85, 60, 'min'),
        unmooring_time=PERT(45, 85, 60, 'min'),
        alongside_mooring_time=PERT(45, 90, 60, 'min'),
        alongside_unmooring_time=PERT(45, 90, 60, 'min'),
        alongside_transfer_time=PERT(1.72, 4.28, 2.45, 'h')
    )

Also the weather data needs to be stated, just as the distances and logistical configuration.
The first time a new weather dataset is used the markov model needs to be trained, which can be done by setting the train_model parameter to True.
In runs after the first run the markov model can be used from the cache by setting the train_model parameter to False, this saves training time.

.. code:: python

    weather = WeatherData(
        file_name="data/weather_data.csv",
        start_day=1,
        start_month=1,
        synthetic=True,
        synthetic_data_samples=100,
        train_model=True
    )

    distances = Distances(
        supply_to_wind_farm=(355, 'km'),
        supply_to_intermediate=(460, 'km'),
        intermediate_to_wind_farm=(260, 'km'),
        intermediate_within_port=(5.39, 'km'),
        supply_within_port=(15.32, 'km')
    )

Then the experiment object can be created and the simulation can be run.

.. code:: python

    # Define experiment class
    experiment = Experiment(
        installation_vessel=installation_vessel,
        logistic_configuration=LogisticType.OFFSHORE_TRANSFER_2,
        supply_vessel=barge,
        supply_vessel_2=barge,
        intermediate_location_capacity=10,
        intermediate_location_quays=1,
        intermediate_location_cost=(250000, 'euro/month'),
        intermediate_location_stock_minimum=6,
        distances=distances,
        to_install=30,
        weather_data=weather
    )

    # Run Experiment
    experiment.run_experiment(save_db=True)

It is ofcourse possible to place this code in a loop to run multiple experiments with different parameters.
It is also possible to run multiple experiments in parallel by using the multiprocessing package, in this case the experiment class needs to be redefined each time before the run_experiment function is called in order to create a new instance of the class (avoids interference between processes).

Use the reference to see the full documentation of the different classes and the parameters that can be passed to the
:py:func:`logsim.classes.Experiment.run_experiment` function.
