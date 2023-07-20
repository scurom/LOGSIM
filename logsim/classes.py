import logging
from dataclasses import dataclass
from datetime import datetime
from math import isnan, ceil

import os
import json
import numpy as np
import pandas as pd
import pint

import logsim.utilities as util
from logsim.classes_salabim import SimulationEnvironment
from logsim.database import Database
from logsim.distributions import Distribution
from logsim.enumerations import LogisticType, ComponentConfigurationError, KPIModes, Modes, Location, NoWindowFoundError
from logsim.simulation_process import InstallationVesselComponent, SupplyVesselComponent, FoundationGenerator
from logsim.weather import WeatherData

logger = logging.getLogger('logsim')


class Vessel:
    """
    Class for defining vessel characteristics. Contains most important vessel characteristics for simulation. Some
    parameters are obligatory, others are optional. The optional parameters are only required for specific vessels.

    :param foundation_capacity: Number of foundations that can be transported
    :param sailing_speed: Sailing speed of vessel as distribution
    :param day_rate: Day rate of vessel
    :param mobilization_rate: The cost of mobilizing the vessel for a project
    :param sailing_cost: Sailing cost of vessel
    :param load_time: Load time of vessel as distribution
    :param mooring_time: Mooring time of vessel as distribution
    :param unmooring_time: Unmooring time of vessel as distribution
    :param within_port_speed: Speed of vessel within port as distribution
    """
    foundation_capacity: int
    sailing_speed: Distribution
    day_rate: tuple
    mobilization_rate: tuple
    sailing_cost: tuple
    load_time: Distribution
    mooring_time: Distribution
    unmooring_time: Distribution
    within_port_speed: Distribution

    def __init__(self, foundation_capacity: int, sailing_speed: Distribution, day_rate: tuple, mobilization_rate: tuple,
                 sailing_cost: tuple, load_time: Distribution, mooring_time: Distribution,
                 unmooring_time: Distribution, within_port_speed: Distribution):
        self.foundation_capacity = foundation_capacity
        self.sailing_speed: Distribution = sailing_speed
        self.load_time: Distribution = load_time  # type: ignore
        self.mooring_time: Distribution = mooring_time  # type: ignore
        self.unmooring_time: Distribution = unmooring_time  # type: ignore
        self.within_port_speed: Distribution = within_port_speed
        if day_rate[1] != 'euro/day':
            raise ComponentConfigurationError("Day rate must be in euro/day")
        self.day_rate = day_rate
        if mobilization_rate[1] != 'euro':
            raise ComponentConfigurationError("Mobilization rate must be in euro")
        self.mobilization_rate = mobilization_rate
        if sailing_cost[1] != 'euro/hour':
            raise ComponentConfigurationError("Sailing cost must be in euro/hour")
        self.sailing_cost = sailing_cost

    def as_dict(self) -> dict:
        """
        Helper function to convert class to dictionary which is required for saving to database.
        Calls correct functions for Distribution and Pint types to get string representation.

        :return: Dictionary with class attributes
        """
        d = {
            "foundation_capacity": self.foundation_capacity,
            "sailing_speed": self.sailing_speed.as_str(),
            "within_port_speed": self.within_port_speed.as_str(),
            "day_rate": self.day_rate.to('euro/day'),
            "mobilization_rate": self.mobilization_rate.to('euro'),
            "load_time": self.load_time.as_str(),
            "mooring_time": self.mooring_time.as_str(),
            "unmooring_time": self.unmooring_time.as_str()
        }
        return d


@dataclass
class Distances:
    """
    Class for defining distances between locations. Contains most important distances for simulation. Need all to be
    defined, but can be 0 if not applicable.

    :param supply_to_intermediate: Distance between supply port and intermediate location
    :param supply_to_wind_farm: Distance between supply port and wind farm
    :param intermediate_to_wind_farm: Distance between intermediate location and wind farm
    :param intermediate_within_port: Distance between intermediate location and port
    :param supply_within_port: Distance between supply port and port
    """
    supply_to_intermediate: pint.Quantity | None = None
    supply_to_wind_farm: pint.Quantity | None = None
    intermediate_to_wind_farm: pint.Quantity | None = None
    intermediate_within_port: pint.Quantity | None = None
    supply_within_port: pint.Quantity | None = None

    def as_dict(self) -> dict:
        """
        Helper function to convert class to dictionary which is required for saving to database.

        :return: Dictionary with class attributes
        """
        d = {
            "supply_to_intermediate": self.supply_to_intermediate.to('km'),
            "supply_to_wind_farm": self.supply_to_wind_farm.to('km'),
            "intermediate_to_wind_farm": self.intermediate_to_wind_farm.to('km')
        }
        # If var type is Quantity convert to float
        for k, v in d.items():
            if isinstance(v, pint.Quantity):
                d[k] = v.magnitude

        return d


class InstallationVessel(Vessel):
    """
    Class representing the installation vessel. Inherits from :class:`Vessel`.

    :param foundation_capacity: The foundation capacity of the vessel
    :param sailing_speed: The sailing speed of the vessel
    :param day_rate: The day rate of the vessel
    :param mobilization_rate: The cost of mobilizing the vessel for a project
    :param sailing_cost: The sailing cost of the vessel
    :param dp_fuel_cost: The DP fuel cost of the vessel
    :param installation_limit_period: The installation limit period of the vessel
    :param installation_limit_wave: The installation limit wave of the vessel
    :param installation_time: The installation time of the vessel
    :param sailing_limit_period: The sailing limit period of the vessel
    :param sailing_limit_wave: The sailing limit wave of the vessel
    :param alongside_limit_period: The alongside limit period of the vessel
    :param alongside_limit_wave: The alongside limit wave of the vessel
    :param load_time: The load time of the vessel
    :param mooring_time: The mooring time of the vessel
    :param unmooring_time: The unmooring time of the vessel
    :param alongside_mooring_time: The alongside mooring time of the vessel
    :param alongside_unmooring_time: The alongside unmooring time of the vessel
    :param alongside_transfer_time: The alongside transfer time of the vessel
    :param within_port_speed: Sailing speed within ports
    """

    def __init__(self, foundation_capacity: int, sailing_speed: Distribution, day_rate: tuple, mobilization_rate: tuple,
                 sailing_cost: tuple, dp_fuel_cost: tuple,
                 installation_limit_period: float, installation_limit_wave: float, installation_time: Distribution,
                 sailing_limit_period: float, sailing_limit_wave: float, alongside_limit_period: float,
                 alongside_limit_wave: float, load_time: Distribution,
                 mooring_time: Distribution, unmooring_time: Distribution, alongside_mooring_time: Distribution,
                 alongside_unmooring_time: Distribution, alongside_transfer_time: Distribution,
                 within_port_speed: Distribution):
        super().__init__(foundation_capacity, sailing_speed, day_rate, mobilization_rate, sailing_cost, load_time,
                         mooring_time, unmooring_time, within_port_speed)
        self.dp_fuel_cost = dp_fuel_cost
        self.installation_limit_period = installation_limit_period
        self.installation_limit_wave = installation_limit_wave
        self.installation_time: Distribution = installation_time
        self.sailing_limit_period = sailing_limit_period
        self.sailing_limit_wave = sailing_limit_wave
        self.alongside_limit_period = alongside_limit_period
        self.alongside_limit_wave = alongside_limit_wave
        self.alongside_mooring_time: Distribution = alongside_mooring_time
        self.alongside_unmooring_time: Distribution = alongside_unmooring_time
        self.alongside_transfer_time: Distribution = alongside_transfer_time

    def as_dict(self) -> dict:
        """
        Helper function to convert class to dictionary which is required for saving to database.
        :return: dictionary with class attributes
        """
        d = super().as_dict()
        d.update(
            {
                "installation_limit_period": self.installation_limit_period,
                "installation_limit_wave": self.installation_limit_wave,
                "installation_time": self.installation_time.as_str(),
                "sailing_limit_period": self.sailing_limit_period,
                "sailing_limit_wave": self.sailing_limit_wave,
                "alongside_limit_period": self.alongside_limit_period,
                "alongside_limit_wave": self.alongside_limit_wave,
                "alongside_mooring_time": self.alongside_mooring_time.as_str(),
                "alongside_unmooring_time": self.alongside_unmooring_time.as_str(),
                "alongside_transfer_time": self.alongside_transfer_time.as_str(),
            })
        for k, v in d.items():
            if isinstance(v, pint.Quantity):
                d[k] = round(v.magnitude, 2)
        return d


class SupplyVessel(Vessel):
    """
        Class representing a supply vessel. Inherits from :class:`Vessel`.

        :param foundation_capacity: The foundation capacity of the vessel
        :param sailing_speed: The sailing speed of the vessel
        :param day_rate: The day rate of the vessel
        :param mobilization_rate: The cost of mobilizing the vessel for a project
        :param sailing_cost: The sailing cost of the vessel
        :param sailing_limit_period: The sailing limit period of the vessel
        :param sailing_limit_wave: The sailing limit wave of the vessel
        :param load_time: The load time of the vessel
        :param mooring_time: The mooring time of the vessel
        :param unmooring_time: The unmooring time of the vessel
        :param within_port_speed: Sailing speed within ports
        """

    def __init__(self, foundation_capacity: int, sailing_speed: Distribution, day_rate: tuple, mobilization_rate: tuple,
                 sailing_cost: tuple, sailing_limit_period: float, sailing_limit_wave: float,
                 load_time: Distribution, mooring_time: Distribution, unmooring_time: Distribution,
                 within_port_speed: Distribution):
        super().__init__(foundation_capacity, sailing_speed, day_rate, mobilization_rate, sailing_cost, load_time,
                         mooring_time, unmooring_time, within_port_speed)
        self.sailing_limit_period = sailing_limit_period
        self.sailing_limit_wave = sailing_limit_wave

    def as_dict(self) -> dict:
        """
        Helper function to convert class to dictionary which is required for saving to database.
        :return: dictionary with class attributes
        """
        d = super().as_dict()
        d.update(
            {
                "sailing_limit_period": self.sailing_limit_period,
                "sailing_limit_wave": self.sailing_limit_wave,
            })
        for k, v in d.items():
            if isinstance(v, pint.Quantity):
                d[k] = round(v.magnitude, 2)
        return d


@dataclass
class Experiment:
    """
    Class for defining experiment characteristics. Contains most important experiment characteristics for simulation.
    Need to pass required vessels of type Vessel. Optional parameters are only required for specific experiments.

    :param installation_vessel: Installation vessel
    :param logistic_configuration: Logistic configuration
    :param distances: Distances between locations
    :param to_install: Number of foundations to install
    :param use_weather: Boolean to indicate if weather should be used
    :param decision_rules: If decision rules should be used or vessels should only look 1 operation ahead.
    :param _sim: Simulation environment
    :param _raw_results: dict
    :param results: list
    :param weather_data: Weather data
    :param intermediate_location_capacity: Capacity of intermediate location
    :param intermediate_location_quays: Number of quays at intermediate location
    :param intermediate_location_stock_minimum: Stock at the intermediate location before starting installation vessel
    :param intermediate_location_cost: Cost of rental of intermediate location in euro's/month
    :param supply_vessel: Offshore barge
    :param supply_vessel_2: Second offshore barge
    """
    installation_vessel: InstallationVessel
    logistic_configuration: LogisticType
    distances: Distances
    to_install: int
    use_weather: bool = True
    decision_rules: bool = True

    intermediate_location_capacity: int | None = None
    intermediate_location_quays: int | None = None
    intermediate_location_stock_minimum: int | None = None
    intermediate_location_cost: tuple | None = None
    results: list | None = None
    weather_data: WeatherData | None = None
    supply_vessel: SupplyVessel | None = None
    supply_vessel_2: SupplyVessel | None = None

    _raw_results: dict | None = None
    _sim: SimulationEnvironment = SimulationEnvironment()

    def __post_init__(self):
        """ Check if all required parameters are defined according to used logistic configuration.
        :class:`.LogisticType` """
        # Create empty dictionary for results
        self._raw_results = {}
        self.results = []

        if self.use_weather and self.weather_data is None:
            raise ComponentConfigurationError("Weather data is required, but not provided")

        match self.logistic_configuration:
            case LogisticType.DIRECT_1:
                if self.distances.supply_to_wind_farm is None:
                    raise ComponentConfigurationError(
                        f"Distance supply to wind farm is not defined, but is required for logistic configuration "
                        f"{self.logistic_configuration}")
            case LogisticType.OFFSHORE_TRANSFER_2:
                if self.distances.supply_to_wind_farm is None:
                    raise ComponentConfigurationError(
                        f"Distance supply to wind farm is not defined, but is required for logistic configuration "
                        f"{self.logistic_configuration}")
                if self.supply_vessel is None:
                    raise ComponentConfigurationError(
                        f"Supply vessel is required for logistic configuration {self.logistic_configuration}")
            case LogisticType.PORT_TRANSFER_3:
                if (self.distances.intermediate_to_wind_farm or self.distances.supply_to_intermediate) is None:
                    raise ComponentConfigurationError(
                        f"Distance intermediate to wind farm or supply to intermediate is not defined, but is required")
                if self.supply_vessel is None:
                    raise ComponentConfigurationError("Supply vessel is required for logistic configuration")
            case LogisticType.PORT_TRANSSHIPMENT_4:
                if (self.distances.intermediate_to_wind_farm or self.distances.supply_to_intermediate) is None:
                    raise ComponentConfigurationError(
                        f"Distance intermediate to wind farm or supply to intermediate is not defined, but is required")
                if self.supply_vessel is None:
                    raise ComponentConfigurationError("Supply vessel is required for logistic configuration")
                if self.intermediate_location_capacity is None or self.intermediate_location_quays is None:
                    raise ComponentConfigurationError(
                        "Intermediate location maximums (quays and capacity) are required for logistic configuration")
                if self.intermediate_location_stock_minimum is None:
                    raise ComponentConfigurationError(
                        "Provide a intermediate_location_stock_minimum to state the minimum required ")
            case LogisticType.OFFSHORE_TRANSFER_BUFFER_5:
                if (self.distances.intermediate_to_wind_farm or self.distances.supply_to_intermediate) is None:
                    raise ComponentConfigurationError(
                        f"Distance intermediate to wind farm or supply to intermediate is not defined, but is required")
                if self.supply_vessel is None:
                    raise ComponentConfigurationError("Supply vessel is required for logistic configuration")
                if self.supply_vessel_2 is None:
                    raise ComponentConfigurationError("Second supply vessel is required for logistic configuration")
                if self.intermediate_location_capacity is None or self.intermediate_location_quays is None:
                    raise ComponentConfigurationError(
                        "Intermediate location maximums (quays and capacity) are required for logistic configuration")
                if self.intermediate_location_stock_minimum is None:
                    raise ComponentConfigurationError(
                        "Provide an 'intermediate_location_stock_minimum' to state the minimum required ")
            case _:
                raise NotImplementedError(
                    f"Logistic configuration {self.logistic_configuration} is not implemented yet")

    def save_sample_result(self, iv: "InstallationVesselComponent", supply_vessel: "SupplyVesselComponent",
                           supply_vessel_2: "SupplyVesselComponent",
                           sample_no: int, run_time: float, sample_error: bool):
        """
        Save sample result for experiment. Sample result contains data of installation vessel and supply vessels.

        :param iv: Installation vessel component run in simulation
        :type iv: "InstallationVesselComponent"
        :param supply_vessel: Barge component run in simulation
        :type supply_vessel: SupplyVesselComponent
        :param supply_vessel_2: Second barge component run in simulation
        :type supply_vessel_2: SupplyVesselComponent
        :param sample_no: Sample number, used for identification
        :type sample_no: int
        :param run_time: Run time of simulation
        :type run_time: float
        :param sample_error: Boolean indicating if sample resulted in an error
        :type sample_error: bool
        """
        if supply_vessel is None:
            supply_data = None
            supply_foundations_data = None
            supply_location_data = None
        else:
            supply_data = supply_vessel.get_mode_data()
            supply_foundations_data = self._sim.env.get_store_length(supply_vessel.foundations_on_board)
            supply_location_data = supply_vessel.get_location_data()

        if supply_vessel_2 is None:
            supply_2_data = None
            supply_2_foundations_data = None
            supply_2_location_data = None
        else:
            supply_2_data = supply_vessel_2.get_mode_data()
            supply_2_foundations_data = self._sim.env.get_store_length(supply_vessel_2.foundations_on_board)
            supply_2_location_data = supply_vessel_2.get_location_data()

        intermediate_location_rental = 0
        if self.logistic_configuration in [LogisticType.PORT_TRANSSHIPMENT_4, LogisticType.OFFSHORE_TRANSFER_BUFFER_5]:
            intermediate_location_rental = self.intermediate_location_cost.to('euro/month').magnitude * ceil(
                run_time / 730.48)

        # Empty result for sample
        sample_result = {
            'sample_no': sample_no,
            'iv_data': iv.get_mode_data(),
            'iv_location_data': iv.get_location_data(),
            'supply_data': supply_data,
            'supply_location_data': supply_location_data,
            'supply_2_data': supply_2_data,
            'supply_2_location_data': supply_2_location_data,
            'iv_foundations': self._sim.env.get_store_length(iv.foundations_on_board),
            'supply_foundations': supply_foundations_data,
            'supply_2_foundations': supply_2_foundations_data,
            'project_duration': run_time,
            'error_occurred': sample_error,
            'intermediate_location_rental': intermediate_location_rental
        }
        self._raw_results[sample_no] = sample_result

    def calculate_kpis(self, iv=None, barge=None, barge_2=None, save_db=False, save_json=False, experiment_id=None,
                       experiment_error=None, batch_datetime='Undefined') -> None:
        """
        After simulation, calculate KPIs for each sample and save to database if requested.

        :param iv: Installation vessel component run in simulation
        :param barge: Barge component run in simulation
        :param barge: Barge component run in simulation
        :param barge_2: Second barge component run in simulation
        :param save_db: Save results to database
        :param experiment_id: Experiment id to use when saving results in database
        :param experiment_error: Boolean indicating if experiment resulted in an error (if one of the samples resulted in an error)
        :param batch_datetime: Datetime that can be passed to group multiple experiments together with a common datetime
            reflecting the batch it was run in (for example multiprocessing)
        :param save_db: Boolean indicating if results should be saved to database
        :param save_json: Boolean indicating if results should be saved to json
        :param experiment_id: Experiment id to use when saving results in database
        :param experiment_error: Boolean indicating if experiment resulted in an error (if one of the samples resulted
            in an error)
        :param batch_datetime: Datetime that can be passed to group multiple experiments together with a common datetime
            reflecting the batch it was run in (for example multiprocessing)
        """

        def add_duration(columns):
            for column in columns:
                if sample[column] is not None:
                    diff = np.diff(sample[column][:, 0], axis=0)
                    diff = np.array([np.append(diff, 0)]).T
                    sample[column] = np.append(sample[column], diff, axis=1)

        def create_kpi_object(df_data, df_loc_data, sample_number, vessel):
            res_kpi = {
                'sample_no': sample_number,
                'waiting_for_weather': df_data[df_data['mode'].isin(KPIModes.WAITING_FOR_WEATHER.value)][
                    'duration'].sum(),
                'waiting_other': df_data[df_data['mode'].isin(KPIModes.WAITING_OTHER.value)]['duration'].sum(),
                'waiting': df_data[df_data['mode'].isin(KPIModes.WAITING.value)]['duration'].sum(),
                'operation': df_data[df_data['mode'].isin(KPIModes.OPERATING.value)]['duration'].sum(),
                'sailing': df_data[df_data['mode'].isin(KPIModes.SAILING.value)]['duration'].sum(),
                'sailing_fuel_cost': df_data[df_data['mode'].isin(KPIModes.SAILING.value)][
                                         'duration'].sum() * vessel.parameters.sailing_cost.to('euro/hour').magnitude,
                'offshore_sailing': df_data[df_data['mode'] == Modes.SAILING]['duration'].sum(),
                'waiting_weather_installing': df_data[df_data['mode'] == Modes.WAITING_WEATHER_INSTALLING][
                    'duration'].sum(),
                'waiting_weather_sailing': df_data[df_data['mode'] == Modes.WAITING_WEATHER_SAILING]['duration'].sum(),
                'waiting_weather_transfer': df_data[df_data['mode'] == Modes.WAITING_WEATHER_TRANSFER][
                    'duration'].sum(),
                'not_docked_duration': df_data['start'].max() - df_data[df_data['mode'] != Modes.DOCKED]['start'].min(),
                'operating_fuel_cost': 0,
                'vessel_usage_cost': (df_data['start'].max() - df_data[df_data['mode'] != Modes.DOCKED][
                    'start'].min()) * vessel.parameters.day_rate.to('euro/hour').magnitude +
                                     vessel.parameters.mobilization_rate.to('euro').magnitude,
                'time_at_supply': df_loc_data[df_loc_data['location'] == Location.SUPPLY_PORT]['duration'].sum(),
                'time_at_intermediate': df_loc_data[df_loc_data['location'] == Location.INTERMEDIATE_LOCATION][
                    'duration'].sum(),
                'time_at_wind_farm': df_loc_data[df_loc_data['location'] == Location.WIND_FARM]['duration'].sum(),
                'time_offshore': df_loc_data[df_loc_data['location'] == Location.OFFSHORE]['duration'].sum()}

            for k, v in res_kpi.items():
                if isnan(v):
                    res_kpi[k] = 0

            if hasattr(vessel.parameters, 'dp_fuel_cost'):
                res_kpi['operating_fuel_cost'] = (df_data[df_data['mode'].isin(KPIModes.ON_DP.value)]['duration'].sum()
                                                  * vessel.parameters.dp_fuel_cost.to('euro/hour').magnitude)

            res_kpi['total_vessel_cost'] = res_kpi['sailing_fuel_cost'] + res_kpi['operating_fuel_cost'] + res_kpi[
                'vessel_usage_cost']

            res_kpi = {k: round(v, 2) for k, v in res_kpi.items()}
            # res_kpi = {k: round(v, 2) for k, v in res_kpi.items() if }
            return res_kpi

        # Add durations for each sample and calculate KPI. Add each sample to results list
        results = []
        for sample in self._raw_results.values():
            sample_dic = {}
            sample_no = sample['sample_no']
            add_duration(['iv_data', 'iv_location_data', 'supply_data', 'supply_location_data', 'supply_2_data',
                          'supply_2_location_data', 'iv_foundations', 'supply_foundations', 'supply_2_foundations'])

            df_iv = pd.DataFrame(sample['iv_data'], columns=['start', 'mode', 'duration'])
            df_iv_loc = pd.DataFrame(sample['iv_location_data'], columns=['start', 'location', 'duration'])[
                ['location', 'duration']].groupby('location').sum().reset_index()

            sample_dic['iv'] = create_kpi_object(df_iv, df_iv_loc, sample_no, iv)

            if sample['supply_data'] is not None:
                df_sup = pd.DataFrame(sample['supply_data'], columns=['start', 'mode', 'duration'])
                df_sup_loc = pd.DataFrame(sample['supply_location_data'], columns=['start', 'location', 'duration'])[
                    ['location', 'duration']].groupby('location').sum().reset_index()
                sample_dic['supply'] = create_kpi_object(df_sup, df_sup_loc, sample_no, barge)

            if sample['supply_2_data'] is not None:
                df_sup = pd.DataFrame(sample['supply_2_data'], columns=['start', 'mode', 'duration'])
                df_sup_loc = pd.DataFrame(sample['supply_2_location_data'], columns=['start', 'location', 'duration'])[
                    ['location', 'duration']].groupby('location').sum().reset_index().fillna(0)
                sample_dic['supply_2'] = create_kpi_object(df_sup, df_sup_loc, sample_no, barge_2)

            # Add global parameters
            sample_dic['error_occurred'] = sample['error_occurred']
            sample_dic['project_duration'] = sample['project_duration']
            sample_dic['intermediate_location_rental'] = sample['intermediate_location_rental']

            results.append(sample_dic)

        self.results = results

        # Add vessel information
        if self.supply_vessel is not None:
            supply_data = self.supply_vessel.as_dict()
        else:
            supply_data = None

        if self.supply_vessel_2 is not None:
            supply_2_data = self.supply_vessel_2.as_dict()
        else:
            supply_2_data = None

        # Create big dictionary with all experiment parameters and results
        res_dict = {
            'experiment_id': experiment_id,
            'batch_datetime': batch_datetime,
            'distances': self.distances.as_dict(),
            'logistic_configuration': self.logistic_configuration.name,
            'use_weather': self.use_weather,
            'to_install': self.to_install,
            'intermediate_location_stock_minimum': self.intermediate_location_stock_minimum,
            'intermediate_location_capacity': self.intermediate_location_capacity,
            'intermediate_location_cost': self.intermediate_location_cost.to('euro/month').magnitude,
            'weather_data': {
                'file_name': self.weather_data.file_name,
                'no_samples': int(self.weather_data.no_samples),
                'sample_hours': self.weather_data.sample_hours,
                'start_day': self.weather_data.start_day,
                'start_month': self.weather_data.start_month,
                'synthetic': self.weather_data.synthetic,
                'scale_factor': self.weather_data.scale_factor
            },
            'installation_vessel': self.installation_vessel.as_dict(),
            'supply_vessel': supply_data,
            'supply_vessel_2': supply_2_data,
            'results': self.results,
            'run_datetime': datetime.today().replace(microsecond=0),
            'error_occurred': experiment_error
        }

        if save_db:
            # Save to database
            logger.info("Insert into database")
            Database.insert("results", res_dict)
        if save_json:
            file_name = f"results/{experiment_id}.json"
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, 'w') as f:
                json.dump(res_dict, f, indent=4, default=str)

    def run_experiment(self, animate=False, trace=False, plot=False, save_db=False, save_json=False, experiment_id=None,
                       batch_datetime='Undefined') -> None:
        """
        Run the experiment. Main method to be called. First sets the experiment to the simulation environment (
        :class:`SimulationEnvironment`). Then will go over each sample of the weather_data (either synthetic or not)
        and run the experiment for each sample by creating a new salabim environment, setting the vessel parameters,
        weather sample, creating the components and running the simulation. The results are then saved at each
        simulation run in the :attr:`_raw_results` dictionary. After all samples are run, the results are calculated
        and saved to the :attr:`results` dictionary by calling the appropriate functions.

        :param animate: Animate the simulation
        :param trace: Trace the simulation
        :param plot: Plot the results of the last simulation run
        :param save_db: Save the results to the DB (set in database.py)
        :param experiment_id: Experiment ID that can be used to group multiple runs together with a common id in the DB
        :param batch_datetime: Datetime that can be passed to group multiple experiments together with a common datetime
            reflecting the batch it was run in (for example multiprocessing)
        """
        iv, barge, barge_2 = None, None, None
        self._sim.set_experiment(self)
        experiment_error = False

        if save_json and experiment_id is None:
            raise ValueError("Please provide an experiment_id when saving to json")

        # Run for X samples
        for sample_no in range(1, self.weather_data.no_samples + 1):
            sample_error = False
            logging.debug(f"Start simulation run {sample_no}")

            self._sim.create_salabim_env(trace=trace)

            # If weather is used, set the sample
            if self._sim.experiment.use_weather:
                self._sim.set_weather_sample(sample_no)

            # Create foundations at supply port
            FoundationGenerator(suppress_trace=True)

            # Create components
            iv = InstallationVesselComponent(self.installation_vessel)
            match self.logistic_configuration:
                case LogisticType.DIRECT_1:
                    barge = None
                    barge_2 = None
                case LogisticType.OFFSHORE_TRANSFER_2:
                    barge = SupplyVesselComponent(self.supply_vessel)
                    barge_2 = None
                case LogisticType.PORT_TRANSFER_3:
                    barge = SupplyVesselComponent(self.supply_vessel)
                    barge_2 = None
                case LogisticType.PORT_TRANSSHIPMENT_4:
                    barge = SupplyVesselComponent(self.supply_vessel)
                    barge_2 = None
                case LogisticType.OFFSHORE_TRANSFER_BUFFER_5:
                    barge = SupplyVesselComponent(self.supply_vessel)
                    barge_2 = SupplyVesselComponent(self.supply_vessel_2, supply_vessel_number=2)
                case _:
                    raise ValueError("Logistic configuration not implemented")

            if animate:
                util.animate(self._sim, iv, barge, barge_2)

            try:
                self._sim.env.run()
            # Exception NotWindowFoundError is raised when no window is available, can be due to the fact that the
            # simulation takes too long, silence this error as we don't want to stop the simulation, instead an
            # error flag is added to the results.
            except NoWindowFoundError:
                logger.warning("No window check available, Exception")
                experiment_error, sample_error = True, True
            except Exception:
                logger.warning("Unexpected Exception")
                raise Exception

            if (self._sim.env.shared_data['left_to_install'] > 0) and not sample_error:
                raise Exception("Missing some exception or error")

            if plot:
                util.plot(self._sim, iv, barge, sample_no, barge_2)

            # proces results
            self.save_sample_result(iv, barge, barge_2, sample_no, self._sim.env.now(), sample_error)

        if save_db:
            if Database.DATABASE is None:
                Database.initialize()
            else:
                logger.info("DB already initialized, will use created client pool")

        self.calculate_kpis(iv=iv, barge=barge, barge_2=barge_2, save_db=save_db, save_json=save_json,
                            experiment_id=experiment_id,
                            experiment_error=experiment_error, batch_datetime=batch_datetime)
