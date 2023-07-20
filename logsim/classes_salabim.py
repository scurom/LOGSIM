# Use type checking constant to avoid circular imports and only import for type checking
import logging
import sys
from typing import TYPE_CHECKING
from typing import Union

import numpy as np
import salabim as sim

from logsim.enumerations import Stores, Resources, NoWindowFoundError
from logsim.utilities import UREG
from logsim.weather import get_next_window, get_next_window_3_limits, get_next_window_2_limits

if TYPE_CHECKING:
    from logsim.classes import Experiment


class SalabimEnvironment(sim.Environment):
    """
    Salabim environment is created every simulation run. It is instantiated with the experiment class and the
    necessary stores and resources. It also contains the shared data dictionary, which is used to share data between
    the different components of the simulation. Components will register themselves with the environment, so that
    they can be accessed by other components. The :class:`SimulationEnvironment` includes methods that will set
    the weather sample variable in this class, which is used to determine the weather conditions for the simulation.

    :param experiment: Experiment class that contains the experiment parameters
    :param trace: Boolean that determines whether the simulation should be traced
    :param args: sim.Environment arguments
    :param kwargs: sim.Environment keyword arguments
    """

    stores: Stores
    resources: Resources
    shared_data: dict
    weather_sample: "np.ndarray | None"
    weather_sample_lazyframe: "LazyFrame | None"
    experiment: "Experiment"
    use_weather: bool
    experiment_no: int | None
    no_samples: int | None
    sample_hours: int | None
    components: dict[str]

    def __init__(self, experiment, trace, *args, **kwargs):
        super().__init__(trace, *args, **kwargs)
        self.stores: Stores = Stores(
            sim.Store("supply_vessels_alongside"),
            sim.Store("foundations_at_supply"),
            sim.Store("foundations_at_intermediate"))
        self.resources: Resources = Resources(sim.Resource("intermediate_quay"))
        self.shared_data: dict = {"left_to_install": 0}
        self.weather_sample: "LazyFrame | None" = None
        self.experiment: "Experiment" = experiment
        self.use_weather: bool = experiment.use_weather
        self.decision_rules: bool = experiment.decision_rules
        self.experiment_no: int | None = None
        self.no_samples: int | None = None
        self.sample_hours: int | None = None
        self.components: dict[str] = {}

    @staticmethod
    def get_store_length(store: sim.Store) -> np.array:
        """Return store length by stacking time and length of store in numpy array"""
        store_monitor = store.length
        return np.column_stack((store_monitor._t, store_monitor._x))

    def get_next_window(self, window_size: int | float, hm0_limit: float, tp_limit: float,
                        start_time_check: float = None) -> float:
        """
        Get waiting time until next window of weather data is available. Calls the
        :func:`logsim.weather.get_next_window` function of the weather module.

        :param window_size: The size of the window
        :param hm0_limit: The limit for the wave height
        :param tp_limit: The limit for the wave period
        :param start_time_check: The time from which the waiting time should be calculated

        :return: The waiting time until the next window of weather data is available in hours
        """

        if start_time_check is None:
            start_time_check = self.now()

        if self.weather_sample is None:
            raise ValueError("Weather sample is not set")
        # If not to max need to increase
        elif 10000 > self.now() > self.sample_hours:
            raise NoWindowFoundError(
                "Simulation time is longer than weather data that is available, increase synthetic data hours")
        try:
            waiting_time = get_next_window(self.weather_sample, start_time_check, window_size, hm0_limit,
                                           tp_limit)
        except:
            raise NoWindowFoundError("Cannot find an available window, not enough data available or other error")

        assert waiting_time >= 0
        return waiting_time

    def get_next_window_3_limits(self, window_sizes: list[int | float], hm0_limits: list[float],
                                 tp_limits: list[float], start_time_check: float = None) -> float:
        if start_time_check is None:
            start_time_check = self.now()
        if self.weather_sample is None:
            raise ValueError("Weather sample is not set")
        # If not to max need to increase
        elif 10000 > self.now() > self.sample_hours:
            raise NoWindowFoundError(
                "Simulation time is longer than weather data that is available, increase synthetic data hours")
        try:
            waiting_time = get_next_window_3_limits(self.weather_sample, start_time_check, window_sizes, hm0_limits,
                                                    tp_limits)
        except:
            raise NoWindowFoundError("Cannot find an available window, not enough data available or other error")

        assert waiting_time >= 0
        return waiting_time

    def get_next_window_2_limits(self, window_sizes: list[int | float], hm0_limits: list[float],
                                 tp_limits: list[float], start_time_check: float = None) -> float:
        if start_time_check is None:
            start_time_check = self.now()
        if self.weather_sample is None:
            raise ValueError("Weather sample is not set")
        # If not to max need to increase
        elif 10000 > self.now() > self.sample_hours:
            raise NoWindowFoundError(
                "Simulation time is longer than weather data that is available, increase synthetic data hours")
        try:
            waiting_time = get_next_window_2_limits(self.weather_sample, start_time_check, window_sizes, hm0_limits,
                                                    tp_limits)
        except:
            raise NoWindowFoundError("Cannot find an available window, not enough data available or other error")

        assert waiting_time >= 0
        return waiting_time


class SimulationEnvironment:
    """
    Simulation Environment is the main environment used to run all simulations. It is used to hold the environment
    that is currently run and contains the unit registry. If multiple experiments need to be run in parallel for each
    of them a SimulationEnvironment is created that will on its turn create a SalabimEnvironment for every run.
    """

    env: SalabimEnvironment | None
    ureg: UREG
    experiment_no: int
    to_install: int | None
    no_samples: int | None
    sample_hours: int | None
    experiment: "Experiment"
    shared_data: dict["str", "int"]
    results = dict[str, np.array] | None

    def __init__(self):
        self.env = None
        self.ureg = UREG
        self.experiment_no = 0
        self.to_install = None
        self.no_samples = None
        self.sample_hours = None
        self.results = None

        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)

        # Define own quantities
        self.ureg.define('euro = [cost] = EUR')

        if sys.version_info < (3, 10):
            raise Exception("Python 3.10 or a more recent version is required.")

    def set_experiment(self, experiment: "Experiment"):
        """
        Set experiment and set the quantities that are used in the experiment. Transform all quantities to the
        correct units by using the :func:`logsim.classes.SimulationEnvironment.set_quantity` function that creates
        ureg quantity objects.

        :param experiment: Experiment class that contains the experiment parameters
        """
        self.experiment = experiment
        self.to_install = experiment.to_install

        if experiment.use_weather:
            self.no_samples = experiment.weather_data.no_samples
            self.sample_hours = experiment.weather_data.sample_hours

        for name, var in experiment.installation_vessel.__dict__.items():
            if type(var) == tuple:
                setattr(experiment.installation_vessel, name, self.set_quantity(var[0], var[1]))

        for name, var in experiment.distances.__dict__.items():
            if type(var) == tuple:
                setattr(experiment.distances, name, self.set_quantity(var[0], var[1]))

        for name, var in experiment.__dict__.items():
            if type(var) == tuple:
                setattr(experiment, name, self.set_quantity(var[0], var[1]))

        # If offshore barge
        if experiment.supply_vessel is not None:
            for name, var in experiment.supply_vessel.__dict__.items():
                if type(var) == tuple:
                    setattr(experiment.supply_vessel, name, self.set_quantity(var[0], var[1]))

    def create_salabim_env(self, trace: bool = False, *args, **kwargs):
        """
        Every run this function is called to create a new SalabimEnvironment that is used to run the simulation. Old
        SalabimEnvironment is overwritten. Will be saved as within the main run function:
        :func:`logsim.classes.Experiment.run_experiment` the :func:`logsim.classes.Experiment.save_sample_results` is
        called.

        :param trace: Boolean that indicates if the simulation should be traced
        :param args: Arguments that are passed to the SalabimEnvironment
        :param kwargs: Keyword arguments that are passed to the SalabimEnvironment
        """
        self.env = SalabimEnvironment(self.experiment, trace, *args, **kwargs)
        self.env.shared_data["left_to_install"] = self.to_install
        self.env.experiment_no = self.experiment_no
        if self.experiment.intermediate_location_capacity is not None:
            self.env.stores.foundations_at_intermediate.set_capacity(self.experiment.intermediate_location_capacity)
        if self.experiment.intermediate_location_quays is not None:
            self.env.resources.intermediate_quay.set_capacity(self.experiment.intermediate_location_quays)

    def set_weather_sample(self, sample_no: int):
        """Called when weather is used and a new weather sample is needed. Sets the weather sample in the
        SalabimEnvironment. Each environment has its own weather sample.

        :param sample_no: Integer that indicates the weather sample number
        """
        if self.experiment.use_weather is False:
            raise ValueError("Weather is not used so cannot set weather sample for salabim environment")

        self.env.weather_sample, self.env.weather_sample_lazyframe = self.experiment.weather_data.get_sample(sample_no)

        self.env.weather_sample_no = sample_no
        self.env.no_samples = self.no_samples
        self.env.sample_hours = self.sample_hours

    def set_quantity(self, value: float, unit: str) -> float:
        """
        Function to set a quantity in the correct unit. Will return the quantity in hours if the dimension is time.
        This is because in the simulation hours as the base unit for time is used. All other units are converted to the
        base unit.

        :param value: Float that contains the value of the quantity
        :param unit: String that contains the unit of the quantity
        :return: Float that contains the value of the quantity in the correct unit (h if time, base unit otherwise)
        """
        quant = self.ureg.Quantity(value, unit)
        if quant.check('[time]'):
            return quant.to('h').magnitude
        else:
            return quant.to_base_units()


class Component(sim.Component):
    """
    Base class for all components. Contains some manual monitors, like mode (as sim.Monitor) and location (as
    sim.State). Manually implemented these to allow for correct typing and enforce usage of 'int8' type for storing
    the modes/states. This saves a lot of memory.
    """
    env: SalabimEnvironment
    mode: sim.Monitor
    location: sim.State

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode: sim.Monitor = sim.Monitor(name=self.name() + ".mode", level=True, initial_tally=0, env=self.env,
                                             type='int8')
        self.location: sim.State = sim.State("location", value=0, type='int8', env=self.env)
        self.env.components[self.name()] = self

    def from_store_item(self) -> sim.Component:
        """
        Function to retrieve store item. Overwritten to enforce correct typing and enforce never returns None.

        :return sim.Component: Component that is retrieved from the store
        """
        item = super().from_store_item()
        assert item is not None
        return item

    def get_mode_data(self) -> np.array:
        """
        Returns a numpy array with the mode data from simulation objects

        :return np.ndarray: Numpy array with mode data in form of 'time', 'mode value'
        """
        return np.column_stack(self.mode.tx(add_now=False))

    def get_location_data(self) -> np.array:
        """
        Returns a numpy array with the location data from simulation objects

        :return np.ndarray: Numpy array with location data in form of 'time', 'location value'
        """
        return np.column_stack(self.location.value.tx(add_now=False))

    def from_store_mode(self, store: sim.Store, mode: int) -> sim.Component:
        """
        Function to retrieve store item with integer mode (based on :class:`logsim.enumerations.Modes`) from store.

        :param store: Store that is used to retrieve the item from
        :param mode: Integer that indicates the mode of the component
        :type mode: int (:class:`logsim.enumerations.Modes`)
        :return sim.Component: Component that is retrieved from the store
        """
        return self.from_store(store, mode=mode)  # type: ignore

    def hold_mode(self, duration: Union[float, int], mode: int) -> None:
        """
        Function to hold component with integer mode (based on :class:`logsim.enumerations.Modes`) for a certain
        duration.

        :param duration: Duration that the component is held
        :param mode: Integer that indicates the mode of the component
        :type mode: int (:class:`logsim.enumerations.Modes`)
        """
        self.hold(duration, mode=mode)  # type:ignore

    def wait_mode(self, state: sim.State, mode: int) -> None:
        self.wait(state, mode=mode)

    def request_mode(self, resource: sim.Resource, mode: int) -> None:
        """
        Function to request resource with integer mode (based on :class:`logsim.enumerations.Modes`) for a certain
        duration.

        :param resource: Resource that is requested
        :param mode: Integer that indicates the mode of the component
        :type mode: int (:class:`logsim.enumerations.Modes`)"""
        self.request(resource, mode=mode)

    def passivate_mode(self, mode: int) -> None:
        """
        Function to passivate component with integer mode (based on :class:`logsim.enumerations.Modes`).

        :param mode: Integer that indicates the mode of the component
        :type mode: int (:class:`logsim.enumerations.Modes`)
        """
        self.passivate(mode=mode)  # type:ignore

    def set_mode(self, mode: int = None) -> None:
        """
        Function to set mode of component without any other action (mode based on :class:`logsim.enumerations.Modes`).

        :param mode: Integer that indicates the mode of the component
        :type mode: int (:class:`logsim.enumerations.Modes`)
        """
        if mode is not None:
            self._mode_time = self.env._now
            self.mode.tally(mode)
