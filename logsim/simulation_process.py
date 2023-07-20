from __future__ import annotations

import logging
from math import ceil
# Use type checking constant to avoid circular imports and only import for type checking
from typing import TYPE_CHECKING, cast

import salabim as sim

# Import overriden salabim classes
from logsim.classes_salabim import Component
from logsim.enumerations import LogisticType, Modes, Location

if TYPE_CHECKING:
    from logsim.classes import InstallationVessel, SupplyVessel
    from logsim.classes_salabim import SalabimEnvironment

logger = logging.getLogger(__name__)


class FoundationGenerator(Component):
    """This class will generate the foundations for the simulation."""

    env: SalabimEnvironment

    def process(self):
        # Generate foundations, automatically all present
        for x in range(0, self.env.shared_data["left_to_install"]):
            # Add foundation to store
            yield self.to_store(self.env.stores.foundations_at_supply, sim.Component("foundation"))


class SharedProcesses:
    """This class contains processes that are shared between vessels and can be called from multiple vessels."""

    @staticmethod
    def enter_supply(vessel: InstallationVesselComponent | SupplyVesselComponent, duration: float = None, ) -> None:
        """
        Process of entering port

        :param vessel: Vessel entering port
        :param duration: Duration of entering port
        """
        if duration is None:
            duration = vessel.parameters.within_port_speed.get_random_sailing_duration(
                self.env.experiment.distances.supply_within_port)
        yield vessel.hold_mode(duration, mode=Modes.ENTERING_PORT)

    @staticmethod
    def leave_supply(vessel: InstallationVesselComponent | SupplyVesselComponent, duration: float = None, ) -> None:
        """
        Process of leaving port

        :param vessel: Vessel entering port
        :param duration: Duration of leaving port
        """
        if duration is None:
            duration = vessel.parameters.within_port_speed.get_random_sailing_duration(
                vessel.env.experiment.distances.supply_within_port)

        yield vessel.hold_mode(duration, mode=Modes.LEAVING_PORT)

    @staticmethod
    def enter_intermediate(vessel: InstallationVesselComponent | SupplyVesselComponent,
                           duration: float = None, ) -> None:
        """
        Process of entering port

        :param vessel: Vessel entering port
        :param duration: Duration of entering port
        """
        if duration is None:
            duration = vessel.parameters.within_port_speed.get_random_sailing_duration(
                vessel.env.experiment.distances.intermediate_within_port)

        yield vessel.hold_mode(duration, mode=Modes.ENTERING_PORT)

    @staticmethod
    def leave_intermediate(vessel: InstallationVesselComponent | SupplyVesselComponent,
                           duration: float = None, ) -> None:
        """
        Process of leaving port

        :param vessel: Vessel entering port
        :param duration: Duration of leaving port
        """
        if duration is None:
            duration = vessel.parameters.within_port_speed.get_random_sailing_duration(
                vessel.env.experiment.distances.intermediate_within_port)

        yield vessel.hold_mode(duration, mode=Modes.LEAVING_PORT)

    @staticmethod
    def load_from_supply_port(vessel: InstallationVesselComponent | SupplyVesselComponent, ) -> None:
        """Method checks if foundations are left on supply side. If there are foundations available the vessel will
        load the foundations one by one according to the space available on deck. Loading takes 1 hour per
        foundation.

        :param vessel: Vessel loading foundations
        """

        if len(vessel.env.stores.foundations_at_supply) == 0:
            yield vessel.passivate_mode(mode=Modes.DOCKED)

        yield vessel.hold_mode(vessel.parameters.mooring_time.random(), mode=Modes.MOORING)

        for x in range(vessel.foundations_on_board.available_quantity()):
            # Check if there are any foundations left on supply side
            if len(vessel.env.stores.foundations_at_supply) == 0:
                break
            yield vessel.from_store_mode(vessel.env.stores.foundations_at_supply, mode=Modes.WAITING_FROM_STORE)
            yield vessel.hold_mode(vessel.parameters.load_time.random(), mode=Modes.LOADING_FROM_SUPPLY)
            yield vessel.to_store(vessel.foundations_on_board, vessel.from_store_item())

        yield vessel.hold_mode(vessel.parameters.unmooring_time.random(), mode=Modes.UNMOORING)

    @staticmethod
    def load_from_intermediate_port(vessel: InstallationVesselComponent | SupplyVesselComponent, ) -> None:
        """First calculate how much still needs to be transported and how much is possible to add to the vessel. This is
        the number of foundations to load in the vessel. If there are not enough foundations at the quay side to load
        to this preferred value the vessel will passivate itself and wait with mooring and loading until there are
        enough foundations at quay side. If there are enough foundations will moor, load (1 hour per foundation)
        and unmoor again. The quay that was requested is released again after unmooring.

        :param vessel: Vessel loading foundations
        """

        def check_load():
            still_to_be_transported = (len(vessel.env.stores.foundations_at_intermediate) + len(
                vessel.env.stores.foundations_at_supply) + len(
                vessel.env.components["supplyvesselcomponent.0"].foundations_on_board))

            possible_to_add = vessel.parameters.foundation_capacity - len(vessel.foundations_on_board)
            to_load = min(possible_to_add, still_to_be_transported)
            available_quant = (vessel.env.stores.foundations_at_intermediate.available_quantity())
            if len(vessel.env.components["supplyvesselcomponent.0"].foundations_on_board) > available_quant:
                storage_full_load = len(vessel.env.stores.foundations_at_intermediate)
                to_load = min(storage_full_load, to_load)
            return to_load

        while len(vessel.env.stores.foundations_at_intermediate) < check_load():
            yield vessel.passivate_mode(mode=Modes.WAITING_FOR_SUPPLY_AVAILABILITY)

        yield vessel.request_mode(vessel.env.resources.intermediate_quay, mode=Modes.WAITING_QUAY)
        yield vessel.hold_mode(vessel.parameters.mooring_time.random(), mode=Modes.MOORING)

        for x in range(check_load()):
            yield vessel.from_store_mode(vessel.env.stores.foundations_at_intermediate, mode=Modes.WAITING_FROM_STORE, )
            yield vessel.hold_mode(vessel.parameters.load_time.random(), mode=Modes.LOADING_FROM_SUPPLY)
            yield vessel.to_store(vessel.foundations_on_board, vessel.from_store_item())

        yield vessel.hold_mode(vessel.parameters.unmooring_time.random(), mode=Modes.UNMOORING)

        if vessel.env.components["supplyvesselcomponent.0"].mode.get() == Modes.WAITING_FOR_SUPPLY_AVAILABILITY:
            vessel.env.components["supplyvesselcomponent.0"].activate()

        vessel.release(vessel.env.resources.intermediate_quay)

    @staticmethod
    def sail_wind_to_supply(vessel: InstallationVesselComponent | SupplyVesselComponent, enter_duration: float = None,
                            sailing_duration: float = None) -> None:
        """Sail from the wind farm towards the supply port. Check if sailing is possible by checking the
        sail duration against the next 'duration' weather window. Wait for weather if required, sail to supply port
        and enter port by calling SharedProcesses.enter()"""
        if sailing_duration is None:
            sailing_duration = (vessel.parameters.sailing_speed.get_random_sailing_duration(
                vessel.env.experiment.distances.supply_to_wind_farm))

        if enter_duration is None:
            enter_duration = vessel.parameters.within_port_speed.get_random_sailing_duration(
                vessel.env.experiment.distances.supply_within_port)

        if vessel.env.use_weather:
            waiting_time = vessel.env.get_next_window(sailing_duration + enter_duration,
                                                      vessel.parameters.sailing_limit_wave,
                                                      vessel.parameters.sailing_limit_period, )
            if waiting_time > 0:
                logger.debug(f"At {vessel.env.now()} need to wait {waiting_time} hours for transfer")
                yield vessel.hold_mode(waiting_time, mode=Modes.WAITING_WEATHER_SAILING)
        vessel.location.set(Location.OFFSHORE)
        yield vessel.hold_mode(sailing_duration, mode=Modes.SAILING)
        yield from SharedProcesses.enter_supply(vessel, enter_duration)
        vessel.location.set(Location.SUPPLY_PORT)

    @staticmethod
    def sail_supply_to_intermediate(vessel: InstallationVesselComponent | SupplyVesselComponent,
                                    enter_intermediate_duration: float = None,
                                    leaving_supply_duration: float = None,
                                    sailing_duration: float = None) -> None:
        """Sail from the supply port towards the intermediate port. Check if sailing is possible by checking the sail
        duration against the next 'duration' weather window. Wait for weather if required, leave supply port (
        SharedProcesses.leave()) and sail to intermediate port. Enter port again (SharedProcesses.enter())"""

        if sailing_duration is None:
            sailing_duration = vessel.parameters.sailing_speed.get_random_sailing_duration(
                vessel.env.experiment.distances.supply_to_intermediate)

        if leaving_supply_duration is None:
            leaving_supply_duration = vessel.parameters.within_port_speed.get_random_sailing_duration(
                vessel.env.experiment.distances.supply_within_port)

        if enter_intermediate_duration is None:
            enter_intermediate_duration = vessel.parameters.within_port_speed.get_random_sailing_duration(
                vessel.env.experiment.distances.intermediate_within_port)

        if vessel.env.use_weather:
            waiting_time = vessel.env.get_next_window(
                sailing_duration + enter_intermediate_duration + leaving_supply_duration,
                vessel.parameters.sailing_limit_wave,
                vessel.parameters.sailing_limit_period, )
            if waiting_time > 0:
                logger.debug(f"At {vessel.env.now()} need to wait {waiting_time} hours for transfer")
                yield vessel.hold_mode(waiting_time, mode=Modes.WAITING_WEATHER_SAILING)
        yield from SharedProcesses.leave_supply(vessel, duration=leaving_supply_duration)
        vessel.location.set(Location.OFFSHORE)
        yield vessel.hold_mode(sailing_duration, mode=Modes.SAILING)
        yield from SharedProcesses.enter_intermediate(vessel, duration=enter_intermediate_duration)
        vessel.location.set(Location.INTERMEDIATE_LOCATION)

    @staticmethod
    def sail_wind_to_intermediate(vessel: InstallationVesselComponent | SupplyVesselComponent,
                                  enter_duration: float = None, sailing_duration: float = None, ) -> None:
        """Sail from the wind farm towards the intermediate port. Check if sailing is possible by checking the sail
        duration against the next 'duration' weather window. Wait for weather if required, sail to intermediate port
        and enter port by calling SharedProcesses.enter()."""
        if sailing_duration is None:
            sailing_duration = (vessel.parameters.sailing_speed.get_random_sailing_duration(
                vessel.env.experiment.distances.intermediate_to_wind_farm))

        if enter_duration is None:
            enter_duration = vessel.parameters.within_port_speed.get_random_sailing_duration(
                vessel.env.experiment.distances.intermediate_within_port)

        if vessel.env.use_weather:
            waiting_time = vessel.env.get_next_window(sailing_duration + enter_duration,
                                                      vessel.parameters.sailing_limit_wave,
                                                      vessel.parameters.sailing_limit_period, )
            if waiting_time > 0:
                logger.debug(f"At {vessel.env.now()} need to wait {waiting_time} hours for transfer")
                yield vessel.hold_mode(waiting_time, mode=Modes.WAITING_WEATHER_SAILING)
        vessel.location.set(Location.OFFSHORE)
        yield vessel.hold_mode(sailing_duration, mode=Modes.SAILING)
        yield from SharedProcesses.enter_intermediate(vessel, enter_duration)
        vessel.location.set(Location.INTERMEDIATE_LOCATION)


class InstallationVesselComponent(Component):
    """Installation vessel component (inherits from Salabim Component). Describes installation vessel with foundations aboard.

    :param vessel_parameters: Vessel parameters
    """
    env: SalabimEnvironment
    parameters: InstallationVessel
    foundations_on_board: sim.Store
    location: sim.State
    draws = {'installation_duration': []}

    def __init__(self, vessel_parameters: InstallationVessel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = vessel_parameters
        self.foundations_on_board = sim.Store("foundations_on_board_iv", capacity=self.parameters.foundation_capacity)
        self.installation_end_time = 0
        self.wind_farm_working = sim.State("wind_farm_location", value=False)

    def process(self):
        """Main process of IV. According to the logistic configuration, the IV will sail to the appropriate location
        and the repeating process will start."""
        self.location.set(Location.SUPPLY_PORT)
        match self.env.experiment.logistic_configuration:
            case LogisticType.OFFSHORE_TRANSFER_2:
                yield from self.sail_supply_to_wind()
                self.wind_farm_working.set(True)
            case LogisticType.PORT_TRANSFER_3:
                yield from SharedProcesses.sail_supply_to_intermediate(self)
            case LogisticType.PORT_TRANSSHIPMENT_4:
                supply_vessel = self.env.components["supplyvesselcomponent.0"]
                yield self.wait_mode(supply_vessel.stock_build_up, mode=Modes.DOCKED)
                yield from SharedProcesses.sail_supply_to_intermediate(self)
            case LogisticType.OFFSHORE_TRANSFER_BUFFER_5:
                supply_vessel = self.env.components["supplyvesselcomponent.0"]
                yield self.wait_mode(supply_vessel.stock_build_up, mode=Modes.DOCKED)
                yield from self.sail_supply_to_wind()
                self.wind_farm_working.set(True)

        # Start process
        while True:
            if len(self.foundations_on_board) > 0:
                yield from self.installation_process()

            if self.env.shared_data["left_to_install"] == 0:
                yield from SharedProcesses.sail_wind_to_supply(self)
                yield self.passivate_mode(mode=Modes.DOCKED)

            # Restock if needed
            if len(self.foundations_on_board) == 0:
                match self.env.experiment.logistic_configuration:
                    case LogisticType.DIRECT_1:
                        yield from self.restock_at_supply_port()
                    case LogisticType.OFFSHORE_TRANSFER_2:
                        yield from self.transfer_offshore_restock()
                    case LogisticType.PORT_TRANSFER_3:
                        yield from self.transfer_at_intermediate_port()
                    case LogisticType.PORT_TRANSSHIPMENT_4:
                        yield from self.restock_at_intermediate_port()
                    case LogisticType.OFFSHORE_TRANSFER_BUFFER_5:
                        yield from self.transfer_offshore_restock()
                    case _:
                        raise NotImplementedError

    def installation_process(self) -> None:
        """Governing the installation process itself in this function. Check if weather allows installation of one
        monopole and install this one"""
        # Check if weather is used
        # Calculate expected duration
        total_time = 0
        current_time = self.env.now()

        self.wind_farm_working.set(False)
        if len(self.draws['installation_duration']) == 0:
            # Pre-draw random numbers for all foundations on board, used for expected duration and installation time

            random_draws = [self.parameters.installation_time.random() for _ in
                            range(0, len(self.foundations_on_board))]
        else:
            # If direct sailing draws already taken so use these
            random_draws = self.draws['installation_duration']

            # Draw for missing foundations not yet drawn in the check
            random_draws.extend([self.parameters.installation_time.random() for _ in
                                 range(0, len(self.foundations_on_board) - len(random_draws))])
            self.draws['installation_duration'] = []

        # Calculate total time to communicate this to other vessels
        if self.env.use_weather:
            for foundation in range(0, len(self.foundations_on_board)):
                waiting_time = self.env.get_next_window(ceil(random_draws[foundation]),
                                                        self.parameters.installation_limit_wave,
                                                        self.parameters.installation_limit_period, current_time)
                total_time = total_time + waiting_time + random_draws[foundation]
                current_time = current_time + waiting_time + random_draws[foundation]

        self.installation_end_time = self.env.now() + total_time

        for foundation in range(0, len(self.foundations_on_board)):
            self.wind_farm_working.set(True)
            if self.env.use_weather:
                # get weather waiting_time of installation_time rounded up
                waiting_time = self.env.get_next_window(ceil(random_draws[foundation]),
                                                        self.parameters.installation_limit_wave,
                                                        self.parameters.installation_limit_period, )
                if waiting_time > 0:
                    logger.debug(f"At {self.env.now()} need to wait {waiting_time} hours for Installing")
                    yield self.hold_mode(waiting_time, mode=Modes.WAITING_WEATHER_INSTALLING)

            # Installation time
            self.env.shared_data["left_to_install"] = (self.env.shared_data["left_to_install"] - 1)
            # Request new foundation to install
            yield self.from_store(self.foundations_on_board)
            yield self.hold_mode(random_draws[foundation], mode=Modes.INSTALL)

    def restock_at_intermediate_port(self):
        """Function to use when LogisticType.PORT_TRANSSHIPMENT_4 is used. Sail to intermediate port, restock by mooring
        to quay and sail back."""
        if self.location.get() != Location.INTERMEDIATE_LOCATION:
            yield from SharedProcesses.sail_wind_to_intermediate(self)
        yield from SharedProcesses.load_from_intermediate_port(self)
        yield from self.sail_intermediate_to_wind()

    def transfer_at_intermediate_port(self):
        """Function used when LogisticType.PORT_TRANSFER_3 is used. Sail to intermediate port, transfer and sail
        back."""
        if self.location.get() != Location.INTERMEDIATE_LOCATION:
            yield from SharedProcesses.sail_wind_to_intermediate(self)
        yield from self.transfer_at_port()
        yield from self.sail_intermediate_to_wind()

    def transfer_offshore_restock(self):
        """Function used when LogisticType.OFFSHORE_TRANSFER_2 or LogisticType.OFFSHORE_TRANSFER_BUFFER_5 is used.
        Wait until vessel is alongside, moor this vessel and transfer foundations
        by getting foundation from supply_vessel store and putting them on the vessel store. Unmoor alongside vessel and
        activate the supply_vessel process again."""
        # Wait for vessel to be alongside, wait for queue to contain 1 item
        yield self.from_store_mode(self.env.stores.supply_vessels_alongside, mode=Modes.WAITING_TRANSFER)

        # Cast to SupplyVesselComponent type to allow type checking
        supply_vessel = self.from_store_item()
        supply_vessel = cast(SupplyVesselComponent, supply_vessel)

        supply_vessel.set_mode(mode=Modes.TRANSFER)
        # Take minimum of available on supply_vessel or possible to load on vessel
        to_load = min(self.parameters.foundation_capacity, len(supply_vessel.foundations_on_board))

        loading_duration = (
                sum(supply_vessel.draws['transfer_time_draws']) + supply_vessel.draws['alongside_mooring_time'] +
                supply_vessel.draws['alongside_unmooring_time'])

        if self.env.use_weather:
            if not self.env.decision_rules:
                # Need to manually check weather when no decision rules are used
                waiting_time = self.env.get_next_window(loading_duration, self.parameters.alongside_limit_wave,
                                                        self.parameters.alongside_limit_period, )
                if waiting_time > 0:
                    logger.debug(f"At {self.env.now()} need to wait {waiting_time} hours for Loading")
                    yield self.hold_mode(waiting_time, mode=Modes.WAITING_WEATHER_TRANSFER)

        yield self.hold_mode(supply_vessel.draws['alongside_mooring_time'], mode=Modes.MOORING)
        for x in range(to_load):
            yield self.from_store_mode(supply_vessel.foundations_on_board, mode=Modes.WAITING_FROM_STORE)
            yield self.hold_mode(supply_vessel.draws['transfer_time_draws'][x], mode=Modes.TRANSFER)
            yield self.to_store(self.foundations_on_board, self.from_store_item())

        yield self.hold_mode(supply_vessel.draws['alongside_unmooring_time'], mode=Modes.UNMOORING)

        supply_vessel.draws['alongside_mooring_time'] = 0
        supply_vessel.draws['alongside_unmooring_time'] = 0
        supply_vessel.draws['transfer_time_draws'] = []

        supply_vessel.activate()

    def restock_at_supply_port(self):
        """Used for LogisticType.DIRECT_1"""
        # If not already there (first run) then sail
        if self.location.get() != Location.SUPPLY_PORT:
            yield from SharedProcesses.sail_wind_to_supply(self)

        # Load from supply if moored
        yield from SharedProcesses.load_from_supply_port(self)
        yield from self.sail_supply_to_wind()

    def transfer_at_port(self):
        """Called by transfer_at_intermediate_port. Used for LogisticType.PORT_TRANSFER_3."""
        yield self.from_store_mode(self.env.stores.supply_vessels_alongside, mode=Modes.WAITING_TRANSFER)
        supply_vessel = self.from_store_item()
        supply_vessel = cast(SupplyVesselComponent, supply_vessel)
        supply_vessel.set_mode(mode=Modes.TRANSFER)

        to_load = min(self.parameters.foundation_capacity, len(supply_vessel.foundations_on_board))

        yield self.hold_mode(self.parameters.alongside_mooring_time.random(), mode=Modes.MOORING)
        for x in range(to_load):
            yield self.from_store_mode(supply_vessel.foundations_on_board, mode=Modes.WAITING_FROM_STORE)
            yield self.hold_mode(self.parameters.alongside_transfer_time.random(), mode=Modes.TRANSFER)
            yield self.to_store(self.foundations_on_board, self.from_store_item())
        yield self.hold_mode(self.parameters.alongside_unmooring_time.random(), mode=Modes.UNMOORING)
        supply_vessel.activate()

    def sail_supply_to_wind(self) -> None:
        """Sail from the supply port towards the wind farm. Check if sailing is possible by checking the sail
        duration against the next 'duration' weather window. Wait for weather if required, leave supply port (
        SharedProcesses.leave()) and sail to wind farm."""

        sailing_duration = self.parameters.sailing_speed.get_random_sailing_duration(
            self.env.experiment.distances.supply_to_wind_farm)

        random_draws = [self.parameters.installation_time.random() for _ in
                        range(min(2, self.parameters.foundation_capacity))]
        installation_duration_min_foundations = sum(random_draws)
        self.draws['installation_duration'] = random_draws

        # Draw already to check weather
        exit_supply_time_draw = self.parameters.within_port_speed.get_random_sailing_duration(
            self.env.experiment.distances.supply_within_port)

        if self.env.use_weather:
            if self.env.decision_rules:
                waiting_time = self.env.get_next_window_2_limits(
                    [sailing_duration + exit_supply_time_draw, installation_duration_min_foundations, ],
                    [self.parameters.sailing_limit_wave, self.parameters.installation_limit_wave, ],
                    [self.parameters.sailing_limit_period, self.parameters.installation_limit_period, ], )
            else:
                waiting_time = self.env.get_next_window(sailing_duration + exit_supply_time_draw,
                                                        self.parameters.sailing_limit_wave,
                                                        self.parameters.sailing_limit_period, )
            if waiting_time > 0:
                # Code below can be used for debugging to get pandas dataframe
                # current_hour_take_into_account = self.env.weather_sample.collect().filter(
                #     pl.col('hour') >= floor(self.env.now() - waiting_time)).to_pandas()
                logger.debug(f"Needed to wait for {waiting_time} hours for weather to sail to "
                             f"wind farm and install 2 foundations")
                yield self.hold_mode(waiting_time, mode=Modes.WAITING_WEATHER_SAILING)

        yield from SharedProcesses.leave_supply(self, duration=exit_supply_time_draw)
        self.location.set(Location.OFFSHORE)
        yield self.hold_mode(sailing_duration, mode=Modes.SAILING)
        self.location.set(Location.WIND_FARM)

    def sail_intermediate_to_wind(self) -> None:
        """Sail from the supply port towards the wind farm. Check if sailing is possible by checking the sail
        duration against the next 'duration' weather window. Wait for weather if required, leave supply port (
        SharedProcesses.leave()) and sail to wind farm."""

        sailing_duration = self.parameters.sailing_speed.get_random_sailing_duration(
            self.env.experiment.distances.intermediate_to_wind_farm)
        exit_intermediate_time = self.parameters.within_port_speed.get_random_sailing_duration(
            self.env.experiment.distances.intermediate_within_port)

        if self.env.use_weather:
            if self.env.decision_rules:
                random_draws = [self.parameters.installation_time.random() for _ in
                                range(min(2, self.parameters.foundation_capacity))]
                self.draws['installation_duration'] = random_draws
                installation_duration_min_foundations = sum(random_draws)

                waiting_time = self.env.get_next_window_2_limits(
                    [sailing_duration + exit_intermediate_time,
                     installation_duration_min_foundations, ],
                    [self.parameters.sailing_limit_wave, self.parameters.installation_limit_wave, ],
                    [self.parameters.sailing_limit_period, self.parameters.installation_limit_period, ], )
            else:
                waiting_time = self.env.get_next_window(
                    sailing_duration + exit_intermediate_time,
                    self.parameters.sailing_limit_wave, self.parameters.sailing_limit_period, )
            if waiting_time > 0:
                yield self.hold_mode(waiting_time, mode=Modes.WAITING_WEATHER_SAILING)
        yield from SharedProcesses.leave_intermediate(self, duration=exit_intermediate_time)
        self.location.set(Location.OFFSHORE)
        yield self.hold_mode(sailing_duration, mode=Modes.SAILING)
        self.location.set(Location.WIND_FARM)


class SupplyVesselComponent(Component):
    """Supply vessel component (inherits from Salabim Component). Describes supply vessel with foundations aboard.

    :param vessel_parameters: Vessel parameters
    :param supply_vessel_number: supply vessel number, used to determine
        which vessel to use in LogisticType.OFFSHORE_TRANSFER_BUFFER_5
    :param args: Salabim args
    :param kwargs: Salabim
    kwargs
    """

    env: SalabimEnvironment
    parameters: SupplyVessel
    foundations_on_board: sim.Store
    location: sim.State
    draws = {"enter_supply_time": 0.0, "enter_intermediate_time": 0.0, "sailing_duration_to_intermediate": 0.0,
             "sailing_duration_to_supply": 0.0, "transfer_time_draws": [], "alongside_mooring_time": 0.0,
             "alongside_unmooring_time": 0.0}

    def __init__(self, vessel_parameters: SupplyVessel, supply_vessel_number: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = vessel_parameters
        self.foundations_on_board = sim.Store("foundations_on_board_vessel",
                                              capacity=self.parameters.foundation_capacity)
        self.supply_vessel_number = supply_vessel_number
        self.stock_build_up = sim.State("stock_build_up", value=False)

    def process(self):
        """Process of the supply vessel. Describes the process of the vessel, depending on the logistic
        configuration."""
        self.location.set(Location.SUPPLY_PORT)
        # start setting
        match self.env.experiment.logistic_configuration:
            case LogisticType.OFFSHORE_TRANSFER_BUFFER_5:
                # Vessel 2 only sails from intermediate location onwards, wait until 1 is done
                if self.supply_vessel_number == 2:
                    supply_vessel_1 = self.env.components["supplyvesselcomponent.0"]
                    yield self.wait_mode(supply_vessel_1.stock_build_up, mode=Modes.DOCKED)
                    yield from SharedProcesses.sail_supply_to_intermediate(self)

        while True:
            match self.env.experiment.logistic_configuration:
                case LogisticType.OFFSHORE_TRANSFER_2:
                    yield from self.transfer_offshore()
                case LogisticType.PORT_TRANSFER_3:
                    yield from self.transfer_intermediate()
                case LogisticType.PORT_TRANSSHIPMENT_4:
                    yield from self.transshipment_intermediate()
                case LogisticType.OFFSHORE_TRANSFER_BUFFER_5:
                    yield from self.transfer_offshore_from_intermediate()
                case _:
                    raise NotImplementedError

    def transshipment_intermediate(self):
        """Used for LogisticType.PORT_TRANSSHIPMENT_4 to load from supply port to intermediate port."""
        yield from SharedProcesses.load_from_supply_port(self)
        yield from SharedProcesses.sail_supply_to_intermediate(self)
        yield from self.load_to_intermediate_port()
        yield from self.sail_intermediate_to_supply()

    def transfer_intermediate(self):
        """Used for LogisticType.PORT_TRANSFER_3 to load from supply port to IV within intermediate port by mooring
        alongside and transferring"""
        yield from SharedProcesses.load_from_supply_port(self)
        yield from SharedProcesses.sail_supply_to_intermediate(self)

        self.enter(self.env.stores.supply_vessels_alongside)

        # PROCESS TRANSFERRED TO INSTALLATION VESSEL
        yield self.passivate_mode(mode=Modes.WAITING_TRANSFER)

        yield from self.sail_intermediate_to_supply()

    def transfer_offshore_from_intermediate(self):
        """
        Used for LogisticType.OFFSHORE_TRANSFER_BUFFER_5 to load from intermediate port to IV offshore by mooring
        alongside and transferring. Used only by vessel 2. If this function is called by barge 1 then
        SupplyVesselComponent.transshipment_intermediate() will be called as barge 1 only
        transports from supply to intermediate port.
        """
        if self.supply_vessel_number == 1:
            yield from self.transshipment_intermediate()

        elif self.supply_vessel_number == 2:
            yield from SharedProcesses.load_from_intermediate_port(self)
            yield from self.sail_intermediate_to_wind()

            self.enter(self.env.stores.supply_vessels_alongside)
            yield self.passivate_mode(mode=Modes.WAITING_TRANSFER)

            other_supply_vessel = cast(SupplyVesselComponent, self.env.components["supplyvesselcomponent.0"])

            # Check if other barge is empty, intermediate is empty and supply is empty
            if (len(other_supply_vessel.foundations_on_board) + len(self.env.stores.foundations_at_supply) + len(
                    self.env.stores.foundations_at_intermediate)) == 0:
                yield from SharedProcesses.sail_wind_to_supply(self)
                yield self.passivate_mode(mode=Modes.DOCKED)

            yield from SharedProcesses.sail_wind_to_intermediate(self,
                                                                 enter_duration=self.draws['enter_intermediate_time'],
                                                                 sailing_duration=self.draws[
                                                                     'sailing_duration_to_intermediate'], )

            self.draws['enter_intermediate_time'] = 0
            self.draws['sailing_duration_to_intermediate'] = 0

    def transfer_offshore(self):
        """
        Responsible for transfer offshore process. First sails to wind farm location. Makes use of global store
        supply_vessels_alongside when arrived. This store enables the installation vessel to 'wait' until a vessel
        with stock is available and thus completing the synchronization process.
        """
        yield from SharedProcesses.load_from_supply_port(self)
        yield from self.sail_supply_to_wind()

        # PROCESS TRANSFERRED TO INSTALLATION VESSEL
        while len(self.foundations_on_board) > 0:
            self.enter(self.env.stores.supply_vessels_alongside)
            yield self.passivate_mode(mode=Modes.WAITING_TRANSFER)

        # If already established enter time draw use this one, otherwise draw a new one
        yield from SharedProcesses.sail_wind_to_supply(self, enter_duration=self.draws['enter_supply_time'],
                                                       sailing_duration=self.draws['sailing_duration_to_supply'])

        self.draws['enter_supply_time'] = 0
        self.draws['sailing_duration_to_supply'] = 0

        if len(self.env.stores.foundations_at_supply) == 0:
            yield self.passivate_mode(mode=Modes.DOCKED)

    def sail_supply_to_wind(self) -> None:
        """
        Sail from the supply port towards the wind farm. Check if sailing is possible by checking the sail
        duration against the next 'duration' weather window. Wait for weather if required, leave supply port (
        SharedProcesses.leave()) and sail to wind farm.
        """

        iv = self.env.components["installationvesselcomponent.0"]
        iv = cast(InstallationVesselComponent, iv)

        sailing_duration_to_wind = (
            self.parameters.sailing_speed.get_random_sailing_duration(
                self.env.experiment.distances.supply_to_wind_farm))
        sailing_duration_to_supply = (
            self.parameters.sailing_speed.get_random_sailing_duration(
                self.env.experiment.distances.supply_to_wind_farm))

        self.draws['sailing_duration_to_supply'] = sailing_duration_to_supply

        yield self.wait_mode(iv.wind_farm_working, mode=Modes.WAITING_REQUEST)
        started_sailing_time = iv.installation_end_time - sailing_duration_to_wind

        hold_duration = started_sailing_time - self.env.now()
        if hold_duration > 0:
            yield self.hold_mode(hold_duration, mode=Modes.WAITING_REQUEST)

        alongside_mooring_time_draw = iv.parameters.alongside_mooring_time.random()
        alongside_unmooring_time_draw = iv.parameters.alongside_unmooring_time.random()
        transfer_time_draws = [iv.parameters.alongside_transfer_time.random() for _ in
                               range(0, iv.parameters.foundation_capacity)]

        loading_duration = (sum(transfer_time_draws) + alongside_mooring_time_draw + alongside_unmooring_time_draw)

        exit_supply_time_draw = self.parameters.within_port_speed.get_random_sailing_duration(
            self.env.experiment.distances.supply_within_port
        )
        enter_supply_time_draw = self.parameters.within_port_speed.get_random_sailing_duration(
            self.env.experiment.distances.supply_within_port)

        self.draws['enter_supply_time'] = enter_supply_time_draw
        self.draws['transfer_time_draws'] = transfer_time_draws
        self.draws['alongside_mooring_time'] = alongside_mooring_time_draw
        self.draws['alongside_unmooring_time'] = alongside_unmooring_time_draw

        if self.env.use_weather:
            if self.env.decision_rules:
                waiting_time = self.env.get_next_window_3_limits(
                    window_sizes=[sailing_duration_to_wind + exit_supply_time_draw, loading_duration,
                                  sailing_duration_to_supply + enter_supply_time_draw, ],
                    hm0_limits=[self.parameters.sailing_limit_wave, iv.parameters.alongside_limit_wave,
                                self.parameters.sailing_limit_wave, ],
                    tp_limits=[self.parameters.sailing_limit_period, iv.parameters.alongside_limit_period,
                               self.parameters.sailing_limit_period, ], )
            else:
                waiting_time = self.env.get_next_window(sailing_duration_to_wind + exit_supply_time_draw,
                                                        self.parameters.sailing_limit_wave,
                                                        self.parameters.sailing_limit_period, )
            if waiting_time > 0:
                logger.debug(f"At {self.env.now()} need to wait {waiting_time} hours for transfer")
                yield self.hold_mode(waiting_time, mode=Modes.WAITING_WEATHER_SAILING)
            else:
                logger.debug(f"At {self.env.now()} no waiting time for transfer")

        yield from SharedProcesses.leave_supply(self, exit_supply_time_draw)
        iv.installation_end_time = 0
        self.location.set(Location.OFFSHORE)
        yield self.hold_mode(sailing_duration_to_wind, mode=Modes.SAILING)
        self.location.set(Location.WIND_FARM)

    def sail_intermediate_to_wind(self) -> None:
        """Sail from the supply port towards the wind farm. Check if sailing is possible by checking the sail
        duration against the next 'duration' weather window. Wait for weather if required, leave supply port (
        SharedProcesses.leave()) and sail to wind farm."""

        iv = self.env.components["installationvesselcomponent.0"]
        iv = cast(InstallationVesselComponent, iv)

        sailing_duration_to_wind = (self.parameters.sailing_speed.get_random_sailing_duration(
            self.env.experiment.distances.intermediate_to_wind_farm))
        sailing_duration_to_intermediate = (self.parameters.sailing_speed.get_random_sailing_duration(
            self.env.experiment.distances.intermediate_to_wind_farm))

        start = self.env.now()

        yield self.wait_mode(iv.wind_farm_working, mode=Modes.WAITING_REQUEST)
        if self.env.now() - start > 0:
            logger.debug(f"Barge waited {self.env.now() - start} "
                         f"hours for installation vessel to be present at the wind farm location")

        started_sailing_time = iv.installation_end_time - sailing_duration_to_wind
        hold_duration = started_sailing_time - self.env.now()

        if hold_duration > 0:
            yield self.hold_mode(hold_duration, mode=Modes.WAITING_REQUEST)

        alongside_mooring_time_draw = iv.parameters.alongside_mooring_time.random()
        alongside_unmooring_time_draw = iv.parameters.alongside_unmooring_time.random()
        transfer_time_draws = [iv.parameters.alongside_transfer_time.random() for _ in
                               range(0, iv.parameters.foundation_capacity)]

        loading_duration = (sum(transfer_time_draws) + alongside_mooring_time_draw + alongside_unmooring_time_draw)

        exit_intermediate_time_draw = self.parameters.within_port_speed.get_random_sailing_duration(
            self.env.experiment.distances.intermediate_within_port)
        enter_intermediate_time_draw = self.parameters.within_port_speed.get_random_sailing_duration(
            self.env.experiment.distances.intermediate_within_port)

        self.draws['enter_intermediate_time'] = enter_intermediate_time_draw
        self.draws['transfer_time_draws'] = transfer_time_draws
        self.draws['alongside_mooring_time'] = alongside_mooring_time_draw
        self.draws['alongside_unmooring_time'] = alongside_unmooring_time_draw
        self.draws['sailing_duration_to_intermediate'] = sailing_duration_to_intermediate

        if self.env.use_weather:
            if self.env.decision_rules:
                waiting_time = self.env.get_next_window_3_limits(
                    window_sizes=[sailing_duration_to_wind + exit_intermediate_time_draw, loading_duration,
                                  sailing_duration_to_intermediate + enter_intermediate_time_draw, ],
                    hm0_limits=[self.parameters.sailing_limit_wave, iv.parameters.alongside_limit_wave,
                                self.parameters.sailing_limit_wave, ],
                    tp_limits=[self.parameters.sailing_limit_period, iv.parameters.alongside_limit_period,
                               self.parameters.sailing_limit_period, ], )
            else:
                waiting_time = self.env.get_next_window(sailing_duration_to_wind + exit_intermediate_time_draw,
                                                        self.parameters.sailing_limit_wave,
                                                        self.parameters.sailing_limit_period, )
            if waiting_time > 0:
                logger.debug(f"At {self.env.now()} need to wait {waiting_time} hours for transfer")
                yield self.hold_mode(waiting_time, mode=Modes.WAITING_WEATHER_SAILING)
            else:
                logger.debug(f"At {self.env.now()} no waiting time for transfer")

        yield from SharedProcesses.leave_intermediate(self, duration=exit_intermediate_time_draw)
        iv.installation_end_time = 0
        self.location.set(Location.OFFSHORE)
        yield self.hold_mode(sailing_duration_to_wind, mode=Modes.SAILING)
        self.location.set(Location.WIND_FARM)

    def load_to_intermediate_port(self) -> None:
        """Request a 'quay' resource object. Moor for 0.25 hour. Then unload each foundation one by one, takes an hour
        per foundation to unload. Foundations are transferred from the 'foundations_on_board' store to the
        'foundations_at_intermediate' store. After loading to port check if vessels are waiting to be loaded in the
        port area. If they are this means that they were waiting because they want to be loaded only to full capacity
        and the required numbers of foundations to do this was not present at quay side. Activate these waiting
        vessels again in order to allow them to check again if now enough foundations are present to load to
        capacity. Hold again to unmoor and release the quay resource.
        """
        while self.env.stores.foundations_at_intermediate.available_quantity() < len(self.foundations_on_board):
            yield self.passivate_mode(mode=Modes.WAITING_FOR_SUPPLY_AVAILABILITY)

        yield self.request_mode(self.env.resources.intermediate_quay, mode=Modes.WAITING_QUAY)
        yield self.hold_mode(self.parameters.mooring_time.random(), mode=Modes.MOORING)

        for x in range(len(self.foundations_on_board)):
            yield self.from_store(self.foundations_on_board)
            yield self.hold_mode(self.parameters.load_time.random(), mode=Modes.UNLOADING_FROM_SUPPLY)
            yield self.to_store(self.env.stores.foundations_at_intermediate, self.from_store_item())

        for wait_vessel in self.env.components:
            if self.env.components[wait_vessel].mode.get() == Modes.WAITING_FOR_SUPPLY_AVAILABILITY:
                self.env.components[wait_vessel].activate()

        yield self.hold_mode(self.parameters.unmooring_time.random(), mode=Modes.UNMOORING)

        if (len(self.env.stores.foundations_at_intermediate) >=
                self.env.experiment.intermediate_location_stock_minimum):
            self.stock_build_up.set(True)

        self.release(self.env.resources.intermediate_quay)

    def sail_intermediate_to_supply(self,
                                    enter_supply_duration: float = None,
                                    leaving_intermediate_duration: float = None,
                                    sailing_duration: float = None) -> None:
        """Sail from the intermediate port towards the supply port. Check if sailing is possible by checking the sail
        duration against the next 'duration' weather window. Wait for weather if required, leave intermediate port (
        SharedProcesses.leave()) and sail to supply port. Enter port again (SharedProcesses.enter())."""
        if sailing_duration is None:
            sailing_duration = self.parameters.sailing_speed.get_random_sailing_duration(
                self.env.experiment.distances.supply_to_intermediate)
        if leaving_intermediate_duration is None:
            leaving_intermediate_duration = self.parameters.within_port_speed.get_random_sailing_duration(
                self.env.experiment.distances.intermediate_within_port)

        if enter_supply_duration is None:
            enter_supply_duration = self.parameters.within_port_speed.get_random_sailing_duration(
                self.env.experiment.distances.supply_within_port)

        if self.env.use_weather:
            waiting_time = self.env.get_next_window(
                sailing_duration + leaving_intermediate_duration + enter_supply_duration,
                self.parameters.sailing_limit_wave,
                self.parameters.sailing_limit_period, )
            if waiting_time > 0:
                logger.debug(f"At {self.env.now()} need to wait {waiting_time} hours for transfer")
                yield self.hold_mode(waiting_time, mode=Modes.WAITING_WEATHER_SAILING)
        yield from SharedProcesses.leave_intermediate(self, duration=leaving_intermediate_duration)
        self.location.set(Location.OFFSHORE)
        yield self.hold_mode(sailing_duration, mode=Modes.SAILING)
        yield from SharedProcesses.enter_supply(self, duration=enter_supply_duration)
        self.location.set(Location.SUPPLY_PORT)
