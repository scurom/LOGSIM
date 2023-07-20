from collections import namedtuple
from enum import Enum, IntEnum, auto

Stores = namedtuple("Stores", ["supply_vessels_alongside", "foundations_at_supply", "foundations_at_intermediate"])
Resources = namedtuple('Resources', ['intermediate_quay'])


class LogisticType(Enum):
    """Define the different logistic configurations. 
    The numbers refer to the numbers as in the documentation overview of these different configurations"""
    DIRECT_1 = auto()
    PORT_TRANSFER_3 = auto()
    OFFSHORE_TRANSFER_2 = auto()
    OFFSHORE_TRANSFER_BUFFER_5 = auto()
    PORT_TRANSSHIPMENT_4 = auto()


class Location(IntEnum):
    """The locations a vessel can be at"""
    NONE_LOCATION = 0
    SUPPLY_PORT = 1
    WIND_FARM = 2
    INTERMEDIATE_LOCATION = 3
    SAILING = 4
    OFFSHORE = 5


class Modes(IntEnum):
    """The tracked modes in the simulation"""
    NONE_MODE = 0
    DOCKED = 1
    INSTALL = 2
    TRANSFER = 3
    SAILING = 4
    WAITING_TRANSFER = 5
    WAITING_FROM_STORE = 6
    LOADING_FROM_SUPPLY = 7
    UNLOADING_FROM_SUPPLY = 8
    WAITING_WEATHER_INSTALLING = 9
    WAITING_WEATHER_TRANSFER = 10
    WAITING_WEATHER_SAILING = 11
    MOORING = 12
    UNMOORING = 13
    LEAVING_PORT = 14
    ENTERING_PORT = 15
    WAITING_QUAY = 16
    WAITING_FOR_SUPPLY_AVAILABILITY = 17
    WAITING_REQUEST = 18


class KPIModes(Enum):
    """This enumeration defines the combination of KPI enumeration values that define a KPI value. These will be
    combined in the calculation of the KPIs and added up."""
    WAITING_FOR_WEATHER = [Modes.WAITING_WEATHER_INSTALLING.value, Modes.WAITING_WEATHER_SAILING.value,
                           Modes.WAITING_WEATHER_TRANSFER.value]
    WAITING_OTHER = [Modes.WAITING_TRANSFER.value, Modes.WAITING_FROM_STORE.value, Modes.WAITING_QUAY.value,
                     Modes.WAITING_FOR_SUPPLY_AVAILABILITY.value, Modes.WAITING_REQUEST.value]
    WAITING = [Modes.WAITING_WEATHER_INSTALLING.value, Modes.WAITING_WEATHER_SAILING.value,
               Modes.WAITING_WEATHER_TRANSFER.value, Modes.WAITING_TRANSFER.value, Modes.WAITING_FROM_STORE.value,
               Modes.WAITING_QUAY.value, Modes.WAITING_FOR_SUPPLY_AVAILABILITY.value, Modes.WAITING_REQUEST.value]
    OPERATING = [Modes.INSTALL.value, Modes.TRANSFER.value, Modes.LOADING_FROM_SUPPLY.value,
                 Modes.UNLOADING_FROM_SUPPLY.value, Modes.MOORING.value, Modes.UNMOORING.value]
    ON_DP = [Modes.INSTALL.value, Modes.TRANSFER.value]
    SAILING = [Modes.SAILING.value, Modes.LEAVING_PORT.value, Modes.ENTERING_PORT.value]


class ComponentConfigurationError(Exception):
    """ Configuration not aligned with create components. """
    pass


class NoWindowFoundError(Exception):
    """No available window was found anymore, so max run time of 10000 hours reached"""
    pass
