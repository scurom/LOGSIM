import typing
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import rv_continuous, beta, uniform

if typing.TYPE_CHECKING:
    from scipy.stats._distn_infrastructure import rv_frozen

from logsim.utilities import UREG
import random


class Random(object):
    SEED = None
    GENERATOR = np.random.default_rng(SEED)

    @staticmethod
    def set_seed(seed):
        Random.SEED = seed
        Random.GENERATOR = np.random.default_rng(seed)
        np.random.seed(seed)
        random.seed(seed)


def to_base(value, unit):
    """Helper function to convert values to base unit (when using pint as unit manager)"""
    quant = UREG.Quantity(value, unit)
    if quant.check('[time]'):
        return quant.to('h')
    else:
        return quant.to_base_units()


class Distribution(ABC):
    check_unit: UREG.Quantity
    unit: str
    rv: "rv_frozen"

    def __repr__(self) -> str:
        """Implement representation of distribution in form of string.
        Example for PERT: PERT(minimum=1, maximum=2, mode=1, unit='h')"""
        pass

    def random_array(self, size) -> np.array:
        """Multiple random draws from the distribution according to given size, returns numpy array.

        :param size: size of array
        :return: numpy array of random draws"""
        return self.rv.rvs(size=size)

    def random(self) -> float:
        """Return one random draw as float from distribution

        :return: random draw from distribution
        """
        return round(float(self.rv.rvs()), 2)

    def get_random_sailing_duration(self, distance: UREG.Quantity) -> float:
        """Function that dynamically takes a distance and returns a random sailing duration.

        :param distance: distance to be sailed
        :return: random sailing duration in hours
        """
        if self.check_unit.check('[length]/[time]'):
            speed = self.random()
            speed = UREG.Quantity(speed, self.unit)
            time = (distance / speed)
            return round(time.to('h').magnitude, 2)
        else:
            raise Exception('Unit must be distance/time')

    @abstractmethod
    def as_dict(self) -> dict:
        """Return dictionary representation of distribution, needed for saving to files and the database. Without
        as_dict will throw an error when saving to database."""
        pass

    def as_str(self) -> str:
        pass


class Uniform(Distribution):
    check_unit: UREG.Quantity
    unit: str
    minimum: float
    maximum: float
    rv: "rv_frozen"

    def __init__(self, minimum, maximum, unit):
        self.check_unit = to_base(1, unit)
        self.unit = self.check_unit.units
        self.minimum = round(to_base(minimum, unit).magnitude, 2)
        self.maximum = round(to_base(maximum, unit).magnitude, 2)
        self.rv = uniform(loc=self.minimum, scale=(self.maximum - self.minimum))
        self.rv.random_state = Random.GENERATOR

    def __repr__(self) -> str:
        """Implement representation of distribution in form of string.
        :return: string representation of distribution"""
        return f"Uniform(minimum={self.minimum}, maximum={self.maximum}, unit={self.unit})"

    def as_dict(self) -> dict:
        return {
            "distribution_type": "Uniform",
            "minimum": self.minimum,
            "maximum": self.maximum,
            "unit": str(self.unit),
            "representation": str(self)
        }

    def as_str(self) -> str:
        return f"Uniform[{self.minimum};{self.maximum}] ({self.unit})"


class PERT(Distribution):
    """PERT distribution, based on the scipy.stats.beta distribution. In order to sample relies on
    the PertmGen which is a custom scipy implementation based on super class rv_continuous.
    The PertmGen class is a generator containing all the necessary functions to sample from the PERT distribution or
    plot the CDF/PDF etc. This PERT class is a wrapper around the PertmGen class and is used to create a PERT
    distribution and sample from it to use within the simulation.

    :param minimum: minimum value of the distribution
    :param maximum: maximum value of the distribution
    :param mode: mode of the distribution
    :param unit: unit of the distribution
    :param lmb: lambda parameter of the distribution, default is 4
    """

    check_unit: UREG.Quantity
    unit: str
    minimum: float
    maximum: float
    mode: float
    rv: "rv_frozen"

    def __init__(self, minimum, maximum, mode, unit, lmb=4):
        self.check_unit = to_base(1, unit)
        self.unit = self.check_unit.units
        self.minimum = round(to_base(minimum, unit).magnitude, 2)
        self.maximum = round(to_base(maximum, unit).magnitude, 2)
        self.mode = round(to_base(mode, unit).magnitude, 2)

        pertm = PertmGen(name="pertm")
        self.rv = pertm(self.minimum, self.mode, self.maximum, lmb=lmb)
        self.rv.random_state = Random.GENERATOR

    def __repr__(self) -> str:
        """Implement representation of distribution in form of string.

        :return: string representation of distribution"""
        return f"PERT(minimum={self.minimum}, maximum={self.maximum}, mode={self.mode}, unit={self.unit})"

    def as_dict(self) -> dict:
        """Return dictionary representation of distribution, needed for saving to files and the database.

        :return: dictionary representation of distribution"""

        return {
            "distribution_type": "PERT",
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mode": self.mode,
            "unit": str(self.unit),
            "representation": str(self)
        }

    def as_str(self) -> str:
        return f"PERT[{self.minimum};{self.maximum};{self.mode}] ({self.unit})"


class PertmGen(rv_continuous):
    """Modified beta_PERT distribution that is used to generate random PERT values.
    Based on rv_continuous class from scipy.stats and overwrites all required methods to enable sampling, plotting and
    calculations based on this continuous distribution. No type checking as signature of methods is not compatible with
    type checking. To use: First create an instance of the class, then call the instance with the parameters to freeze
    the distribution. Example: pertm = PertmGen(name="pertm") and then pertm(1, 2, 1, lmb=4). This will allow
    a frozen distribution to be used for sampling, plotting etc.
    """

    @typing.no_type_check
    def _shape(self, minimum, mode, maximum, lmb):
        s_alpha = 1 + lmb * (mode - minimum) / (maximum - minimum)
        s_beta = 1 + lmb * (maximum - mode) / (maximum - minimum)
        return [s_alpha, s_beta]

    @typing.no_type_check
    def _cdf(self, x, minimum, mode, maximum, lmb):
        s_alpha, s_beta = self._shape(minimum, mode, maximum, lmb)
        z = (x - minimum) / (maximum - minimum)
        cdf = beta.cdf(z, s_alpha, s_beta)
        return cdf

    @typing.no_type_check
    def _ppf(self, p, minimum, mode, maximum, lmb):
        s_alpha, s_beta = self._shape(minimum, mode, maximum, lmb)
        ppf = beta.ppf(p, s_alpha, s_beta)
        ppf = ppf * (maximum - minimum) + minimum
        return ppf

    @typing.no_type_check
    def _mean(self, minimum, mode, maximum, lmb):
        mean = (minimum + lmb * mode + maximum) / (2 + lmb)
        return mean

    @typing.no_type_check
    def _var(self, minimum, mode, maximum, lmb):
        mean = self._mean(minimum, mode, maximum, lmb)
        var = (mean - minimum) * (maximum - mean) / (lmb + 3)
        return var

    @typing.no_type_check
    def _skew(self, minimum, mode, maximum, lmb):
        mean = self._mean(minimum, mode, maximum, lmb)
        skew1 = (minimum + maximum - 2 * mean) / 4
        skew2 = (mean - minimum) * (maximum - mean)
        skew2 = np.sqrt(7 / skew2)
        skew = skew1 * skew2
        return skew

    @typing.no_type_check
    def _kurt(self, minimum, mode, maximum, lmb):
        a1, a2 = self._shape(minimum, mode, maximum, lmb)
        kurt1 = a1 + a2 + 1
        kurt2 = 2 * (a1 + a2) ** 2
        kurt3 = a1 * a2 * (a1 + a2 - 6)
        kurt4 = a1 * a2 * (a1 + a2 + 2) * (a1 + a2 + 3)
        kurt5 = 3 * kurt1 * (kurt2 + kurt3)
        kurt = kurt5 / kurt4 - 3  # scipy defines kurtosis of std normal distribution as 0 instead of 3
        return kurt

    @typing.no_type_check
    def _stats(self, minimum, mode, maximum, lmb):
        mean = self._mean(minimum, mode, maximum, lmb)
        var = self._var(minimum, mode, maximum, lmb)
        skew = self._skew(minimum, mode, maximum, lmb)
        kurt = self._kurt(minimum, mode, maximum, lmb)
        return mean, var, skew, kurt
