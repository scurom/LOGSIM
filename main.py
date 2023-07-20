import logging
import uuid

from logsim.classes import Experiment, InstallationVessel, SupplyVessel, Distances, WeatherData
from logsim.distributions import PERT, Uniform, Random
from logsim.enumerations import LogisticType

logger = logging.getLogger(__name__)

# Set seed before creating experiment!
Random.set_seed(2000)

installation_vessel = InstallationVessel(
    foundation_capacity=3,
    sailing_speed=PERT(5.02, 8.99, 8.11, 'knots'),
    within_port_speed=PERT(4.51, 8.13, 5.53, 'knots'),
    day_rate=(200000, 'euro/day'),
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
barge = SupplyVessel(
    foundation_capacity=3,
    sailing_speed=Uniform(4, 6, 'knots'),
    within_port_speed=PERT(4.51, 8.13, 5.53, 'knots'),
    day_rate=(25000, 'euro/day'),
    mobilization_rate=(200000, 'euro'),
    sailing_cost=(534.23, 'euro/hour'),
    sailing_limit_wave=3,
    sailing_limit_period=8,
    load_time=PERT(1.72, 4.28, 2.45, 'h'),
    mooring_time=PERT(45, 85, 60, 'min'),
    unmooring_time=PERT(45, 85, 60, 'min'),
)
weather = WeatherData(
    file_name="data/weather_data.csv",
    start_day=1,
    start_month=1,
    synthetic=True,
    synthetic_data_samples=100
)

distances = Distances(
    supply_to_wind_farm=(355, 'km'),
    supply_to_intermediate=(460, 'km'),
    intermediate_to_wind_farm=(260, 'km'),
    intermediate_within_port=(5.39, 'km'),
    supply_within_port=(15.32, 'km')
)

# Run for each logistic type
for log_type in LogisticType:
    experiment_id = str(uuid.uuid4())
    logger.info(f"Start on {log_type}")

    # Define experiment class
    experiment = Experiment(
        installation_vessel=installation_vessel,
        logistic_configuration=log_type,
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
    experiment.run_experiment(save_db=False, save_json=False, experiment_id=experiment_id)
