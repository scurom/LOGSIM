import copy
import json
from math import floor
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import salabim as sim
from pint import UnitRegistry

from logsim.enumerations import Modes

# Provide global unit registry utility
UREG = UnitRegistry(system='mks')


def prepare_experiments(date_time_start):
    with open('experiments.json') as json_file:
        data = json.load(json_file)

    a_distances = [entry[0] for entry in data['distances']['supply_to_wind_farm']]
    b_distances = [entry[0] for entry in data['distances']['intermediate_to_wind_farm']]

    comb = [(a, b, round(sqrt(a ** 2 + b ** 2), 2)) for a in a_distances for b in b_distances if (b <= a)]
    data['distances'] = {}
    data['distances']['combination'] = comb

    # Create empty dictionary with base values
    base_dict = {}
    base_dict['start_datetime'] = date_time_start
    for category in data.keys():
        base_dict[category] = {}
        for parameter in data[category]:
            base_dict[category][parameter] = data[category][parameter][0]

    # Create experiments
    experiments = []
    # Add base experiment
    base_dict['count'] = 0
    experiments.append(copy.deepcopy(base_dict))
    counter = 1
    for category in data.keys():
        for parameter in data[category]:
            for parameter_value in data[category][parameter][1:]:
                base_dict[category][parameter] = parameter_value
                base_dict['count'] = counter
                experiments.append(copy.deepcopy(base_dict))
                counter += 1
            base_dict[category][parameter] = data[category][parameter][0]
    return experiments


def plot(sim, iv, barge, sample_no, barge_2=None):
    # Set autolayout
    plt.rcParams.update({'figure.autolayout': True, 'figure.figsize': (10, 5)})

    # Examples to get data on modes (used in multiple places to avoid states (boiler code) and stores
    iv_data = iv.get_mode_data()
    iv_foundations = sim.env.get_store_length(iv.foundations_on_board)

    if barge is not None:
        barge_data = barge.get_mode_data()
        barge_foundations = sim.env.get_store_length(barge.foundations_on_board)

    # Barge State Plot
    state_array = np.array([])
    if barge is not None:
        for x in barge_data[:, 1]:
            state_array = np.append(state_array, Modes(x).name)
        # Create plot of states
        plt.step(barge_data[:, 0], state_array, where='post')
        plt.title(f'Barge state plot')
        plt.show()

    state_array = np.array([])
    if barge_2 is not None:
        barge_data_2 = barge_2.get_mode_data()
        barge_foundations_2 = sim.env.get_store_length(barge_2.foundations_on_board)
        for x in barge_data_2[:, 1]:
            state_array = np.append(state_array, Modes(x).name)
        # Create plot of states
        plt.step(barge_data_2[:, 0], state_array, where='post')
        plt.title(f'Barge 2 state plot')
        plt.show()

    # Foundation data plot
    plt.step(iv_foundations[:, 0], iv_foundations[:, 1], label='Installation Vessel', where='post')
    # Barge data plot
    if barge is not None:
        plt.step(barge_foundations[:, 0], barge_foundations[:, 1], label='Barge', where='post')
    if barge_2 is not None:
        plt.step(barge_foundations_2[:, 0], barge_foundations_2[:, 1], label='Barge 2', where='post')
    plt.title(f'Foundations on board')
    plt.legend()
    plt.show()

    # IV State Plot
    state_array = np.array([])
    for x in iv_data[:, 1]:
        state_array = np.append(state_array, Modes(x).name)
    # Create plot of states
    plt.step(iv_data[:, 0], state_array, label='iv', where='post')
    plt.title(f'IV state plot')
    plt.show()

    if sim.experiment.use_weather:
        # Weather data plot
        s_arr, s = sim.experiment.weather_data.get_sample(sample_no)
        max_data_frame = s.collect().filter(pl.col('hour') < max(iv_data[:, 0]))

        plt.plot(max_data_frame.get_column('hm0_sample'))
        plt.axhline(y=sim.experiment.installation_vessel.installation_limit_wave)
        plt.title('Wave Height with IV installation limit')
        plt.show()

        plt.plot(max_data_frame.get_column('tp_sample'))
        plt.axhline(y=sim.experiment.installation_vessel.installation_limit_period)
        plt.title('Tp Limit with IV installation limit')
        plt.show()


def plot_sample_data(sample_data):
    plt.rcParams.update({'figure.autolayout': True, 'figure.figsize': (10, 5)})
    plt.plot(sample_data.get_column('hm0_sample'))
    plt.title('Wave Height with IV')
    plt.show()

    plt.plot(sample_data.get_column('tp_sample'))
    plt.title('Tp Limit with IV')
    plt.show()


def plot_limit_active(plot_duration, sample_no, hm0_limit, tp_limit):
    sample = sim.experiment.weather_data.get_sample(sample_no)
    # filter out data after last event
    sample = sample.collect().filter(pl.col('hour') < plot_duration)
    sample_with_window = sample.with_columns([
        pl.when(pl.col("hm0_sample") <= hm0_limit)
        .then(pl.lit(0))
        .otherwise(pl.lit(1)).alias("above_limit_wave"),
        pl.when(pl.col("tp_sample") <= tp_limit)
        .then(pl.lit(0))
        .otherwise(pl.lit(1)).alias("above_limit_period"),
    ]
    )

    def morse(lists, labels=None):
        plt.rcParams.update({'figure.autolayout': True, 'figure.figsize': (10, 5)})
        for i, list_ in enumerate(lists):
            for j, value in enumerate(list_):
                if value:
                    plt.plot([j, j + 1], [i, i], 'r', linewidth=5.0)
        plt.axis([0, j + 1, -1, i + 1])
        if labels is None:
            labels = ['List' + str(i) for i in range(1, len(lists) + 1)]
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='minor',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off'
        )
        plt.tick_params(
            axis='y',  # changes apply to the y-axis
            which='minor',  # minor ticks are affected
            left='off',  # ticks along the left edge are off
            right='off'  # ticks along the right edge are off
        )
        plt.xlabel('index')
        # add label every 10 hours
        plt.yticks(range(len(lists)), labels)
        plt.show()

    morse([sample_with_window.get_column('above_limit_wave'),
           sample_with_window.get_column('above_limit_period')],
          labels=['Wave Limit', 'Period Limit'])


def animate(sim, iv, barge, barge_2=None):
    def get_coordinates_mode(vessel, location_dict):
        loc = vessel.location.value.get()
        if loc == 0:
            return "Not Started"
        coord = location_dict[loc.name]
        return coord[0], coord[1] + 50

    def get_weather_window_value(sim, hm0_limit, tp_limit):
        current_time = floor(sim.env.now())
        weather_window = \
            sim.env.weather_sample_lazyframe.collect().filter(pl.col('hour') == current_time).select(
                ['hm0_sample', 'tp_sample']).to_numpy()[0]
        if weather_window[0] > hm0_limit or weather_window[1] > tp_limit:
            return "LIMITED BY WEATHER", 'red'
        else:
            return "NO LIMIT", 'green'

    location_dict = {"SUPPLY_PORT": (100, 300),
                     "INTERMEDIATE_LOCATION": (400, 600),
                     "WIND_FARM": (700, 300),
                     "OFFSHORE": (400, 300)}
    sim.env.AnimateRectangle(text="Supply Port",
                             spec=(location_dict['SUPPLY_PORT'][0],
                                   location_dict['SUPPLY_PORT'][1],
                                   location_dict['SUPPLY_PORT'][0] + 150,
                                   location_dict['SUPPLY_PORT'][1] + 30))
    sim.env.AnimateRectangle(text="Intermediate Port",
                             spec=(location_dict['INTERMEDIATE_LOCATION'][0],
                                   location_dict['INTERMEDIATE_LOCATION'][1],
                                   location_dict['INTERMEDIATE_LOCATION'][0] + 150,
                                   location_dict['INTERMEDIATE_LOCATION'][1] + 30))
    sim.env.AnimateRectangle(text="Wind Farm",
                             spec=(location_dict['WIND_FARM'][0],
                                   location_dict['WIND_FARM'][1],
                                   location_dict['WIND_FARM'][0] + 150,
                                   location_dict['WIND_FARM'][1] + 30))
    # Animate vessel
    sim.env.AnimateText(text=lambda: f"IV {iv.mode.get().name}",
                        x=lambda: get_coordinates_mode(iv, location_dict)[0],
                        y=lambda: get_coordinates_mode(iv, location_dict)[1] - 20)

    if barge is not None:
        sim.env.AnimateText(text=lambda: f"Barge {barge.mode.get().name}",
                            x=lambda: get_coordinates_mode(barge, location_dict)[0],
                            y=lambda: get_coordinates_mode(barge, location_dict)[1])
        sim.env.AnimateText(
            text=lambda: f"Foundations on board Barge 1: {barge.foundations_on_board.length.get()}",
            offsety=60)
    if barge_2 is not None:
        sim.env.AnimateText(text=lambda: f"Barge 2 {barge_2.mode.get().name}",
                            x=lambda: get_coordinates_mode(barge_2, location_dict)[0],
                            y=lambda: get_coordinates_mode(barge_2, location_dict)[1] + 20)
        sim.env.AnimateText(
            text=lambda: f"Foundations on board Barge 2: {barge_2.foundations_on_board.length.get()}",
            offsety=80)

    # Animate stores
    sim.env.AnimateText(
        text=lambda: f"Foundations at Supply Port: {sim.env.stores.foundations_at_supply.length.get()}")
    sim.env.AnimateText(
        text=lambda: f"Foundations at Intermediate Location: {sim.env.stores.foundations_at_intermediate.length.get()}",
        offsety=20)
    sim.env.AnimateText(text=lambda: f"Foundations on board IV: {iv.foundations_on_board.length.get()}",
                        offsety=40)

    if sim.experiment.use_weather is True:
        sim.env.AnimateText(text=lambda: f"Weather 2.5/8: {get_weather_window_value(sim, 2.5, 8)[0]}",
                            offsety=100, textcolor=lambda: get_weather_window_value(sim, 2.5, 8)[1])
        sim.env.AnimateText(text=lambda: f"Weather 1.5/8: {get_weather_window_value(sim, 1.5, 8)[0]}",
                            offsety=120, textcolor=lambda: get_weather_window_value(sim, 1.5, 8)[1])

    sim.env.animate(True)
