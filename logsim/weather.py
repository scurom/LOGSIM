import hashlib
import logging
import os
import pickle
import random
from datetime import timedelta, datetime
from math import floor, ceil
from typing import Any

import numpy as np
import polars as pl
from dateutil.rrule import rrule, DAILY
from numpy import ndarray

logger = logging.getLogger(__name__)


class MarkovChain:
    def __init__(self, n: int):
        """
        This class enables the creation of a Markov Chain of order n. This class can be saved to the disk. In this way
        the markov chain can be used to create synthetic weather data by sampling from the transition matrix.

        :param int n: The order of the Markov Chain
        """
        self.transitions = {}
        self.possible_start_states = {}
        self.start_matrix = None
        self.n = n
        self.last_n_state = None
        self.mc_states = {}
        self.mc_lookup_state = {}
        self.states = None
        self.possible_start_state_array = None
        self.bin_low_values = {}
        self.bin_high_values = {}

    def fit(self, data: np.array) -> None:
        """
        This function fits the Markov Chain to the given data. The data is a numpy array of integers that represent the
        states of the weather in the given order. The weights are optional and can be used to weight the data.

        :param np.array data: The data to fit the Markov Chain to.
        """
        # Save last n states to avoid going to these states
        self.last_n_state = tuple(data[-self.n:])

        for i in range(len(data) - self.n):
            # Get current state
            current_state = tuple(data[i:i + self.n])
            next_state = data[i + self.n]

            # Add current state to state count
            if current_state not in self.possible_start_states:
                # Check if not end state
                self.possible_start_states[current_state] = 1
            else:
                self.possible_start_states[current_state] += 1

            # Check if current from state is already in transitions
            if current_state not in self.transitions:
                # Add new transition
                self.transitions[current_state] = {}

            # Check if current to next state is not the latest state
            current_state_list = list(current_state[-self.n + 1:])
            current_state_list.append(next_state)
            new_comb_state = tuple(current_state_list)

            if new_comb_state != self.last_n_state:
                if next_state not in self.transitions[current_state]:
                    # Add new transition
                    self.transitions[current_state][next_state] = 1
                else:
                    # Update transition
                    self.transitions[current_state][next_state] += 1

        # Creating mapping dictionary of row number and state tuple key from possible_start_states
        self.mc_states = {i: state for i, state in enumerate(self.possible_start_states.keys())}

        # save lookup table with mc_states inverse
        self.mc_lookup_state = {state: i for i, state in self.mc_states.items()}

        # Calculate start transition probabilities
        self.possible_start_states = {k: v / sum(self.possible_start_states.values()) for k, v in
                                      self.possible_start_states.items()}

        # calculate transition probabilities within each state
        for state in self.transitions:
            self.transitions[state] = {k: v / sum(self.transitions[state].values()) for k, v in
                                       self.transitions[state].items()}

        self.possible_start_state_array = np.array([list(self.mc_states.keys()),
                                                    list(self.possible_start_states.values())]).T

        logger.info("Done fitting Markov Chain")

    def find_next_state(self, current_row: int) -> tuple[int, int, tuple[int, int]]:
        """
        This helper function can be used to return a random new state based on the current state and the corresponding
        probabilities stored in the transition matrix.

        :param current_row: The row number of the transition matrix that corresponds to the current state of the
            weather (row, state mappings are stored in self.possible_from_states)
        :return: The new state, the new row number and the new combination of states. The new row is the row number of
            the transition matrix that corresponds to the new state of the weather.
            The new state is the state number of the new state.
            The new combination of states is the combination of the new state and the previous state in a tuple.
        """
        # Set current row and current state
        current_state = self.mc_states[current_row]
        current_row = self.transitions[current_state]

        choices = np.array(list(current_row.keys()))
        probabilities = np.array(list(current_row.values()))

        # Take random new state according to probabilities
        new_state = np.random.choice(choices, p=probabilities)

        # new combination is combination of old state t minus 1 and now
        if self.n == 1:
            new_comb_state = new_state
        else:
            current_state_list = list(current_state[-self.n + 1:])
            current_state_list.append(new_state)
            new_comb_state = tuple(current_state_list)

        # Get the new row based on the new combination of states
        new_row = self.mc_lookup_state[new_comb_state]

        return new_state, new_row, new_comb_state


class WeatherData:
    def __init__(self, file_name: str, start_day: int, start_month: int,
                 synthetic: bool = True, synthetic_data_samples: int = None,
                 train_model: bool = False, bin_tuple: tuple[int, int] = (15, 15), markov_order: int = 2,
                 timedelta_days: int = 15, sample_hours: int = 10000, scale_factor: float = 1.0,
                 experiment_cache=None) -> None:
        """This class contains weather data to be sampled within the simulation. Make sure the data is loaded with a
        1hour interval with a ISO8601 formatted date_time column

        :param file_name: The name of the file to load the
        data from (make sure ISO8601 formatted date_time column is present)
        :param start_day: The day of the month to
        start the synthetic data from
        :param start_month: The month to start the synthetic data from
        :param synthetic: Whether to use synthetic data or not
        :param synthetic_data_samples: The number of samples to create
        :param train_model: Overwrite the cached markov chain if available and retrain it
        :param bin_tuple: The bin size to use for the markov chain in order (Hs, Tp)
        :param markov_order: The order of the markov chain
        :param timedelta_days: The number of days to use for the timedelta when standardizing the data
        :param sample_hours: The number of hours to create for each sample
        :param scale_factor: The scale factor used to scale samples up or down
        :param experiment_cache: A string that defines the file name of the cache, this is required to generate
        own data within a multiprocessing pool in order to avoid writing and reading the same file.
        """

        self.file_name: str = file_name
        self.start_day = start_day
        self.start_month = start_month
        self.input_data: pl.LazyFrame = self._read_input_data()
        self._hash: str = self._hash_file()
        self.sample_hours = sample_hours
        self.no_samples = None
        self._weather_data: pl.LazyFrame | None = None
        self.synthetic = synthetic
        self.scale_factor = scale_factor
        self.experiment_cache: int | None = experiment_cache

        if synthetic:
            # Check if synthetic_data_samples is set
            if synthetic_data_samples is None:
                raise ValueError("synthetic_data_samples is not set, specify the number of samples to create")
            self.no_samples = synthetic_data_samples

            self._mc: MarkovChain | None = None
            # try to load the weather_data_from_cache and otherwise will generate it
            if not train_model:
                self._retrieve_synthetic_data()
            else:
                # Train the markov chain and generate the synthetic data
                self.train_markov_model(bin_tuple=bin_tuple, markov_order=markov_order, timedelta_days=timedelta_days)
                self.generate_synthetic_data()

        else:
            logger.info("Input data is used as sample")
            self._mc: MarkovChain | None = None
            self._weather_data: pl.LazyFrame = self.split_input_data()

    def __repr__(self) -> str:
        """Returns a string representation of the WeatherData object"""
        return f"WeatherData(start = {self.start_day}-{self.start_month})"

    def _read_input_data(self) -> pl.LazyFrame:
        """Reads the input data from the file_name and returns a polars DataFrame"""
        df = (
            pl.scan_csv(self.file_name)
            .with_columns(
                [
                    pl.col("date_time").str.strptime(
                        pl.Datetime, '%Y-%m-%dT%H:%M').alias('date_time')
                ])
            .with_columns(
                pl.col('date_time').cast(pl.Date).alias('date'),
                pl.col('date_time').cast(pl.Time).alias('time')
            )
            .drop('date_time')
        )
        return df

    def _hash_file(self) -> str:
        """Hashes the file_name and returns the hash in order to check cache files"""
        # Hash the csv file
        with open(self.file_name, "rb") as f:
            data = f.read()
            sha256hash = hashlib.md5(data).hexdigest()

        return sha256hash

    def _write_cache(self, data: dict, file_type: str) -> None:
        """
        Writes the data to a cache pickle file (containing the object)

        :param data: The data to write to the cache file
        :file_type: The type of data to write to the cache file (matrix, synthetic_data)
        """
        # If experiment cache (so multiprocessing cache) is set, use that as prefix
        if self.experiment_cache is not None:
            file_name = f'.cache/{self.experiment_cache}_{self._hash}_{file_type}.pkl'
        else:
            file_name = f'.cache/{self._hash}_{file_type}.pkl'
        if not os.path.exists('.cache'):
            os.makedirs('.cache')
        if os.path.exists(file_name):
            os.remove(file_name)
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)

    def _read_cache(self, file_type: str) -> dict | None:
        """
        Reads the cache file and returns the data

        :param file_type: The type of data to read from the cache file (matrix, synthetic_data)
        :return: The data from the cache file as a dict
        """
        if self.experiment_cache is not None and file_type != 'matrix':
            file_name = f'.cache/{self.experiment_cache}_{self._hash}_{file_type}.pkl'
        else:
            file_name = f'.cache/{self._hash}_{file_type}.pkl'

        if not os.path.exists(file_name):
            return None
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        return data

    def _generate_random_sample(self, hours) -> np.array:
        """
        Given a markov chain model matrix, generate a random sample of a given length by randomly drawing from the
        markov matrix according to the given probabilities for each state.
        The random drawn state is repeated for the drawn duration.

        :param hours: Number of hours to generate
        :return: Array of random states that are drawn from the markov chain (Hs and Tp)
        """
        x_t_list = []
        state_list = []

        # Get probabilities of occurrences of each start state
        start_choices = self._mc.possible_start_state_array[:, 0]
        start_prob = self._mc.possible_start_state_array[:, 1]

        # Take random start state according to probabilities
        current_row = np.random.choice(start_choices, p=start_prob)
        hours_left = hours

        for x in range(0, hours):
            draw_values = []
            new_state, new_row, new_comb_state = self._mc.find_next_state(current_row)
            current_row = new_row
            state_list.append(new_state)

            # Get the low and high values for the bins of this state
            # Get the low and high values for the bins of this state
            low, high = self._mc.bin_low_values[new_state], self._mc.bin_high_values[new_state]

            # Draw random duration for this state between low high
            # Exclude high value itself from draw
            choice_list = list(range(int(low[2]), int(high[2])))

            duration = random.choice(choice_list)

            # For each column (h0, tp) take random draw between bin definitions
            for col in range(0, 2):
                draw_values.append(random.uniform(low[col], high[col]))

            # Repeat this value for the duration of the state
            for _ in range(0, duration):
                x_t_list.append(draw_values)

            hours_left = hours_left - duration

            # If enough hours simulated break
            if hours_left <= 0:
                break

        # Return array of sample, make sure not more than max hours
        return np.array(x_t_list)[:self.sample_hours]

    def generate_synthetic_data(self) -> pl.LazyFrame | None:
        """
        Generate synthetic data based on the transition probabilities of a cached Markov Chain model. The set
        parameters of the class are used to generate the appropriate amount of samples with the appropriate size. The
        synthetic data is generated based on the standardized model. After generation the data standardization is
        reverted based on the given start day and month. The generated data is cached and is retrieved if using the same
        parameters.

        :return: A Polars LazyFrame containing the generated synthetic data samples or None if
            cached data is available.
        """
        logger.info("Start generating synthetic data")
        # Read matrix data
        logger.info("Reading matrix data")
        res = self._read_cache('matrix')
        if res is None:
            raise ValueError('No Markov Chain model available. Add `train_model=True` parameter to train model.')
        else:
            self._mc = res['mc']

        logger.info("Creating data frame of transformation_table")
        self._mc.transformation_table = pl.DataFrame(self._mc.transformation_table)

        def build_day_array(arr, start_month, start_day) -> np.array:
            """Build day array (so transformation table) based on first sample"""
            n_rows = arr.shape[0]
            n_full_groups = n_rows // 24
            n_remaining_rows = n_rows % 24

            days_to_consider = n_full_groups
            repeating_numbers = np.repeat(np.arange(0, n_full_groups), 24)
            if n_remaining_rows > 0:
                days_to_consider += 1
                repeating_numbers = np.concatenate([repeating_numbers, np.full(n_remaining_rows, n_full_groups)])

            # create dict with dates
            days_list = [f"{date.day}-{date.month}" for date in
                         (datetime(2023, start_month, start_day) + timedelta(n) for n in
                          range(days_to_consider))]
            days_dict = {ind: days_list[ind] for ind in range(0, len(days_list))}

            repeating_days_list = []
            for row in repeating_numbers:
                repeating_days_list.append(days_dict[row])

            return np.array(repeating_days_list)

        # Variables used for sampling
        sample_list = []
        df_samples = None
        sample_no = 1

        # Generate 1000 samples
        for x in range(0, self.no_samples):
            sample_list.append(self._generate_random_sample(self.sample_hours))
            logger.info(f"Generated sample {x + 1} of {self.no_samples}")

        # Based on first sample generate array of days
        repeating_days = build_day_array(sample_list[0], start_month=self.start_month, start_day=self.start_day)

        # Transform standardized generated samples back to original scale
        for sample in sample_list:
            sample_with_days = np.concatenate([sample, repeating_days.reshape(-1, 1)], axis=1)
            df_sample = pl.DataFrame(sample_with_days, schema={'hm0_standard': pl.Float64, 'tp_standard': pl.Float64,
                                                               'day_month': pl.Utf8})
            df_sample = df_sample.join(self._mc.transformation_table, on='day_month')
            df_sample = df_sample.with_columns([
                (pl.col('hm0_std') * pl.col('hm0_standard') + pl.col('hm0_mean')).alias('hm0_sample'),
                (pl.col('tp_std') * pl.col('tp_standard') + pl.col('tp_mean')).alias('tp_sample'),
                pl.lit(sample_no).alias('sample')
            ]).select('hm0_sample', 'tp_sample', 'sample', 'day_month').with_row_count(name='hour')
            # Add dataframe with sample to dataframe with all samples
            if sample_no == 1:
                df_samples = df_sample
            else:
                df_samples = pl.concat([df_samples, df_sample], how="diagonal")
            sample_no = sample_no + 1

        # Set all minus values to 0 in df_samples (can occur due to rounding)
        df_samples = df_samples.with_columns([
            pl.col('hm0_sample').clip_min(0).alias('hm0_sample'),
            pl.col('tp_sample').clip_min(0).alias('tp_sample')])

        # Write samples to cache
        data_to_cache = {'df_samples': df_samples, 'synthetic_data_samples': self.no_samples,
                         'synthetic_data_hours': self.sample_hours,
                         'start_month': self.start_month, 'start_day': self.start_day,
                         'scale_factor': self.scale_factor}

        # Write samples to cache
        self._write_cache(data_to_cache, file_type='synthetic_data')
        # Save samples in memory
        self._weather_data = df_samples.lazy()
        return

    def _retrieve_synthetic_data(self) -> pl.LazyFrame | None:
        """
        Retrieve synthetic data from cache and check if settings match. If settings do not match, generate new data.

        :return: LazyFrame containing the synthetic data or None if no data is available in cache, will call generation
        function.
        """
        file_res = self._read_cache(file_type='synthetic_data')
        if file_res is None:
            logger.info('No synthetic data available in cache, generate data first')
            self.generate_synthetic_data()
            return None
        # check if settings match
        elif file_res['synthetic_data_samples'] != self.no_samples or \
                file_res['synthetic_data_hours'] != self.sample_hours or \
                file_res['start_month'] != self.start_month or \
                file_res['start_day'] != self.start_day or \
                file_res['scale_factor'] != self.scale_factor:
            logger.info('Synthetic data in cache does not match current settings, generate data first')
            self.generate_synthetic_data()
            return
        else:
            self._weather_data = file_res['df_samples'].lazy()
            return

    def split_input_data(self):
        """Split training data into yearly samples"""
        date_input = self.input_data.collect().select(['Hm0', 'Tp', 'date', 'time']).to_pandas()
        date_input.columns = ['hm0_sample', 'tp_sample', 'date', 'time']
        unique_years = date_input['date'].dt.year.unique()
        date_values = date_input['date'].dt.date.unique()
        sample_no = 1
        for year in unique_years:
            start_date = datetime(year, self.start_month, self.start_day)
            end_date = datetime(year, self.start_month, self.start_day) + timedelta(days=ceil(self.sample_hours / 24))
            # check if start date and end date are in the input data
            # Check below is not working, always false fix this
            if start_date.date() not in date_values or end_date.date() not in date_values:
                logger.warning(f'No data for start date {start_date}')
                continue
            else:
                # Get sample data, defined by start and end date
                # Add column with sample number
                date_input.loc[
                    (date_input['date'] >= start_date) & (date_input['date'] < end_date), 'sample'] = sample_no

            # Increment sample number
            sample_no += 1
        # Remove rows with no sample number
        date_input = date_input.dropna(subset=['sample'])
        # Convert sample number to integer
        date_input['sample'] = date_input['sample'].astype(int)
        # add for each sample a count of the row
        date_input['hour'] = date_input.groupby('sample').cumcount() + 1
        # Only keep rows with hour number smaller than sample hours
        date_input = date_input[date_input['hour'] <= self.sample_hours]
        # Add column with day and month in format {day_month}
        date_input['day_month'] = date_input['date'].dt.strftime('%d-%m')
        date_input = date_input[
            ['hour', 'hm0_sample', 'tp_sample', 'sample', 'day_month', 'date', 'time']]
        self.no_samples = date_input['sample'].max()
        return pl.LazyFrame(date_input)

    def get_sample(self, sample_no: int) -> (np.array, pl.LazyFrame):
        """
        Get a sample from the synthetic data by filtering the LazyFrame on the sample number. Tests if the sample number
        is valid and if the synthetic data is available in the cache.

        :param int sample_no: The sample number to retrieve
        :return: Polars LazyFrame with the sample data
        """
        if self._weather_data is None:
            raise ValueError(
                'No synthetic data available in cache, generate data first with function \'generate_synthetic_data()\'')

        if sample_no < 1 or sample_no > self.no_samples:
            raise ValueError(
                f'Invalid sample number. Please choose a number between 1 and {self.no_samples}')

        # Check if scaling is required
        if self.scale_factor != 1.0:
            sample_lazyframe = (self._weather_data.filter(pl.col('sample') == sample_no)
                                .with_columns([pl.col('hm0_sample') * self.scale_factor,
                                               pl.col('tp_sample') * self.scale_factor]))
        else:
            sample_lazyframe = self._weather_data.filter(pl.col('sample') == sample_no)

        sample_arr = sample_lazyframe.select('hour', 'hm0_sample', 'tp_sample').collect().to_numpy()
        return sample_arr, sample_lazyframe

    def train_markov_model(self, bin_tuple: tuple[int, int] = (15, 15), markov_order: int = 2,
                           timedelta_days: int = 15) -> None:
        """
        Train markov model based on synthetic data. The trained markov model (based on the MarkovChain class) is
        saved in the cache and is retrieved when generating synthetic data with the function
        :func:`generate_synthetic_data()`. This allows for faster generation of synthetic data without having to
        train the markov model every time. Specify the number of bins for Hm0 and Tp and the markov order to
        train. The timedelta_days parameter can be used to influence the offset used for standardizing the weather
        data. The higher the value, the more days are used for standardizing the weather data (by default 15 days
        back and forth). This function does not return anything, but saves the trained markov model in the cache.

        :param tuple[int, int] bin_tuple: Tuple with number of bins for Hs and Tp
        :param int markov_order: Order of markov model
        :param int timedelta_days: Number of days to look back and forth to standardize hourly weather data
        """

        def train_create_transformation_table(df_input) -> pl.DataFrame:
            # Iterate over each day/month based on year 2023
            transform_data = []
            for mid_date in (datetime(2023, 1, 1) + timedelta(n) for n in range(365)):
                # Go timedelta_days back and forth to standardize on
                start = mid_date - timedelta(timedelta_days)
                end = mid_date + timedelta(timedelta_days)
                # Create all days in between
                day_month_list = pl.DataFrame([(int(dt.day), int(dt.month)) for dt in rrule(
                    DAILY, dtstart=start, until=end)], schema=['day', 'month'])
                # Select only days that consist of these days in between
                df_filter = df_input.join(day_month_list, on=['day', 'month'], how='inner')
                # Create new dataframe with all mean and std values
                df_filter = (
                    {
                        "day": mid_date.day,
                        "month": mid_date.month,
                        "hm0_mean": df_filter.select('Hm0').to_numpy().mean(),
                        "hm0_std": df_filter.select('Hm0').to_numpy().std(),
                        "tp_mean": df_filter.select('Tp').to_numpy().mean(),
                        "tp_std": df_filter.select('Tp').to_numpy().std(),
                    }
                )
                transform_data.append(df_filter)

            # Create dataframe with all mean and std values calculated for each day/month
            calc_transformation_table = pl.DataFrame(transform_data, orient='row').with_columns([
                (
                    pl.concat_str([pl.col('day'), pl.col('month')], sep="-").alias('day_month')
                )
            ])
            return calc_transformation_table

        def train_create_states(df_input,
                                bins_def_input,
                                bin_list_input) -> tuple[list[Any], ndarray, ndarray, ndarray]:

            # Create separate dictionary with all bin definitions (low and high)
            bin_definitions = {}
            for col in bins_def_input:
                high = []
                for bin_no in range(0, len(bins_def_input[col])):
                    # if at last bin start
                    if bin_no == len(bins_def_input[col]) - 1:
                        high.append(df_input.get_column(col).max())
                    else:
                        high.append(bins_def_input[col][bin_no + 1])
                bin_definitions[col] = {}
                bin_definitions[col]['low'] = bins_def_input[col]
                bin_definitions[col]['high'] = np.array(high)

            # Calculate duration/persistence of each state and add to array
            arr = df_input.select(['Hm0Class', 'TpClass']).to_numpy()
            # Split the array into sub-arrays where a value change happens
            sub_arrays = np.split(arr, np.where(np.diff(arr, axis=0) != 0)[0] + 1)

            res = []
            for subarray in sub_arrays:
                # if subarray not empty
                if subarray.size != 0:
                    # add first item of array and length of the array to list
                    res.append(np.append(subarray[0], len(subarray)))
            res_array = np.array(res)

            # Bin durations to reduce complexity -> TODO: Make this not hardcoded?
            duration_bin_low = np.array(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80, 90])
            duration_bin_high = np.array(
                [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100])

            bin_definitions['duration'] = {}
            bin_definitions['duration']['low'] = duration_bin_low
            bin_definitions['duration']['high'] = duration_bin_high
            # Digitize the durations to the bins
            res_array[:, 2] = np.digitize(res_array[:, 2], duration_bin_low)
            # subtract 1 to get the correct bin number
            res_array[:, 2] = res_array[:, 2] - 1

            # Define state limits
            bin_low_values_array = np.array(
                [(x, y, z) for x in bin_definitions['hm0_standard']['low'] for y in
                 bin_definitions['tp_standard']['low']
                 for z in bin_definitions['duration']['low']])
            bin_high_values_array = np.array(
                [(x, y, z) for x in bin_definitions['hm0_standard']['high'] for y in
                 bin_definitions['tp_standard']['high']
                 for z in bin_definitions['duration']['high']])

            # Define all possible states in the form of (Hm0, Tp, Duration)
            states_arr = np.array(
                [(x, y, z) for x in range(0, bin_list_input[0]) for y in range(0, bin_list_input[1])
                 for z in range(0, len(duration_bin_low))])

            # Get state number for each row in res_array
            state_ind_list = []
            for row_entry in res_array:
                condition = (states_arr[:, 0] == row_entry[0]) & (states_arr[:, 1] == row_entry[1]) & (
                        states_arr[:, 2] == row_entry[2])
                state_ind_list.append(np.where(condition)[0][0])

            return state_ind_list, states_arr, bin_low_values_array, bin_high_values_array

        def train_create_bins(df_input, column_list_input, no_bins_list) -> tuple[np.ndarray, dict]:
            bin_number_value = []
            bins_def_dict = {}
            for x in range(0, len(column_list_input)):
                data = df_input.get_column(column_list_input[x]).to_numpy()
                bins = np.linspace(data.min(), data.max(), no_bins_list[x], endpoint=False)
                digitized = np.digitize(data, bins)
                # subtract value for each bin, ease indexing
                digitized = digitized - 1
                bin_number_value.append(digitized)
                bins_def_dict[column_list_input[x]] = bins
            return np.column_stack(bin_number_value), bins_def_dict

        # check if training is needed
        if self._weather_data is not None:
            logger.warning(
                'Trained model available in cache, skipping training step. Overwrite with overwrite=True in class '
                'constructor.')
            return

        logger.info("Starting training of synthetic data")

        # Variables
        column_list = ['hm0_standard', 'tp_standard']

        df = self.input_data.collect()
        bin_list = list(bin_tuple)

        # Create day/month columns
        df = df.with_columns([
            pl.col("date").dt.day().cast(pl.Int64).alias('day'),
            pl.col("date").dt.month().cast(pl.Int64).alias('month').alias('month')]
        )

        # Get global table with transformation information (due to standardization) for columns Hm0, Tp
        transformation_table = train_create_transformation_table(df)
        # Join transform table
        df = df.join(transformation_table, how='left', on=['day', 'month'])

        # standardize data
        df = df.with_columns(
            [
                ((pl.col('Hm0') - pl.col('hm0_mean')) / pl.col('hm0_std')).alias('hm0_standard'),
                ((pl.col('Tp') - pl.col('tp_mean')) / pl.col('tp_std')).alias('tp_standard')
            ]
        )

        # Drop nulls again (as some values are from leap years)
        df = df.drop_nulls()

        # Create bins and add to polars dataframe
        bin_data, bins_def = train_create_bins(df, column_list_input=column_list, no_bins_list=bin_list)
        class_data = pl.from_numpy(bin_data, schema=['Hm0Class', 'TpClass'])
        df = pl.concat([df, class_data], how="horizontal")

        state_ind, states, bin_low_values, bin_high_values = train_create_states(df, bins_def, bin_list)

        mc = MarkovChain(n=markov_order)
        mc.fit(state_ind)

        # set required data. Now bin low and high values needed for sampling.
        mc.transformation_table = transformation_table.to_arrow()
        mc.states = states
        mc.bin_low_values = bin_low_values
        mc.bin_high_values = bin_high_values

        data_to_store = {'mc': mc}
        logger.info("Finished training of synthetic data, writing to cache")
        # save data to cache
        self._write_cache(data_to_store, file_type="matrix")
        return


# Below is code for generic utility functions
def get_next_window(sample: np.array, current_hour: float, window_size: int | float, hm0_limit: float,
                    tp_limit: float) -> float:
    """
    Given a specific hour and a window size this function will return the first possible window.
    First the case is checked when the start is immediately happening. If not possible the hours are checked one
    by one until the next weather window is found.

    :param sample: The numpy array with the weather data
    :param current_hour: The current simulation hour
    :param window_size: The size of the window that is required
    :param hm0_limit: The limit for the Hm0 to determine the next possible window with
    :param tp_limit: The limit for the Tp to determine the next possible window with
    :return float: The waiting time until the next possible window as decimal hour. An exception is raised if no window
        is found.
    """
    # Check if now is possible
    start_val = current_hour
    end_val = window_size + current_hour
    start_floor = floor(start_val)
    end_floor = floor(end_val)

    max_hm0_1, max_tp_1 = get_max_array_val(sample, [start_floor], [end_floor], 0)

    if (max_hm0_1 <= hm0_limit) & (max_tp_1 <= tp_limit):
        return 0

    # Otherwise loop until possible
    current_check_hour = ceil(current_hour)
    start_floor = 0
    end_floor = floor(window_size)

    while True:
        max_hm0_1, max_tp_1 = get_max_array_val(sample, [start_floor],
                                                [end_floor],
                                                current_check_hour)
        if (max_hm0_1 <= hm0_limit) & (max_tp_1 <= tp_limit):
            return round(current_check_hour - current_hour, 2)
        else:
            current_check_hour += 1


def get_next_window_3_limits(sample: np.array, current_hour: float, window_sizes: list[int | float],
                             hm0_limits: list[float],
                             tp_limits: list[float]) -> float:
    """
    Given a specific hour and 3 window sizes this function will return the first possible window.
    First the case is checked when the start is immediately happening. If not possible the hours are checked one
    by one until the next weather window is found.

    :param sample: The numpy array with the weather data
    :param current_hour: The current simulation hour
    :param window_sizes: The sizes of the windows that are required
    :param hm0_limits: The limits for the Hm0 to determine the next possible window with
    :param tp_limits: The limits for the Tp to determine the next possible window with
    :return float: The waiting time until the next possible window as decimal hour. An exception is raised if no window
        is found.
    """

    # Check if now is possible
    start_val = np.array(window_sizes[:-1])
    start_val = np.insert(start_val, 0, current_hour).cumsum()
    end_val = np.array(window_sizes).cumsum() + current_hour
    start_floor = [int(x) for x in np.floor(start_val)]
    end_floor = [int(x) for x in np.floor(end_val)]
    max_hm0_1, max_tp_1, max_hm0_2, max_tp_2, max_hm0_3, max_tp_3 = get_max_array_val(sample, start_floor, end_floor, 0)

    if (max_hm0_1 <= hm0_limits[0]) & (max_tp_1 <= tp_limits[0]) & (max_hm0_2 <= hm0_limits[1]) & (
            max_tp_2 <= tp_limits[1]) & (max_hm0_3 <= hm0_limits[2]) & (max_tp_3 <= tp_limits[2]):
        return 0

    # Otherwise loop until possible
    current_check_hour = ceil(current_hour)
    start_val = np.array(window_sizes[:-1])
    start_val = np.insert(start_val, 0, 0).cumsum()
    end_val = np.array(window_sizes).cumsum()
    start_floor = [int(x) for x in np.floor(start_val)]
    end_floor = [int(x) for x in np.floor(end_val)]

    while True:
        max_hm0_1, max_tp_1, max_hm0_2, max_tp_2, max_hm0_3, max_tp_3 = get_max_array_val(sample, start_floor,
                                                                                          end_floor,
                                                                                          current_check_hour)
        if (max_hm0_1 <= hm0_limits[0]) & (max_tp_1 <= tp_limits[0]) & (max_hm0_2 <= hm0_limits[1]) & (
                max_tp_2 <= tp_limits[1]) & (max_hm0_3 <= hm0_limits[2]) & (max_tp_3 <= tp_limits[2]):
            return round(current_check_hour - current_hour, 2)
        else:
            current_check_hour += 1


def get_max_array_val(np_weather, start_floor, end_floor, current_check_hour):
    if len(start_floor) == 1:
        max_hm0_1 = np_weather[start_floor[0] + current_check_hour:end_floor[0] + current_check_hour + 1, 1].max()
        max_tp_1 = np_weather[start_floor[0] + current_check_hour:end_floor[0] + current_check_hour + 1, 2].max()
        return max_hm0_1, max_tp_1
    elif len(start_floor) == 2:
        max_hm0_1 = np_weather[start_floor[0] + current_check_hour:end_floor[0] + current_check_hour + 1, 1].max()
        max_tp_1 = np_weather[start_floor[0] + current_check_hour:end_floor[0] + current_check_hour + 1, 2].max()
        max_hm0_2 = np_weather[start_floor[1] + current_check_hour:end_floor[1] + current_check_hour + 1, 1].max()
        max_tp_2 = np_weather[start_floor[1] + current_check_hour:end_floor[1] + current_check_hour + 1, 2].max()
        return max_hm0_1, max_tp_1, max_hm0_2, max_tp_2
    elif len(start_floor) == 3:
        max_hm0_1 = np_weather[start_floor[0] + current_check_hour:end_floor[0] + current_check_hour + 1, 1].max()
        max_tp_1 = np_weather[start_floor[0] + current_check_hour:end_floor[0] + current_check_hour + 1, 2].max()
        max_hm0_2 = np_weather[start_floor[1] + current_check_hour:end_floor[1] + current_check_hour + 1, 1].max()
        max_tp_2 = np_weather[start_floor[1] + current_check_hour:end_floor[1] + current_check_hour + 1, 2].max()
        max_hm0_3 = np_weather[start_floor[2] + current_check_hour:end_floor[2] + current_check_hour + 1, 1].max()
        max_tp_3 = np_weather[start_floor[2] + current_check_hour:end_floor[2] + current_check_hour + 1, 2].max()
        return max_hm0_1, max_tp_1, max_hm0_2, max_tp_2, max_hm0_3, max_tp_3


def get_next_window_2_limits(sample: np.array, current_hour: float, window_sizes: list[int | float],
                             hm0_limits: list[float],
                             tp_limits: list[float]) -> float:
    """
    Given a specific hour and 2 window sizes this function will return the first possible window.
    First the case is checked when the start is immediately happening. If not possible the hours are checked one
    by one until the next weather window is found.

    :param sample: The numpy array with the weather data
    :param current_hour: The current simulation hour
    :param window_sizes: The sizes of the windows that are required
    :param hm0_limits: The limits for the Hm0 to determine the next possible window with
    :param tp_limits: The limits for the Tp to determine the next possible window with
    :return float: The waiting time until the next possible window as decimal hour. An exception is raised if no window
        is found.
    """

    # Check if now is possible
    start_val = np.array(window_sizes[:-1])
    start_val = np.insert(start_val, 0, current_hour).cumsum()
    end_val = np.array(window_sizes).cumsum() + current_hour
    start_floor = [int(x) for x in np.floor(start_val)]
    end_floor = [int(x) for x in np.floor(end_val)]
    max_hm0_1, max_tp_1, max_hm0_2, max_tp_2 = get_max_array_val(sample, start_floor, end_floor, 0)

    if (max_hm0_1 <= hm0_limits[0]) & (max_tp_1 <= tp_limits[0]) & (max_hm0_2 <= hm0_limits[1]) & (
            max_tp_2 <= tp_limits[1]):
        return 0

    # Otherwise loop until possible
    current_check_hour = ceil(current_hour)
    start_val = np.array(window_sizes[:-1])
    start_val = np.insert(start_val, 0, 0).cumsum()
    end_val = np.array(window_sizes).cumsum()
    start_floor = [int(x) for x in np.floor(start_val)]
    end_floor = [int(x) for x in np.floor(end_val)]

    while True:
        max_hm0_1, max_tp_1, max_hm0_2, max_tp_2 = get_max_array_val(sample, start_floor,
                                                                     end_floor,
                                                                     current_check_hour)
        if (max_hm0_1 <= hm0_limits[0]) & (max_tp_1 <= tp_limits[0]) & (max_hm0_2 <= hm0_limits[1]) & (
                max_tp_2 <= tp_limits[1]):
            return round(current_check_hour - current_hour, 2)
        else:
            current_check_hour += 1
