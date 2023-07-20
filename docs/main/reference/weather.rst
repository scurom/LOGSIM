Weather
====================================================

The weather module contains two classes, the :class:`logsim.weather.MarkovChain` class and the :class:`logsim.weather.WeatherData` class .
The WeatherData class is designed to train, use and save a Markov model that is able to generate random weather data.
The weather data class takes different setting parameters as the number of bins, the number of hours to generate and the number of samples to generate.
When the weather module class is initialized, it will look into the cache to retrieve previously trained models (checks it based on the input file name).
The columns name should match the following names exactly: [date_time,Hm0, Tp] in order for the WeatherData class to work properly, respectively the date time in hours, the critical wave height and the peak period.
The date_time column should be in the ISO8601 format, for example: 2003-01-03T11:00.
The data should be in hourly intervals.
When new data is present and no models are present in the cache, the WeatherData class will train a new model and save it in the cache.

The WeatherData class uses the MarkovChain class to fit a Markov model to the data and to store the transition matrix.
The MarkovChain class is also used to generate a random new state based on the current state and the corresponding probabilities stored in the transition matrix.
This is done by using the :meth:`logsim.weather.MarkovChain.find_next_state` method of the MarkovChain class.


.. automodule:: logsim.weather
    :members:
    :undoc-members:
