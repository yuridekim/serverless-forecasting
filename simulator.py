import math
import pandas as pd
import numpy as np
import logging
import sys

from dataclasses import dataclass, field
from datetime import timedelta

from tqdm import tqdm

logging.getLogger("prophet.plot").disabled = True
logging.getLogger("cmdstanpy").disabled = True
from prophet import Prophet


MAX_TIMESTAMP = pd.Timestamp(sys.maxsize)

# Parameterizing normal distributions for different events
# in a function invocation's lifecycle
@dataclass
class Distribution:
    acquire: tuple[int, int]
    execution: tuple[int, int]
    release: tuple[int, int]

# Functions can be invoked using a container
@dataclass
class Container:
    creation: pd.Timestamp
    expiration: pd.Timestamp
    in_use_until: pd.Timestamp
    active_time: pd.Timedelta

# Stores all metadata from an entire function's simulation
@dataclass
class Function:
    warming_period: int
    min_num_containers: int
    containers: list[Container] = field(default_factory=list)
    predictions: list[pd.DataFrame] = field(default_factory=list)
    history: pd.DataFrame = field(default_factory=
                                  lambda: pd.DataFrame({
                                      "ds": [], 
                                      "y": [], 
                                      "available_containers": []}))
    cur_time: pd.Timestamp = pd.Timestamp(0)
    last_predictor_update: pd.Timestamp = pd.Timestamp(0)



def s(num: float) -> pd.Timedelta:
    return pd.Timedelta(seconds=num)

class ServerlessSimulator:
    # Direct DataFrame from trace
    _invocations: pd.DataFrame

    # Each function gets its own metadata
    _functions: dict[str, Function]

    # Distributions to pull from depending on if it's a warm or cold start
    _warm_start_dist: Distribution
    _cold_start_dist: Distribution

    # Default values to create functions with
    _default_warming_period: int
    _default_min_num_containers: int

    # Knobs for prediction
    _predictor_interval: int
    _predictor_history_interval: int
    _granularity: int

    def __init__(self, 
            invocations: pd.DataFrame, 
            warm_start_dist: Distribution, 
            cold_start_dist: Distribution, 
            default_warming_period: int, 
            default_min_num_containers: int, 
            predictor_interval: int, 
            predictor_history_interval: int,
            granularity: int
        ):
        
        self._invocations = invocations
        self._warm_start_dist = warm_start_dist
        self._cold_start_dist = cold_start_dist
        self._default_warming_period = default_warming_period
        self._default_min_num_containers = default_min_num_containers
        self._predictor_interval = predictor_interval
        self._predictor_history_interval = predictor_history_interval
        self._granularity = granularity

        self._functions = {}
        # self.reset()

    def reset(self):
        # Create all functions with default warming period and min warmed containers
        self._functions = {}
        for _, inv in self._invocations.iterrows():
            if inv["func"] not in self._functions.keys():
                self._functions[inv["func"]] = Function(self._default_warming_period, self._default_min_num_containers)

    # Actually do the simulation
    def run(self, func=None, samples=None, with_model=True):
        # Each invocation has a log that tracks simulated run data
        logs = pd.DataFrame({
                "ds": [], 
                "func": [], 
                "is_warm_start": [],
                "acquire_time": [],
                "execute_time": [],
                "release_time": []
            })

        container_logs = pd.DataFrame({
            "creation": [],
            "expiration": [],
            "active_time": []
        })

        invocations = self._invocations
        if func is not None:
            invocations = invocations[invocations["func"] == func]

        if samples is not None:
            invocations = invocations.head(samples)

        for _, inv in tqdm(invocations.iterrows(), total=len(invocations)):
            if inv["func"] not in self._functions.keys():
                self._functions[inv["func"]] = Function(self._default_warming_period, self._default_min_num_containers)
            
            f = inv["func"]
            fun = self._functions[f]
            
            # Create history until current timestamp, add one invocation in the last bin
            # to signify that a function was run at this time
            self._update_state(f, inv["start_timestamp"], with_model, container_logs)

            # Get an available container—if none available, it's a cold start
            is_warm_start, (a, e, r) = self._get_available_container(f, inv["duration"])

            # Log simulation data
            logs.loc[len(logs)] = {
                "ds": fun.cur_time, 
                "func": f, 
                "is_warm_start": is_warm_start,
                "acquire_time": fun.cur_time + s(a),
                "execute_time": fun.cur_time + s(a + e),
                "release_time": fun.cur_time + s(a + e + r)
            }

        self._cleanup(container_logs)

        return logs, container_logs
    
    def _update_state(self, f: str, timestamp: pd.Timestamp, with_model: bool, container_logs: pd.DataFrame):
        fun = self._functions[f]

        # This is the first invocation for this function; add it to the history
        # and set the current timestamp as the first bin
        if len(fun.history) == 0:
            fun.history.loc[0] = {"ds": timestamp.floor('1s'), "y": 0, "available_containers": 0}
            fun.cur_time = timestamp.floor('1s')
            fun.last_predictor_update = timestamp.floor('1s')

        # Simulate strictly per second
        fun.cur_time = fun.cur_time.floor('1s')

        # Get the last time bin in this function's history
        last = fun.history.loc[len(fun.history) - 1]

        # Determine the number of bins needed to "catch up" to the latest 
        # invocation
        ints = math.floor((timestamp.timestamp() - last['ds'].timestamp()) / self._granularity) + 1

        # For each bin (including the current one) until the latest timestamp
        for i in range(ints):
            # This not the current bin, so we need to add a bin to "catch up"
            if i != 0:
                fun.cur_time += s(self._granularity)
                fun.history.loc[len(fun.history)] = {"ds": fun.cur_time, "y": 0, "available_containers": 0}

            # Based on warming period and min containers, prune (or create)
            self._adjust_containers(f, container_logs)
            # If needed, fit a model to predict the next interval (and adjust the
            # warming period and min containers knobs as necessary)
            if with_model: self._adjust_model(f)

            # Log the number of warmed containers
            conts = fun.history.loc[len(fun.history) - 1, "available_containers"]
            fun.history.loc[len(fun.history) - 1, "available_containers"] = max(conts, len(fun.containers))

        # For posterity, set the current timestamp to the non rounded version
        fun.cur_time = timestamp

        # Add an invocation to our logs
        fun.history.loc[len(fun.history) - 1, "y"] += 1

        # Adjust and fit model if necessary for the new timestamp
        self._adjust_containers(f, container_logs)
        if with_model: self._adjust_model(f)
    
    def _adjust_model(self, f):
        fun = self._functions[f]

        # Check if the model is due for an update
        if fun.cur_time - fun.last_predictor_update < s(self._predictor_interval):
            return
        fun.last_predictor_update = fun.cur_time

        # Create prophet model
        prophet = Prophet(changepoint_prior_scale=1)
        prophet.add_seasonality(name='seconds_level', period=60, fourier_order=20)
        prophet.add_seasonality(name="minute_level", period=3600, fourier_order=5)

        # Fit it to the past history interval
        hist = fun.history.iloc[max(0, len(fun.history) - self._predictor_history_interval):]
        prophet.fit(hist)

        # Get the future DataFrame to predict
        last = fun.history.loc[len(fun.history) - 1]
        future = pd.DataFrame({'ds': [last['ds'] + timedelta(seconds=i * self._granularity) 
                                      for i in range(1, self._predictor_interval // self._granularity + 1)]})

        # Run prediction model and filter by future timestamps
        forecast = prophet.predict(future)
        future_timestamps = set(future['ds'])
        forecast = forecast[forecast['ds'].isin(future_timestamps)][['ds', 'yhat']]

        # If failed, revert to default strategy
        if forecast is None or forecast.empty:
            fun.warming_period = self._default_warming_period
            fun.min_num_containers = self._default_min_num_containers
            return

        # Get the number of forecasted invocations (nonzero)
        invocation_counts = {row['ds']: max(0, row['yhat']) for _, row in forecast.iterrows()}

        # Set the number of warmed to the maximum forecasted number of 
        # invocations
        max_concurrent = max(invocation_counts.values(), default=0)
        fun.min_num_containers = int(round(max_concurrent))

        # Get the number of active bins (interval where invocation > 0) and 
        # set that as the warming period
        active_seconds = [ds for ds, count in invocation_counts.items() if count > 0]
        fun.warming_period = (int((max(active_seconds) - min(active_seconds)).total_seconds()) + 1 if active_seconds else 0) * self._granularity

        # Keep log of these predictions
        fun.predictions.append(forecast)

    def _adjust_containers(self, f, container_logs: pd.DataFrame):
        fun = self._functions[f]

        minw = fun.min_num_containers
        cw = 0

        for cont in fun.containers:
            # If expired and not in use, remove it
            if cont.expiration <= fun.cur_time and cont.in_use_until <= fun.cur_time:
                container_logs.loc[len(container_logs)] = {
                    "creation": cont.creation,
                    "expiration": cont.expiration,
                    "active_time": cont.active_time
                }

                fun.containers.remove(cont)

            # Count the number of permanently warmed containers
            if cont.expiration == MAX_TIMESTAMP:
                # If reached the max, demote container to regular warming 
                # period
                if cw >= minw:
                    cont.expiration = fun.cur_time  # + s(fun.warming_period)

                cw += 1

        # If we haven't reached the number of permanent containers, create some
        if cw < minw:
            to_add = minw - cw

            # Try to promote current containers
            for cont in fun.containers:
                if to_add == 0: break
                cont.expiration = MAX_TIMESTAMP
                to_add -= 1

            # For the rest, create them
            if to_add > 0:
                for _ in range(to_add):
                    fun.containers.append(
                        Container(fun.cur_time,
                                  MAX_TIMESTAMP, pd.Timestamp(0),
                                  pd.Timedelta(seconds=0))
                    )
                
    def _get_stats(self, is_warm_start: bool) -> tuple[float, float, float]:
        dist = self._warm_start_dist if is_warm_start else self._cold_start_dist

        # We're assuming a normal distribution for these events
        return (np.random.normal(dist.acquire[0], dist.acquire[1]),
                np.random.normal(dist.execution[0], dist.execution[1]),
                np.random.normal(dist.release[0], dist.release[1]))
    
    def _get_available_container(self, f: str, duration: pd.Timedelta) -> tuple[bool, tuple[float, float, float]]:
        fun = self._functions[f]

        a, e, r = self._get_stats(False)
        is_warm_start = False

        # Find a container that isn't expired and isn't in use
        for cont in fun.containers:
            if cont.expiration > fun.cur_time and cont.in_use_until < fun.cur_time:
                is_warm_start = True
                a, e, r = self._get_stats(True)
                cont.expiration = fun.cur_time + s(fun.warming_period)
                cont.in_use_until = fun.cur_time + s(a + e + r) + duration
                cont.active_time = cont.active_time + s(a + e + r) + duration
                break

        # Couldn't find a container, had to create one
        if not is_warm_start:
            fun.containers.append(
                Container(fun.cur_time,
                          fun.cur_time + s(fun.warming_period), fun.cur_time + s(a + e + r) + duration,
                          s(a + e + r) + duration)
            )

        return is_warm_start, (a, e, r)

    def _cleanup(self, container_logs: pd.DataFrame) -> None:
        """
        Add final entries to container logs (those that haven't expired by the end of the simulation)
        """
        for function in self._functions.values():
            for container in function.containers:
                container_logs.loc[len(container_logs)] = {
                    "creation": container.creation,
                    "expiration": function.cur_time,  # expire everything upon destruction of the universe
                    "active_time": container.active_time
                }
