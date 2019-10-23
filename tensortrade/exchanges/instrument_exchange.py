# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import sys
import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Dict, Union, List, Generator
from gym.spaces import Space

from tensortrade.trades import Trade
from tensortrade.features import FeaturePipeline

TypeString = Union[type, str]

class InstrumentExchange(object, metaclass=ABCMeta):
    """An abstract instrument exchange for use within a trading environment."""

    def __init__(self, base_instrument: str = 'USD', dtype: TypeString = np.float16, feature_pipeline: FeaturePipeline = None):
        """
        Arguments:
            base_instrument: The exchange symbol of the instrument to store/measure value in.
            dtype: A type or str corresponding to the dtype of the `observation_space`.
            feature_pipeline: A pipeline of feature transformations for transforming observations.
        """
        self._base_instrument = base_instrument
        self._dtype = dtype
        self._feature_pipeline = feature_pipeline

    @property
    def base_instrument(self) -> str:
        """The exchange symbol of the instrument to store/measure value in."""
        return self._base_instrument

    @base_instrument.setter
    def base_instrument(self, base_instrument: str):
        self._base_instrument = base_instrument

    @property
    def dtype(self) -> TypeString:
        """A type or str corresponding to the dtype of the `observation_space`."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: TypeString):
        self._dtype = dtype

    @property
    def feature_pipeline(self) -> FeaturePipeline:
        """A pipeline of feature transformations for transforming observations."""
        return self._feature_pipeline

    @feature_pipeline.setter
    def feature_pipeline(self, feature_pipeline: FeaturePipeline):
        self._feature_pipeline = feature_pipeline

    @property
    def base_precision(self) -> float:
        """The floating point precision of the base instrument."""
        return self._base_precision

    @base_precision.setter
    def base_precision(self, base_precision: float):
        self._base_precision = base_precision

    @property
    def instrument_precision(self) -> float:
        """The floating point precision of the instrument to be traded."""
        return self._instrument_precision

    @instrument_precision.setter
    def instrument_precision(self, instrument_precision: float):
        self._instrument_precision = instrument_precision

    @property
    @abstractmethod
    def initial_balance(self) -> float:
        """The initial balance of the base symbol on the exchange."""
        raise NotImplementedError

    @property
    @abstractmethod
    def balance(self) -> float:
        """The current balance of the base symbol on the exchange."""
        raise NotImplementedError

    @property
    @abstractmethod
    def portfolio(self) -> Dict[str, float]:
        """The current balance of each symbol on the exchange (non-positive balances excluded)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def trades(self) -> List[Trade]:
        """A list of trades made on the exchange since the last reset."""
        raise NotImplementedError

    @property
    @abstractmethod
    def performance(self) -> pd.DataFrame:
        """The performance of the active account on the exchange since the last reset."""
        raise NotImplementedError

    @property
    @abstractmethod
    def generated_space(self) -> Space:
        """The initial shape of the observations generated by the exchange, before feature transformations."""
        raise NotImplementedError

    @property
    @abstractmethod
    def generated_columns(self) -> List[str]:
        """The list of column names of the observation data frame generated by the exchange, before feature transformations."""
        raise NotImplementedError

    @property
    def observation_space(self) -> Space:
        """The final shape of the observations generated by the exchange, after feature transformations."""
        if self._feature_pipeline is not None:
            return self._feature_pipeline.transform_space(self.generated_space, self.generated_columns)

        return self.generated_space

    @property
    def net_worth(self) -> float:
        """Calculate the net worth of the active account on the exchange.

        Returns
            The total portfolio value of the active account on the exchange.
        """
        net_worth = self.balance
        portfolio = self.portfolio

        if not portfolio:
            return net_worth

        for symbol, amount in portfolio.items():
            current_price = self.current_price(symbol=symbol)
            net_worth += current_price * amount

        return net_worth

    @property
    def profit_loss_percent(self) -> float:
        """Calculate the percentage change in net worth since the last reset.

        Returns:
            The percentage change in net worth since the last reset.
        """
        return float(self.net_worth / self.initial_balance) * 100

    @property
    @abstractmethod
    def has_next_observation(self) -> bool:
        """If `False`, the exchange's data source has run out of observations.

        Resetting the exchange may be necessary to continue generating observations.

        Returns:
            Whether or not the specified instrument has a next observation.
        """
        raise NotImplementedError

    @abstractmethod
    def _create_observation_generator(self) -> Generator[pd.DataFrame, None, None]:
        raise NotImplementedError

    def next_observation(self) -> np.ndarray:
        """Generate the next observation from the exchange.

        Returns:
            The next multi-dimensional list of observations.
        """
        observation = next(self._observation_generator, None)

        if isinstance(observation, pd.DataFrame):
            observation = observation.fillna(0, axis=1)

            return observation.values

        return observation


    def instrument_balance(self, symbol: str) -> float:
        """The current balance of the specified symbol on the exchange, denoted in the base instrument.

        Arguments:
            symbol: The symbol to retrieve the balance of.

        Returns:
            The balance of the specified exchange symbol, denoted in the base instrument.
        """
        portfolio = self.portfolio

        if symbol in portfolio.keys():
            return portfolio[symbol]

        return 0

    @abstractmethod
    def current_price(self, symbol: str) -> float:
        """The current price of an instrument on the exchange, denoted in the base instrument.

        Arguments:
            symbol: The exchange symbol of the instrument to get the price for.

        Returns:
            The current price of the specified instrument, denoted in the base instrument.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_trade(self, trade: Trade) -> Trade:
        """Execute a trade on the exchange, accounting for slippage.

        Arguments:
            trade: The trade to execute.

        Returns:
            The filled trade.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset the feature pipeline, initial balance, trades, performance, and any other temporary stateful data."""
        if self._feature_pipeline is not None:
            self.feature_pipeline.reset()

        self._observation_generator = self._create_observation_generator()
