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
# limitations under the License

from typing import Dict, Tuple

from tensortrade.base import Identifiable
from tensortrade.base.exceptions import InsufficientFunds
from tensortrade.instruments import Quantity

from .ledger import Ledger, Transaction


class Wallet(Identifiable):
    """A wallet stores the balance of a specific instrument on a specific exchange."""

    ledger = Ledger()

    def __init__(self, exchange: 'Exchange', quantity: 'Quantity'):
        self._exchange = exchange
        self._initial_size = quantity.size
        self._instrument = quantity.instrument
        self._balance = quantity
        self._locked = {}

    @classmethod
    def from_tuple(cls, wallet_tuple: Tuple['Exchange', 'Instrument', float]):
        exchange, instrument, balance = wallet_tuple
        return cls(exchange, Quantity(instrument, balance))

    @property
    def exchange(self) -> 'Exchange':
        return self._exchange

    @exchange.setter
    def exchange(self, exchange: 'Exchange'):
        raise ValueError("You cannot change a Wallet's Exchange after initialization.")

    @property
    def instrument(self) -> 'Instrument':
        return self._instrument

    @instrument.setter
    def instrument(self, instrument: 'Exchange'):
        raise ValueError("You cannot change a Wallet's Instrument after initialization.")

    @property
    def balance(self) -> 'Quantity':
        """The total balance of the wallet available for use."""
        return self._balance

    @balance.setter
    def balance(self, balance: 'Quantity'):
        self._balance = balance

    @property
    def locked_balance(self) -> 'Quantity':
        """The total balance of the wallet locked in orders."""
        locked_balance = Quantity(self.instrument, 0)

        for quantity in self.locked.values():
            locked_balance += quantity.size

        return locked_balance

    @property
    def total_balance(self) -> 'Quantity':
        """The total balance of the wallet, both available for use and locked in orders."""
        total_balance = self._balance

        for quantity in self.locked.values():
            total_balance += quantity.size

        return total_balance

    @property
    def locked(self) -> Dict[str, 'Quantity']:
        return self._locked

    def deallocate(self, path_id: str, reason: str = "DEALLOCATE"):
        if path_id in self.locked.keys():
            quantity = self.locked.pop(path_id, None)

            if quantity is not None:
                self += (quantity.size * self.instrument).reason(reason)

    def __iadd__(self, quantity: 'Quantity') -> 'Wallet':
        if quantity.is_locked:
            if quantity.path_id not in self.locked.keys():
                self._locked[quantity.path_id] = quantity
            else:
                self._locked[quantity.path_id] += quantity
        else:
            self._balance += quantity

        self.ledger.commit(Transaction(
            self.exchange.clock.step,
            self.exchange.name,
            self.instrument,
            "DEPOSIT",
            quantity.path_id,
            quantity.memo,
            quantity,
            self.balance,
            self.locked_balance
        ))
        return self

    def __isub__(self, quantity: 'Quantity') -> 'Wallet':
        if quantity.is_locked and self.locked[quantity.path_id]:
            if quantity > self.locked[quantity.path_id]:
                raise InsufficientFunds(self.locked[quantity.path_id], quantity)
            self._locked[quantity.path_id] -= quantity
        elif not quantity.is_locked:
            if quantity > self._balance:
                raise InsufficientFunds(self.balance, quantity)
            self._balance -= quantity

        self.ledger.commit(Transaction(
            self.exchange.clock.step,
            self.exchange.name,
            self.instrument,
            "WITHDRAW",
            quantity.path_id,
            quantity.memo,
            quantity,
            self.balance,
            self.locked_balance
        ))
        return self

    def reset(self):
        self._balance = Quantity(self._instrument, self._initial_size)
        self._locked = {}

    def __str__(self):
        return '<Wallet: balance={}, locked={}>'.format(self.balance, self.locked_balance)

    def __repr__(self):
        return str(self)
