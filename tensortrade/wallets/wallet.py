
from typing import Dict, Tuple, Union

from tensortrade.base import Identifiable
from tensortrade.base.exceptions import InsufficientFundsForAllocation
from tensortrade.instruments import Quantity


class Wallet(Identifiable):
    """A wallet stores the balance of a specific instrument on a specific exchange."""

    def __init__(self, exchange: 'Exchange', quantity: 'Quantity'):
        self._exchange = exchange
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
            locked_balance += quantity

        return locked_balance

    @property
    def total_balance(self) -> 'Quantity':
        """The total balance of the wallet, both available for use and locked in orders."""
        total_balance = self._balance

        for quantity in self.locked.values():
            total_balance += quantity

        return total_balance

    @property
    def locked(self) -> Dict[str, 'Quantity']:
        return self._locked

    def allocate(self, amount: Union['Quantity', float, int], path_id: str = None) -> 'Quantity':

        if amount > self._balance:
            raise InsufficientFundsForAllocation(self.balance, amount.size)

        if isinstance(amount, Quantity) and amount.path_id and path_id:

            if amount.path_id != path_id:
                raise Exception("Specified path id of quantity and "
                                "allocation parameter do not match: {} {}".format(amount.path_id, path_id))
            else:
                self -= amount.size * amount.instrument

        elif isinstance(amount, Quantity) and not amount.path_id and not path_id:
            self -= amount
            amount = amount.lock_for(path_id)

        elif isinstance(amount, float) or isinstance(amount, int) and path_id:
            self -= amount
            amount = Quantity(self.instrument, amount, path_id=path_id)

        elif isinstance(amount, float) or isinstance(amount, int) and not path_id:
            raise Exception("Can't allocate without valid path id: {}".format(path_id))

        self += amount
        return amount

    def deallocate(self, path_id: str):
        if path_id in self.locked.keys():
            quantity = self.locked.pop(path_id, None)

            if quantity is not None:
                self += quantity.size * self.instrument

    def __iadd__(self, quantity: 'Quantity') -> 'Wallet':
        if quantity.is_locked:
            if quantity.path_id not in self.locked.keys():
                self._locked[quantity.path_id] = quantity
            else:
                self._locked[quantity.path_id] += quantity
        else:
            self._balance += quantity

        return self

    def __isub__(self, quantity: 'Quantity') -> 'Wallet':
        if quantity.is_locked and self.locked[quantity.path_id]:
            self._locked[quantity.path_id] -= quantity
        elif not quantity.is_locked:
            self._balance -= quantity

        return self

    def __str__(self):
        return '<Wallet: balance={}, locked={}>'.format(self.balance, self.locked_balance)

    def __repr__(self):
        return str(self)
