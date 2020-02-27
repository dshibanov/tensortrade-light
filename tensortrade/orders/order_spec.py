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


from typing import Callable

from tensortrade.base import Identifiable
from tensortrade.orders import Order, TradeSide, TradeType


class OrderSpec(Identifiable):

    def __init__(self,
                 side: TradeSide,
                 trade_type: TradeType,
                 pair: 'TradingPair',
                 criteria: Callable[['Order', 'Exchange'], bool] = None):
        self.side = side
        self.type = trade_type
        self.pair = pair
        self.criteria = criteria

    def create_order(self, order: 'Order', exchange: 'Exchange') -> 'Order':
        wallet_instrument = self.side.instrument(self.pair)

        wallet = order.portfolio.get_wallet(exchange.id, instrument=wallet_instrument)
        quantity = wallet.locked.get(order.path_id, 0)

        return Order(step=exchange.clock.step,
                     exchange_name=order.exchange_name,
                     side=self.side,
                     trade_type=self.type,
                     pair=self.pair,
                     quantity=quantity,
                     portfolio=order.portfolio,
                     price=order.price,
                     criteria=self.criteria,
                     start=order.start,
                     end=order.end,
                     path_id=order.path_id)

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "pair": self.pair,
            "criteria": self.criteria
        }

    def __str__(self):
        data = ['{}={}'.format(k, v) for k, v in self.to_dict().items()]
        return '<{}: {}>'.format(self.__class__.__name__, ', '.join(data))

    def __repr__(self):
        return str(self)
