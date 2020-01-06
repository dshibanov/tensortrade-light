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


import pandas as pd

from tensortrade.exchanges import Exchange


class GANExchange(Exchange):
    """A simulated exchange, in which the price history is based off a generative adversarial network
    model with supplied parameters.

    If the `training_data` parameter is not supplied upon initialization, it must be set before
    the exchange can be used within a trading environment.
    """

    def __init__(self, training_data: pd.DataFrame = None, **kwargs):
        super().__init__(**kwargs)

        self._training_data = self.default('training_data', training_data)

        raise NotImplementedError()

    def reset(self):
        super().reset()
