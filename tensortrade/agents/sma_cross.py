# from tensortrade.agents import Agent, ReplayMemory
from ray.rllib.algorithms.algorithm import Algorithm
import pandas as pd
import numpy as np
# from quantlib.utils.parameters import get_param
from quantutils.parameters import get_param

from tensortrade.feed.features import SMA, BBANDS
import tensortrade.env.config as ec
import random


# TODO: make abstract Agent.. or TradeAgent
# Agent(Algorithm):

    # @abstractmethod
    # def get_action(self, state: np.ndarray, **kwargs) -> int:
    #     """Get an action for a specific state in the environment."""
    #     raise NotImplementedError()

class SMACross(Algorithm):


    def __init__(self, config):
        self.config = config
        self.n_actions = get_param(config, 'n_actions')['value']
        self.observation_shape = get_param(config, 'observation_space_shape')['value']
        self.last_action = 1
        self.features = self.get_features()
        self.timeframes = get_param(self.config, 'timeframes')['value']
        self.last_bar_open_time = 0

    def restore(self, path: str, **kwargs):
        self.policy_network = tf.keras.models.load_model(path)
        self.target_network = tf.keras.models.clone_model(self.policy_network)
        self.target_network.trainable = False

    def get_features(self):
        param = get_param(self.config, 'fast_ma')['value']
        from collections import OrderedDict
        sma = SMA(OrderedDict({'timeperiod': param}))
        sma_params = sma.get_function_params()
        timeframes = get_param(self.config, 'timeframes')['value']
        upper_timeframe = timeframes[-1]
        lower_timeframe = timeframes[0]

        self.features={
                       f'{lower_timeframe}':[SMA({'timeperiod': get_param(self.config, 'fast_ma')['value']}),
                                               SMA({'timeperiod': get_param(self.config, 'slow_ma')['value']})]
                      }

        return self.features


    def save(self, path: str, **kwargs):
        episode: int = kwargs.get('episode', None)

        if episode:
            filename = "policy_network__" + self.id[:7] + "__" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".hdf5"
        else:
            filename = "policy_network__" + self.id[:7] + "__" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".hdf5"

        self.policy_network.save(path + filename)

    def train(self):
        pass

    def get_action(self, state: np.ndarray, **kwargs) -> int:
        header = kwargs['header']
        timeframes = get_param(self.config, 'timeframes')['value']
        features = {}
        is_new_bar = bool(state[header.index("0_new_bar")])
        fast0 = state[header.index(f"0_{timeframes[0]}_SMA_real_{get_param(self.config, 'fast_ma')['value']}")]
        slow0 = state[header.index(f"0_{timeframes[0]}_SMA_real_{get_param(self.config, 'slow_ma')['value']}")]
        if is_new_bar:
            if fast0 > slow0:
                self.last_action = 0
                return 0
            if fast0 <= slow0:
                self.last_action = 1
                return 1
        else:
            return self.last_action

    # def get_state(self, X) -> np.ndarray:
    #     return np.zeros(1)


    def get_state(self, config):
        state=[]
        header=[]
        for i in range(get_param(self.config, 'window_size')['value']):
            for f in self.features[self.timeframes[0]]:
                # TODO: send all quotes not only 'close'
                value = f.calc(config['quotes'][self.timeframes[0]]['close'])
                if type(value) == np.ndarray or type(value) == self.pd.Series:
                    state.append(value[-1])
                else:
                    for v in value:
                        state.append(v[-1])

                header.extend([f'{i}_{self.timeframes[0]}_{h}' for h in f.get_header()])

        new_bar = False
        if  self.last_bar_open_time == 0:
            self.last_bar_open_time = config['quotes'][self.timeframes[0]].index[-1]
        elif self.last_bar_open_time < config['quotes'][self.timeframes[0]].index[-1]:
            self.last_bar_open_time = config['quotes'][self.timeframes[0]].index[-1]
            new_bar = True

        # FIXME: remove next line, its only for testing
        new_bar = True

        state.append(new_bar)
        header.append('0_new_bar')
        return (state, header)
        # for t in self.features:
        for f in self.features[self.timeframes[0]]:
            obs.extend(f.calc(config['quotes'][self.timeframes[0]]['close']))
            header.extend(f.get_header())
        return (obs, header)

class SMACross_TwoScreens(Algorithm):

    def __init__(self, config):
        self.config = config
        self.n_actions = get_param(config, 'n_actions')['value']
        self.observation_shape = get_param(config, 'observation_space_shape')['value']


    def restore(self, path: str, **kwargs):
        self.policy_network = tf.keras.models.load_model(path)
        self.target_network = tf.keras.models.clone_model(self.policy_network)
        self.target_network.trainable = False

    def get_features(self):
        print(self.config)
        param = get_param(self.config, 'upper_screen_fast_ma')['value']

        from collections import OrderedDict
        sma = SMA(OrderedDict({'timeperiod': param}))
        sma_params = sma.get_function_params()
        timeframes = get_param(self.config, 'timeframes')['value']
        upper_timeframe = timeframes[-1]
        lower_timeframe = timeframes[0]



        self.features={f'{upper_timeframe}':[BBANDS({'timeperiod': 15}), SMA({'timeperiod': get_param(self.config, 'upper_screen_fast_ma')['value']}),
                                               SMA({'timeperiod': get_param(self.config, 'upper_screen_slow_ma')['value']})],

                       f'{lower_timeframe}':[SMA({'timeperiod': get_param(self.config, 'lower_screen_fast_ma')['value']}),
                                               SMA({'timeperiod': get_param(self.config, 'lower_screen_slow_ma')['value']})]}


        return self.features

    def save(self, path: str, **kwargs):
        episode: int = kwargs.get('episode', None)

        if episode:
            filename = "policy_network__" + self.id[:7] + "__" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".hdf5"
        else:
            filename = "policy_network__" + self.id[:7] + "__" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".hdf5"

        self.policy_network.save(path + filename)

    def train(self):
        pass

    def get_action(self, state: np.ndarray, **kwargs) -> int:
        header = kwargs['header']
        timeframes = get_param(self.config, 'timeframes')['value']
        features = {}
        return random.randint(0,1)

    def get_state(self, X) -> np.ndarray:
        return np.zeros(1)


if __name__ == "__main__":
    print('hoo')
