
from tensortrade.env.generic import Informer, TradingEnv


class TensorTradeInformer(Informer):

    def __init__(self) -> None:
        super().__init__()

    def info(self, env: 'TradingEnv') -> dict:
        return {
            'step': self.clock.step,
            'net_worth': env.action_scheme.portfolio.net_worth
        }


class MultySymbolEnvInformer(Informer):

    def __init__(self) -> None:
        super().__init__()

    def info(self, env: 'TradingEnv') -> dict:
        return {
            'step': self.clock.step,
            'net_worth': env.action_scheme.portfolio.net_worth,
            # FIXME:
            # we can find out & return current_symbol_code somehow
            # 'symbol_code': env.current_symbol_code,
            'symbol_code': env.observer.symbol_code,
            'end_of_episode': env.observer.end_of_episode
        }
