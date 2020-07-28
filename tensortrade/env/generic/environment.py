# Copyright 2020 The TensorTrade Authors.
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

import gym
import uuid
import logging

from tensortrade.core import TimeIndexed, Clock
from tensortrade.env.generic import (
    ActionScheme,
    RewardScheme,
    Observer,
    Stopper,
    Monitor,
    Renderer
)


class TradingEnv(gym.Env, TimeIndexed):
    """A trading environment made for use with Gym-compatible reinforcement learning algorithms."""

    # agent_id: str = None
    # episode_id: str = None
    #  * Used in ParallelDQN

    def __init__(self,
                 action_scheme: ActionScheme,
                 reward_scheme: RewardScheme,
                 observer: Observer,
                 stopper: Stopper,
                 monitor: Monitor,
                 renderer: Renderer,
                 **kwargs):
        super().__init__()
        self.clock = Clock()

        self.action_scheme = action_scheme
        self.reward_scheme = reward_scheme
        self.observer = observer
        self.stopper = stopper
        self.monitor = monitor
        self.renderer = renderer

        for c in self.components.values():
            c.clock = self.clock

        self.action_space = action_scheme.action_space
        self.observation_space = observer.observation_space

        self._max_episodes = None
        self._max_steps = None
        self._episode = 0

        self._enable_logger = kwargs.get('enable_logger', False)
        if self._enable_logger:
            self.logger = logging.getLogger(kwargs.get('logger_name', __name__))
            self.logger.setLevel(kwargs.get('log_level', logging.DEBUG))

    """
    @property
    def episode(self):
        return self._episode

    @property
    def max_episodes(self):
        return self._max_episodes

    @max_episodes.setter
    def max_episodes(self, max_episodes):
        self._max_episodes = max_episodes

    @property
    def max_steps(self):
        return self._max_steps

    @max_steps.setter
    def max_steps(self, max_steps):
        self._max_steps = max_steps
    """
    # * Used in the rendering systems

    @property
    def components(self):
        return {
            "action_scheme": self.action_scheme,
            "reward_scheme": self.reward_scheme,
            "observer": self.observer,
            "stopper": self.stopper,
            "monitor": self.monitor,
            "renderer": self.renderer
        }

    def step(self, action):
        self.action_scheme.perform(self, action)

        obs = self.observer.observe(self)
        reward = self.reward_scheme.reward(self)
        done = self.stopper.stop(self)
        info = self.monitor.info(self)

        self.clock.increment()

        return obs, reward, done, info

    def reset(self):
        # self.episode_id = str(uuid.uuid4())
        # self._episode += 1
        self.clock.reset()

        for c in self.components.values():
            c.reset()

        obs = self.observer.observe(self)

        self.clock.increment()

        return obs

    def render(self, **kwargs):
        self.renderer.render(self, **kwargs)

    def save(self):
        self.renderer.save()

    def close(self):
        self.renderer.close()
