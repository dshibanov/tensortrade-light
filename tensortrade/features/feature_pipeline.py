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
import numpy as np

from abc import ABCMeta, abstractmethod
from sklearn import Pipeline
from sklearn.utils import check_array


class FeaturePipeline(object):
    """An pipeline for transforming observation data frames into features for learning."""

    def __init__(self, pipeline: Pipeline, dtype: type = np.float16):
        """
        Args:
            pipeline: An `sklearn.Pipeline` instance of feature transformations.
            dtype: The `dtype` elements in the pipeline should be cast to.
        """
        self._pipeline = pipeline
        self._dtype = dtype

    @property
    def pipeline(self):
        """An `sklearn.Pipeline` instance of feature transformations."""
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline: Pipeline):
        self.pipeline = pipeline

    @property
    def dtype(self):
        """The `dtype` elements in the pipeline should be cast to"""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: Pipeline):
        self._dtype = dtype

    def fit_transform(self, observation: pd.DataFrame) -> np.ndarray:
        """Fit and apply the pipeline of feature transformations to an observation frame.

        Args:
            observation: A `pandas.DataFrame` corresponding to an observation within a `TradingEnvironment`.

        Returns:
            A `numpy.ndarray` of features.

        Raises:
            ValueError: In the case that an invalid observation frame has been input.
        """
        try:
            features = check_array(observation, dtype=self._dtype)
        except ValueError as e:
            raise ValueError(f'Invalid observation frame passed to feature pipeline: {e}')

        features = self._pipeline.fit_transform(features)

        if isinstance(features, pd.DataFrame):
            return features.values

        return features
