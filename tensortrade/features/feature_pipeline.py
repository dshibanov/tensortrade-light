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

from typing import Union
from sklearn.utils import check_array

from .transformer import TransformableList

DTypeString = Union[type, str]


class FeaturePipeline(object):
    """An pipeline for transforming observation data frames into features for learning."""

    def __init__(self, *args, **kwargs):
        """
        Arguments:
            pipeline (optional): An `sklearn.Pipeline` instance of feature transformations.
            dtype: The `dtype` elements in the pipeline should be cast to.
        """
        self._pipeline: 'Pipeline' = kwargs.get('pipeline', None)
        self._dtype: DTypeString = kwargs.get('dtype', np.float16)

        if self._pipeline is None:
            self._pipeline = args

        if self._pipeline is None:
            raise ValueError(
                'Feature pipeline requires a list of transformers or `sklearn.Pipeline`.')

    @property
    def pipeline(self) -> 'Pipeline':
        """An `sklearn.Pipeline` instance of feature transformations."""
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline: 'Pipeline'):
        self.pipeline = pipeline

    @property
    def dtype(self) -> DTypeString:
        """The `dtype` elements in the pipeline should be input and output as."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: DTypeString):
        self._dtype = dtype

    def _transform(self, observations: pd.DataFrame) -> TransformableList:
        for transformer in self._pipeline:
            observations = transformer.transform(observations)

        return observations

    def fit_transform(self, observation: pd.DataFrame) -> np.ndarray:
        """Apply the pipeline of feature transformations to an observation frame.

        Arguments:
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

        if isinstance(self._pipeline, 'Pipeline'):
            features = self._pipeline.fit_transform(features)
        else:
            features = self._transform(features)

        if isinstance(features, pd.DataFrame):
            return features.values

        return features
