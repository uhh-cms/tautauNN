# coding: utf-8

from __future__ import annotations

import enum
from typing import Callable

import math
import numpy as np

import tensorflow as tf


class DatasetKind(enum.StrEnum):
    train = enum.auto()
    valid = enum.auto()

    @classmethod
    def from_str(cls, s: str) -> DatasetKind:
        return cls[s]


class MultiDataset(object):

    def __init__(
        self,
        data: tuple[tuple[np.array] | np.array, float],
        batch_size: int = 128,
        kind: DatasetKind | str = "train",
        yield_valid_rest: bool = True,
        seed: int | None = None,
        transform_data: Callable[[MultiDataset, tuple[tf.Tensor, ...]], tuple[tf.Tensor, ...]] | None = None,
    ):
        super().__init__()

        # attributes
        self.batch_size = batch_size
        self.kind = DatasetKind.from_str(kind) if isinstance(kind, str) else kind
        self.yield_valid_rest = yield_valid_rest
        self.seed = seed
        self.transform_data = transform_data
        self.tuple_length = 0
        self.batches_seen = 0

        # create datasets, store counts and relative weights
        self.datasets = []
        self.counts = []
        self.batch_weights = []
        for arrays, batch_weight in data:
            if not isinstance(arrays, tuple):
                arrays = (arrays,)
            self.tuple_length = len(arrays)
            self.datasets.append(arrays)
            self.counts.append(len(arrays[0]))
            self.batch_weights.append(batch_weight)

        if self.tuple_length < 3:
            raise Exception("per dataset, at least three arrays must be given: inputs, targets, weights")

        # when used for validation, set the batch size to the total number of events
        if self.kind == "valid":
            self.batch_size = sum(self.counts)

        # transform batch weights to relative probabilities
        sum_batch_weights = sum(self.batch_weights)
        self.probs = [w / sum_batch_weights for w in self.batch_weights]

        # the number of batches that constitute one iteration cycle through all events
        # (floating point number, to be ceiled or floored depending on whether rest is used)
        self._batches_per_cycle: float = len(self) / self.batch_size

    def __len__(self):
        return sum(self.counts)

    @property
    def batches_per_cycle(self) -> int:
        round_func = math.ceil if self.kind == "valid" and self.yield_valid_rest else math.floor
        return int(round_func(self._batches_per_cycle))

    @property
    def n_datasets(self) -> int:
        return len(self.datasets)

    def __iter__(self):
        return self.iter_train() if self.kind == "train" else self.iter_valid()

    def iter_train(self):
        # preparations
        transform_data = self.transform_data if callable(self.transform_data) else (lambda self, x: x)

        # prepare indices for random sampling
        indices = [np.array([], dtype=np.int32) for _ in range(self.n_datasets)]
        offsets = [0] * self.n_datasets

        # start iterating
        while True:
            # determine batch sizes per dataset for this chunk
            batch_sizes = np.random.multinomial(self.batch_size, self.probs)

            # fill chunks per dataset that eventually form a batch
            chunks = []
            for i, (arrays, _indices, batch_size, offset) in enumerate(zip(self.datasets, indices, batch_sizes, offsets)):
                # update indices and offset
                if len(_indices) - offset < batch_size:
                    new_indices = np.arange(len(arrays[0]), dtype=np.int32)
                    np.random.shuffle(new_indices)
                    _indices = indices[i] = np.concatenate([_indices[offset:], new_indices], axis=0)
                    offset = 0

                # fill the chunk and adjust the offset
                chunks.append([a[_indices[offset:offset + batch_size]] for a in arrays])
                offsets[i] = offset + batch_size

            # yield
            data = tuple(
                np.concatenate([chunk[i] for chunk in chunks], axis=0)
                for i in range(self.tuple_length)
            )
            data = transform_data(self, *data)
            chunks.clear()

            yield tuple(map(tf.convert_to_tensor, data))
            self.batches_seen += 1

    def iter_valid(self):
        # preparations
        transform_data = self.transform_data if callable(self.transform_data) else (lambda self, x: x)

        # start iterating
        dataset_index = 0
        offset = 0
        chunks = []
        n_total = 0
        while True:
            # iterate until batch size is reached
            while n_total < self.batch_size:
                # optionally switch to next dataset and fill indices
                if offset >= self.counts[dataset_index]:
                    dataset_index = (dataset_index + 1) % self.n_datasets
                    offset = 0
                # fill the chunk
                chunks.append([a[offset:offset + self.batch_size - n_total] for a in self.datasets[dataset_index]])
                offset += len(chunks[-1][0])
                n_total += len(chunks[-1][0])
                # manually stop when the last dataset is exhausted and the rest is to be yielded on its own
                # (otherwise, the above will cycle back to the first dataset and fill the chunk)
                if n_total < self.batch_size and dataset_index == self.n_datasets - 1 and self.yield_valid_rest:
                    break

            # concatenate chunks
            data = tuple(
                np.concatenate([chunk[i] for chunk in chunks], axis=0)
                for i in range(self.tuple_length)
            )
            chunks.clear()
            n_total = 0

            # yield
            yield transform_data(self, *data)
            self.batches_seen += 1

    def create_keras_generator(self, input_names: list[str] | None = None, target_names: list[str] | None = None):
        # this assumes that at least three arrays are yielded by the __iter__ method: inputs, targets, weights
        # when input_names are given, the inputs array is split into a dictionary with the given names
        # when there is more than one input array, input_names are mandatory
        if self.tuple_length > 3 and not input_names:
            raise ValueError("input_names must be given when there is more than one output to be yielded")

        # default names
        if not input_names:
            input_names = []
        if not target_names:
            target_names = []

        # start generating
        n_input_names = len(input_names)
        n_target_names = len(target_names)
        for arrays in self:
            n_arrays = len(arrays)
            if n_arrays == 1:
                assert n_input_names in [0, 1]
                yield {input_names[0]: arrays[0]} if input_names else arrays[0]
            elif n_arrays == 2:
                assert n_input_names in [0, 1]
                yield ({input_names[0]: arrays[0]} if input_names else arrays[0]), arrays[1]
            elif n_arrays == 3:
                assert n_input_names in [0, 1]
                yield ({input_names[0]: arrays[0]} if input_names else arrays[0]), *arrays[1:]
            else:
                assert n_arrays == max(n_input_names, 1) + max(n_target_names, 1) + 1
                x, arrays = (
                    (dict(zip(input_names, arrays[:n_input_names])), arrays[n_input_names:])
                    if n_input_names
                    else (arrays[0], arrays[1:])
                )
                y, arrays = (
                    (dict(zip(target_names, arrays[:n_target_names])), arrays[n_target_names:])
                    if n_target_names
                    else (arrays[0], arrays[1:])
                )
                w = arrays[0]
                yield x, y, w
