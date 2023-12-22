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
        transform_data: Callable[[tuple[tf.Tensor, ...]], tuple[tf.Tensor, ...]] | None = None,
    ):
        super().__init__()

        # attributes
        self.kind = DatasetKind.from_str(kind) if isinstance(kind, str) else kind
        self.yield_valid_rest = yield_valid_rest
        self.seed = seed
        self.transform_data = transform_data
        self.batches_seen = 0

        # create datasets, store counts and relative weights
        self.datasets = []
        self.counts = []
        self.batch_weights = []
        for arrays, batch_weight in data:
            if not isinstance(arrays, tuple):
                arrays = (arrays,)
            self.tuple_length = len(arrays)
            self.datasets.append(tf.data.Dataset.from_tensor_slices(arrays))
            self.counts.append(len(arrays[0]))
            self.batch_weights.append(batch_weight)

        # determine batch sizes per dataset during training iterations
        self.batch_sizes = []
        sum_batch_weights = sum(self.batch_weights)
        carry = 0.0
        for batch_weight in self.batch_weights:
            bs = batch_weight / sum_batch_weights * batch_size - carry
            bs_int = int(round(bs))
            carry = bs_int - bs
            self.batch_sizes.append(bs_int)
        if batch_size != sum(self.batch_sizes):
            print(f"batch size is {sum(self.batch_sizes)} but should be {batch_size}")

        # the number of batches that constitute one iteration cycle through all events
        # (floating point number, to be ceiled or floored depending on if rest is used)
        self._batches_per_cycle: float = len(self) / batch_size

    def __len__(self):
        return sum(self.counts)

    @property
    def batch_size(self) -> int:
        return sum(self.batch_sizes)

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
        datasets = self.datasets
        transform_data = self.transform_data if callable(self.transform_data) else lambda x: x

        # shuffle each dataset
        datasets = [
            dataset.shuffle(10 * count, reshuffle_each_iteration=True, seed=self.seed)
            for dataset, count in zip(datasets, self.counts)
        ]

        # repeat each dataset indefinitely
        datasets = [
            dataset.repeat(-1)
            for dataset in datasets
        ]

        # batch each dataset with its specific batch size
        datasets = [
            dataset.batch(bs_size)
            for dataset, bs_size in zip(datasets, self.batch_sizes)
        ]

        # start iterating
        its = [iter(dataset) for dataset in datasets]
        while True:
            chunks = []
            do_continue = False
            do_break = False
            for i, it in enumerate(its):
                try:
                    chunks.append(next(it))
                except tf.errors.DataLossError as e:
                    print(f"\nDataLossError in dataset {i}:\n{e}\n")
                    do_continue = True
                    break
                except StopIteration:
                    # this should not happen since repetition is used
                    print(f"\nStopIteration reached in dataset {i}, which should not happen\n")
                    do_break = True
                    break

            # next batch or stop completely
            if do_continue:
                continue
            if do_break:
                break

            # yield
            yield transform_data(*tuple(
                tf.concat([chunk[i] for chunk in chunks], axis=0)
                for i in range(self.tuple_length)
            ))
            self.batches_seen += 1

    def iter_valid(self):
        # preparations
        datasets = self.datasets
        batch_size = self.batch_size
        transform_data = self.transform_data if callable(self.transform_data) else lambda x: x

        # shuffle each dataset
        datasets = [
            dataset.shuffle(10 * count, reshuffle_each_iteration=False, seed=self.seed)
            for dataset, count in zip(datasets, self.counts)
        ]

        # batch each dataset with the total batch size
        datasets = [
            dataset.batch(batch_size)
            for dataset in datasets
        ]

        # start iterating
        dataset_index = -1
        dataset_iter = None
        chunks = []
        n_total = 0
        while True:
            # iterate until batch size is reached
            while n_total < batch_size:
                # optionally switch to next, unrepeated dataset
                if dataset_iter is None:
                    dataset_index = (dataset_index + 1) % len(datasets)
                    dataset_iter = iter(datasets[dataset_index])
                # get next chunk if iterator not broken or exhausted
                try:
                    chunks.append(chunk := next(dataset_iter))
                    n_total += len(chunk[0])
                except tf.errors.DataLossError as e:
                    print(f"\nDataLossError in dataset {dataset_index}:\n{e}\n")
                    dataset_iter = None
                    continue
                except StopIteration:
                    dataset_iter = None
                    # if this was the last dataset, yield the rest if desired
                    if dataset_index == len(datasets) - 1 and self.yield_valid_rest:
                        break
                    continue

            # concatenate chunks, cut off excess over batch size, but remember it for the next batch
            data = ()
            excess_chunk = ()
            for i in range(self.tuple_length):
                data += ((_data := tf.concat([chunk[i] for chunk in chunks], axis=0))[:batch_size],)
                excess_chunk += (_data[batch_size:],)
            chunks = [excess_chunk]
            n_total = len(excess_chunk[0])

            # yield
            yield transform_data(*data)
            self.batches_seen += 1

    def create_keras_generator(self, input_names: list[str] | None = None):
        # this assumes that at least three arrays are yielded by the __iter__ method: inputs, targets, weights
        # when input_names are given, the inputs array is split into a dictionary with the given names
        # when there is more than one input array, input_names are mandatory
        if self.tuple_length > 3:
            if not input_names:
                raise ValueError("input_names must be given when there is more than one output to be yielded")
            if len(input_names) != self.tuple_length - 2:
                raise ValueError(
                    f"input_names ({len(input_names)}) must have the same length as the number of input arrays "
                    f"(({self.tuple_length - 2}))",
                )

        # start generating
        for arrays in self:
            yield dict(zip(input_names, arrays[:2])) if input_names else arrays[:2], *arrays[2:]
