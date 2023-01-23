# coding: utf-8

from __future__ import annotations

import numpy as np
import tensorflow as tf


class MultiDataset(object):

    def __init__(
        self,
        data: tuple[tuple[np.array] | np.array, float],
        batch_size: int = 128,
        repeat: bool = True,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        super().__init__()

        # create datasets, store counts and relative weights
        self.datasets = []
        self.counts = []
        self.weights = []
        for arrays, weight in data:
            if not isinstance(arrays, tuple):
                arrays = (arrays,)
            self.tuple_length = len(arrays)
            self.datasets.append(tf.data.Dataset.from_tensor_slices(arrays))
            self.counts.append(len(arrays[0]))
            self.weights.append(weight)

        # state attributes
        self.batches_seen = None

        # determine batch sizes per dataset
        self.batch_sizes = []
        sum_weights = sum(self.weights)

        carry = 0.0
        for weight in self.weights:
            bs = weight / sum_weights * batch_size - carry
            bs_int = int(round(bs))
            carry = bs_int - bs
            self.batch_sizes.append(bs_int)

        if batch_size != sum(self.batch_sizes):
            print(f"batch size is {sum(self.batch_sizes)} but should be {batch_size}")

        self.datasets = [
            dataset.batch(bs_size)
            for dataset, bs_size in zip(self.datasets, self.batch_sizes)
        ]

        # shuffling
        if shuffle:
            self.datasets = [
                dataset.shuffle(10 * count, reshuffle_each_iteration=True, seed=seed)
                for dataset, count in zip(self.datasets, self.counts)
            ]

        # repitition
        self.datasets = [
            dataset.repeat(-1 if repeat else 1)
            for dataset in self.datasets
        ]

    @property
    def n_datasets(self):
        return len(self.datasets)

    def __iter__(self):
        self.batches_seen = 0

        its = [iter(dataset) for dataset in self.datasets]
        while True:
            dataset_batches = []
            do_continue = False
            do_break = False
            for i, it in enumerate(its):
                try:
                    dataset_batches.append(next(it))
                except tf.errors.DataLossError as e:
                    print(f"\nDataLossError in dataset {i}:\n{e}\n")
                    do_continue = True
                    break
                except StopIteration:
                    do_break = True
                    break

            if do_continue:
                continue
            if do_break:
                break

            yield tuple(tf.concat([batch[i] for batch in dataset_batches], axis=0)for i in range(self.tuple_length))

            self.batches_seen += 1

    def map(self, *args, **kwargs):
        for key, dataset in list(self._datasets.items()):
            self._datasets[key] = dataset.map(*args, **kwargs)
