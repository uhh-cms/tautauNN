# coding: utf-8

from __future__ import annotations

import os
import math
import glob
import hashlib
import pickle
import inspect

import numpy as np
import numpy.lib.recfunctions as rfn
import tensorflow as tf


debug_layer = tf.autograph.experimental.do_not_convert


def load_sample(data_dir, sample, weight, features, selections, maxevents=1000000, cache_dir=None):
    print(f"loading sample {sample} ...")

    # potentially read from cache
    cache_path = get_cache_path(cache_dir, data_dir, sample, features, selections, maxevents)
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            feature_vecs = pickle.load(f)
        print(f"loaded {len(feature_vecs)} events from cache")

    else:
        feature_vecs = []
        nevents = 0
        # weightsum = 0
        filenames = glob.glob(f"{data_dir}/{sample}/output*.npz")
        for i, filename in enumerate(filenames, 1):
            with np.load(filename) as f:
                # weightsum += f["weightsum"]
                e = f["events"]
                mask = [True] * len(e)
                for (varnames, func) in selections:
                    variables = [e[v] for v in varnames]
                    mask = mask & func(*variables)
                feature_vecs.append(e[features][mask])
                nevents += len(feature_vecs[-1])
                if nevents > maxevents:
                    break
        feature_vecs = np.concatenate(feature_vecs, axis=0)
        print(f"done, found {len(feature_vecs)} events in {i} file(s)")

        # save to cache
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(feature_vecs, f)

    # weight vector
    weights = np.array([weight] * len(feature_vecs), dtype="float32")

    return feature_vecs, weights


def get_cache_path(cache_dir, data_dir, sample, features, selections, maxevents) -> str | None:
    if not cache_dir:
        return None

    cache_key = [
        os.path.expandvars(data_dir),
        sample,
        sorted(features),
        [(feats, inspect.getsource(func)) for feats, func in selections],
        maxevents,
    ]
    return os.path.join(cache_dir, hashlib.sha256(str(cache_key).encode("utf-8")).hexdigest()[:10] + ".pkl")


def calc_new_columns(data, rules):
    columns = []
    column_names = []
    for name, (input_columns, func) in rules.items():
        if name in data.dtype.names:
            continue
        input_values = [columns[column_names.index(c)] if c in column_names else data[c] for c in input_columns]
        column = func(*input_values)
        columns.append(column)
        column_names.append(name)
    data = rfn.rec_append_fields(data, list(rules.keys()), columns, dtypes=["<f4"] * len(columns))
    return data


def calc_4vec_sum(pt1, eta1, phi1, e1, pt2, eta2, phi2, e2):
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)

    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    e = e1 + e2

    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(pt**2 + pz**2)
    theta = np.arccos(pz / p)
    eta = -np.log(np.tan(theta / 2))
    phi = np.arccos(px / pt)
    phi[py == 0] = 0
    phi[py < 0] = -np.arccos(px / pt)[py < 0]

    return pt, eta, phi, e


def calc_energy(pt, eta, phi, m):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    energy = np.sqrt(m**2 + px**2 + py**2 + pz**2)

    return energy


def calc_mass(pt, eta, phi, e):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    mass = np.sqrt(e**2 - px**2 - py**2 - pz**2)
    mass[np.isnan(mass)] = 0

    return mass


def phi_mpi_to_pi(phi):
    larger_pi = phi > math.pi
    smaller_pi = phi < -math.pi
    while np.any(larger_pi) or np.any(smaller_pi):
        phi[larger_pi] -= 2 * math.pi
        phi[smaller_pi] += 2 * math.pi
        larger_pi = phi > math.pi
        smaller_pi = phi < -math.pi
    return phi


def create_tensorboard_callbacks(log_dir):
    add = flush = lambda *args, **kwargs: None
    if log_dir:
        writer = tf.summary.create_file_writer(log_dir)
        flush = writer.flush

        def add(attr, *args, **kwargs):  # noqa
            with writer.as_default():
                getattr(tf.summary, attr)(*args, **kwargs)
    return add, flush


def get_device(device="cpu", num_device=0):
    """
    Check if there is a main gpu and raises error if not. Returns a tf.device wrapper with the device
    Args:
        device: cpu or gpu, Default: CPU

    Returns: tf.device
    """
    if device == "gpu":
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if len(gpus) == 0:
            print("There is no GPU on the working machine, default to CPU")
        else:
            if gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpus[num_device], True)
                except RuntimeError as e:
                    print(e)
            return tf.device(f"/device:GPU:{num_device}")

    return tf.device(f"/device:CPU:{num_device}")


class L2Metric(tf.keras.metrics.Metric):

    def __init__(self, model: tf.keras.Model, name: str = "l2", **kwargs) -> L2Metric:
        super().__init__(name=name, **kwargs)

        # store kernels and l2 norms of dense layers
        self.kernels: list[tf.Tensor] = []
        self.norms: list[np.ndarray] = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense) and layer.kernel_regularizer is not None:
                self.kernels.append(layer.kernel)
                self.norms.append(layer.kernel_regularizer.l2)

        # book the l2 metric
        self.l2: tf.Variable = self.add_weight(name="l2", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.l2.assign(tf.add_n([tf.reduce_sum(k**2) * n for k, n in zip(self.kernels, self.norms)]))

    def result(self):
        return self.l2

    def reset_states(self):
        self.l2.assign(0.0)
