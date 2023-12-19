# coding: utf-8

from __future__ import annotations

import os
from collections import defaultdict
from getpass import getuser
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import tensorflow as tf

from util import load_sample, phi_mpi_to_pi, calc_new_columns, get_device
from custom_layers import CustomEmbeddingLayer
from multi_dataset import MultiDataset


this_dir = os.path.dirname(os.path.realpath(__file__))

i32 = np.int32
f32 = np.float32

device = get_device(device="gpu", num_device=0)

# for debugging
# tf.config.run_functions_eagerly(True)
# tf.debugging.set_log_device_placement(True)


# common column configs (TODO: they should live in a different file)
dynamic_columns = {
    "DeepMET_ResolutionTune_phi": (("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py"), (lambda x, y: np.arctan2(y, x))),
    "met_dphi": (("met_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
    "dmet_resp_px": (("DeepMET_ResponseTune_px", "DeepMET_ResponseTune_py", "DeepMET_ResolutionTune_phi"), (lambda x, y, p: np.cos(-p) * x - np.sin(-p) * y)),
    "dmet_resp_py": (("DeepMET_ResponseTune_px", "DeepMET_ResponseTune_py", "DeepMET_ResolutionTune_phi"), (lambda x, y, p: np.sin(-p) * x + np.cos(-p) * y)),
    "dmet_reso_px": (("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py", "DeepMET_ResolutionTune_phi"), (lambda x, y, p: np.cos(-p) * x - np.sin(-p) * y)),
    "dmet_reso_py": (("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py", "DeepMET_ResolutionTune_phi"), (lambda x, y, p: np.sin(-p) * x + np.cos(-p) * y)),
    "met_px": (("met_et", "met_dphi"), (lambda a, b: a * np.cos(b))),
    "met_py": (("met_et", "met_dphi"), (lambda a, b: a * np.sin(b))),
    "dau1_dphi": (("dau1_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
    "dau2_dphi": (("dau2_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
    "genNu1_dphi": (("genNu1_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
    "genNu2_dphi": (("genNu2_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
    "dau1_px": (("dau1_pt", "dau1_dphi"), (lambda a, b: a * np.cos(b))),
    "dau1_py": (("dau1_pt", "dau1_dphi"), (lambda a, b: a * np.sin(b))),
    "dau1_pz": (("dau1_pt", "dau1_eta"), (lambda a, b: a * np.sinh(b))),
    "dau1_m": (("dau1_px", "dau1_py", "dau1_pz", "dau1_e"), (lambda x, y, z, e: np.sqrt(e ** 2 - (x ** 2 + y ** 2 + z ** 2)))),
    "dau2_px": (("dau2_pt", "dau2_dphi"), (lambda a, b: a * np.cos(b))),
    "dau2_py": (("dau2_pt", "dau2_dphi"), (lambda a, b: a * np.sin(b))),
    "dau2_pz": (("dau2_pt", "dau2_eta"), (lambda a, b: a * np.sinh(b))),
    "dau2_m": (("dau2_px", "dau2_py", "dau2_pz", "dau2_e"), (lambda x, y, z, e: np.sqrt(e ** 2 - (x ** 2 + y ** 2 + z ** 2)))),
    "ditau_deltaphi": (("dau1_dphi", "dau2_dphi"), (lambda a, b: np.abs(phi_mpi_to_pi(a - b)))),
    "ditau_deltaeta": (("dau1_eta", "dau2_eta"), (lambda a, b: np.abs(a - b))),
    "genNu1_px": (("genNu1_pt", "genNu1_dphi"), (lambda a, b: a * np.cos(b))),
    "genNu1_py": (("genNu1_pt", "genNu1_dphi"), (lambda a, b: a * np.sin(b))),
    "genNu1_pz": (("genNu1_pt", "genNu1_eta"), (lambda a, b: a * np.sinh(b))),
    "genNu2_px": (("genNu2_pt", "genNu2_dphi"), (lambda a, b: a * np.cos(b))),
    "genNu2_py": (("genNu2_pt", "genNu2_dphi"), (lambda a, b: a * np.sin(b))),
    "genNu2_pz": (("genNu2_pt", "genNu2_eta"), (lambda a, b: a * np.sinh(b))),
    "bjet1_dphi": (("bjet1_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
    "bjet1_px": (("bjet1_pt", "bjet1_dphi"), (lambda a, b: a * np.cos(b))),
    "bjet1_py": (("bjet1_pt", "bjet1_dphi"), (lambda a, b: a * np.sin(b))),
    "bjet1_pz": (("bjet1_pt", "bjet1_eta"), (lambda a, b: a * np.sinh(b))),
    "bjet2_dphi": (("bjet2_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
    "bjet2_px": (("bjet2_pt", "bjet2_dphi"), (lambda a, b: a * np.cos(b))),
    "bjet2_py": (("bjet2_pt", "bjet2_dphi"), (lambda a, b: a * np.sin(b))),
    "bjet2_pz": (("bjet2_pt", "bjet2_eta"), (lambda a, b: a * np.sinh(b))),
}

# possible values of categorical inputs (TODO: they should live in a different file)
embedding_expected_inputs = {
    "pairType": [0, 1, 2],
    "dau1_decayMode": [-1, 0, 1, 10, 11],  # -1 for e/mu
    "dau2_decayMode": [0, 1, 10, 11],
    "dau1_charge": [-1, 1],
    "dau2_charge": [-1, 1],
    "spin": [0, 2],
}


@dataclass
class ActivationSetting:
    # name of the activation as understood by tf.keras.layers.Activation
    name: str
    # name of the kernel initializer as understood by tf.keras.layers.Dense
    weight_init: str
    # whether to apply batch normalization before or after the activation (and if at all)
    batch_norm: tuple[bool, bool]
    # name of the dropout layer under tf.keras.layers
    dropout_name: str = "Dropout"


# setting for typical activations (TODO: they should live in a different file)
activation_settings = {
    "elu": ActivationSetting("ELU", "he_uniform", (True, False)),
    "relu": ActivationSetting("ReLU", "he_uniform", (False, True)),
    "prelu": ActivationSetting("PReLU", "he_normal", (True, False)),
    "selu": ActivationSetting("selu", "lecun_normal", (False, False), "AlphaDropout"),
    "tanh": ActivationSetting("tanh", "glorot_normal", (True, False)),
    "softmax": ActivationSetting("softmax", "glorot_normal", (True, False)),
    "swish": ActivationSetting("swish", "he_normal", (True, False)),
}


@dataclass
class Sample:
    name: str
    class_weight: float
    labels: list[int]
    spin: int = -1
    mass: float = -1.0


def main(
    model_name: str = "test_lreg50_elu_bn",
    data_dir: str = os.environ["TN_SKIMS_2017"],
    cache_dir: str = os.path.join(os.environ["TN_DATA_BASE"], "cache"),
    tensorboard_dir: str = f"/tmp/tensorboard_{getuser()}",
    samples: list[Sample] = [
        # Sample("SKIM_ggF_Radion_m250", 1.0, [1, 0, 0], 0, 250.0),
        # Sample("SKIM_ggF_Radion_m260", 1.0, [1, 0, 0], 0, 260.0),
        # Sample("SKIM_ggF_Radion_m270", 1.0, [1, 0, 0], 0, 270.0),
        # Sample("SKIM_ggF_Radion_m280", 1.0, [1, 0, 0], 0, 280.0),
        # Sample("SKIM_ggF_Radion_m300", 1.0, [1, 0, 0], 0, 300.0),
        Sample("SKIM_ggF_Radion_m320", 1.0, [1, 0, 0], 0, 320.0),
        Sample("SKIM_ggF_Radion_m350", 1.0, [1, 0, 0], 0, 350.0),
        Sample("SKIM_ggF_Radion_m400", 1.0, [1, 0, 0], 0, 400.0),
        Sample("SKIM_ggF_Radion_m450", 1.0, [1, 0, 0], 0, 450.0),
        Sample("SKIM_ggF_Radion_m500", 1.0, [1, 0, 0], 0, 500.0),
        Sample("SKIM_ggF_Radion_m550", 1.0, [1, 0, 0], 0, 550.0),
        Sample("SKIM_ggF_Radion_m600", 1.0, [1, 0, 0], 0, 600.0),
        Sample("SKIM_ggF_Radion_m650", 1.0, [1, 0, 0], 0, 650.0),
        Sample("SKIM_ggF_Radion_m700", 1.0, [1, 0, 0], 0, 700.0),
        Sample("SKIM_ggF_Radion_m750", 1.0, [1, 0, 0], 0, 750.0),
        Sample("SKIM_ggF_Radion_m800", 1.0, [1, 0, 0], 0, 800.0),
        Sample("SKIM_ggF_Radion_m850", 1.0, [1, 0, 0], 0, 850.0),
        Sample("SKIM_ggF_Radion_m900", 1.0, [1, 0, 0], 0, 900.0),
        Sample("SKIM_ggF_Radion_m1000", 1.0, [1, 0, 0], 0, 1000.0),
        Sample("SKIM_ggF_Radion_m1250", 1.0, [1, 0, 0], 0, 1250.0),
        Sample("SKIM_ggF_Radion_m1500", 1.0, [1, 0, 0], 0, 1500.0),
        Sample("SKIM_ggF_Radion_m1750", 1.0, [1, 0, 0], 0, 1750.0),
        # Sample("SKIM_ggF_BulkGraviton_m250", 1.0, [1, 0, 0], 2, 250.0),
        # Sample("SKIM_ggF_BulkGraviton_m260", 1.0, [1, 0, 0], 2, 260.0),
        # Sample("SKIM_ggF_BulkGraviton_m270", 1.0, [1, 0, 0], 2, 270.0),
        # Sample("SKIM_ggF_BulkGraviton_m280", 1.0, [1, 0, 0], 2, 280.0),
        # Sample("SKIM_ggF_BulkGraviton_m300", 1.0, [1, 0, 0], 2, 300.0),
        Sample("SKIM_ggF_BulkGraviton_m320", 1.0, [1, 0, 0], 2, 320.0),
        Sample("SKIM_ggF_BulkGraviton_m350", 1.0, [1, 0, 0], 2, 350.0),
        Sample("SKIM_ggF_BulkGraviton_m400", 1.0, [1, 0, 0], 2, 400.0),
        Sample("SKIM_ggF_BulkGraviton_m450", 1.0, [1, 0, 0], 2, 450.0),
        Sample("SKIM_ggF_BulkGraviton_m500", 1.0, [1, 0, 0], 2, 500.0),
        Sample("SKIM_ggF_BulkGraviton_m550", 1.0, [1, 0, 0], 2, 550.0),
        Sample("SKIM_ggF_BulkGraviton_m600", 1.0, [1, 0, 0], 2, 600.0),
        Sample("SKIM_ggF_BulkGraviton_m650", 1.0, [1, 0, 0], 2, 650.0),
        Sample("SKIM_ggF_BulkGraviton_m700", 1.0, [1, 0, 0], 2, 700.0),
        Sample("SKIM_ggF_BulkGraviton_m750", 1.0, [1, 0, 0], 2, 750.0),
        Sample("SKIM_ggF_BulkGraviton_m800", 1.0, [1, 0, 0], 2, 800.0),
        Sample("SKIM_ggF_BulkGraviton_m850", 1.0, [1, 0, 0], 2, 850.0),
        Sample("SKIM_ggF_BulkGraviton_m900", 1.0, [1, 0, 0], 2, 900.0),
        Sample("SKIM_ggF_BulkGraviton_m1000", 1.0, [1, 0, 0], 2, 1000.0),
        Sample("SKIM_ggF_BulkGraviton_m1250", 1.0, [1, 0, 0], 2, 1250.0),
        Sample("SKIM_ggF_BulkGraviton_m1500", 1.0, [1, 0, 0], 2, 1500.0),
        Sample("SKIM_ggF_BulkGraviton_m1750", 1.0, [1, 0, 0], 2, 1750.0),
        Sample("SKIM_DY_amc_incl", 1.0, [0, 1, 0], -1, -1.0),
        # Sample("SKIM_TT_fullyLep", 1.0, [0, 0, 1], -1, -1.0),
        # Sample("SKIM_TT_semiLep", 1.0, [0, 0, 1], -1, -1.0),
        # Sample("SKIM_GluGluHToTauTau", 1.0, [0, 0, 0, 0, 1, 0], -1, -1.0),
        # Sample("SKIM_ttHToTauTau", 1.0, [0, 0, 0, 1], -1, -1.0),
    ],
    # additional columns to load
    extra_columns: list[str] = [
        "EventNumber",
    ],
    # selections to apply before training
    selections: list[tuple[tuple[str, ...], Callable]] = [
        (("nbjetscand",), (lambda a: a > 1)),
        (("pairType",), (lambda a: a < 3)),
        (("nleps",), (lambda a: a == 0)),
        (("isOS",), (lambda a: a == 1)),
        (("dau2_deepTauVsJet",), (lambda a: a >= 5)),
        (
            ("pairType", "dau1_iso", "dau1_eleMVAiso", "dau1_deepTauVsJet"),
            (lambda a, b, c, d: (((a == 0) & (b < 0.15)) | ((a == 1) & (c == 1)) | ((a == 2) & (d >= 5)))),
        ),
    ],
    # categorical input features for the network
    cat_input_names: list[str] = [
        "pairType", "dau1_decayMode", "dau2_decayMode", "dau1_charge", "dau2_charge",
    ],
    # continuous input features to the network
    cont_input_names: list[str] = [
        "met_px", "met_py", "dmet_resp_px", "dmet_resp_py", "dmet_reso_px",
        "met_cov00", "met_cov01", "met_cov11",
        "ditau_deltaphi", "ditau_deltaeta",
        *[
            f"dau{i}_{feat}"
            for i in [1, 2]
            for feat in ["px", "py", "pz", "e", "dxy", "dz", "iso"]
        ],
        *[
            f"bjet{i}_{feat}"
            for i in [1, 2]
            for feat in [
                "px", "py", "pz", "e", "btag_deepFlavor", "cID_deepFlavor", "pnet_bb", "pnet_cc", "pnet_b", "pnet_c",
                "pnet_g", "pnet_uds", "pnet_pu", "pnet_undef", "HHbtag",
            ]
        ],
    ],
    # number of layers and units, second entry determines the extra heads (if applicable, otherwise "concatenate")
    units: list[int] = [125] * 5,
    # dimension of the embedding layer output will be embedding_output_dim x N_categorical_features
    embedding_output_dim: int = 5,
    # activation function after each hidden layer
    activation: str = "elu",
    # scale fot the l2 loss term (which is already normalized to the number of weights)
    l2_norm: float = 50.0,
    # dropout percentage
    dropout_rate: float = 0.0,
    # batch norm between layers
    batch_norm: bool = True,
    # batch size
    batch_size: int = 4096,
    # learning rate to start with
    initial_learning_rate: float = 3e-3,
    # half the learning rate if the validation loss hasn't improved in this many validation steps
    learning_rate_patience: int = 4,
    # how even the learning rate is halfed before training is stopped
    max_learning_rate_reductions: int = 5,
    # stop training if the validation loss hasn't improved since this many validation steps
    early_stopping_patience: int = 10,
    # divide events by this based on EventNumber
    train_valid_eventnumber_modulo: int = 4,
    # assign event to validation dataset if the rest is this
    train_valid_eventnumber_rest: int = 0,
    # how frequently to calulcate the validation loss
    validate_every: int = 500,
    # add the generator spin for the signal samples as categorical input -> network parameterized in spin
    parameterize_spin: bool = True,
    # add the generator mass for the signal samples as continuous input -> network parameterized in mass
    parameterize_mass: bool = True,
):
    # some checks
    assert "spin" not in cat_input_names
    assert "mass" not in cont_input_names

    # determine which columns to read
    columns_to_read = set()
    for name in cont_input_names + cat_input_names:
        columns_to_read.add(name)
    for sel_columns, _ in selections:
        columns_to_read |= set(sel_columns)
    columns_to_read |= set(extra_columns)
    # expand dynamic columns, keeping track of those that are needed
    all_dyn_names = set(dynamic_columns)
    dyn_names = set()
    while (to_expand := columns_to_read & all_dyn_names):
        for name in to_expand:
            columns_to_read |= set(dynamic_columns[name][0])
        columns_to_read -= to_expand
        dyn_names |= to_expand

    # order dynamic columns to be added
    all_dyn_names = list(dynamic_columns)
    dyn_names = sorted(dyn_names, key=all_dyn_names.index)

    # get lists of embedded feature values
    possible_cont_input_values = [deepcopy(embedding_expected_inputs[name]) for name in cat_input_names]

    # scan samples and their labels to construct relative weights such that each class starts with equal importance
    labels_to_samples = defaultdict(list)
    for sample in samples:
        labels_to_samples[tuple(sample.labels)].append(sample.name)

    # keep track of spins, masses, number of events per sample, and relative batch weights per sample
    spins: set[f32] = set()
    masses: set[f32] = set()
    all_n_events: list[int] = []
    batch_weights: list[float] = []

    # lists for collection data to be forwarded into the MultiDataset
    cont_inputs_train, cont_inputs_valid = [], []
    cat_inputs_train, cat_inputs_valid = [], []
    labels_train, labels_valid = [], []
    event_weights_train, event_weights_valid = [], []

    # helper to flatten rec arrays
    flatten_rec = lambda r, t: r.astype([(n, t) for n in r.dtype.names], copy=False).view(t).reshape((-1, len(r.dtype)))

    # loop through samples
    for sample in samples:
        rec, event_weights = load_sample(
            data_dir,
            sample.name,
            sample.class_weight,
            list(columns_to_read),
            selections,
            # maxevents=10000,
            cache_dir=cache_dir,
        )
        all_n_events.append(n_events := len(event_weights))

        # compute the batch weight, i.e. the weight that ensure that each class is equally represented in each batch
        batch_weights.append(1 / len(labels_to_samples[tuple(sample.labels)]))

        # add dynamic columns
        rec = calc_new_columns(rec, {name: dynamic_columns[name] for name in dyn_names})

        # prepare arrays
        cont_inputs = flatten_rec(rec[cont_input_names], f32)
        cat_inputs = flatten_rec(rec[cat_input_names], i32)
        labels = np.array([sample.labels] * n_events, dtype=f32)

        # add spin and mass if given
        if parameterize_mass:
            if sample.mass > -1:
                masses.add(float(sample.mass))
            cont_inputs = np.append(cont_inputs, (np.ones(n_events, dtype=f32) * sample.mass)[:, None], axis=1)
        if parameterize_spin:
            if sample.spin > -1:
                spins.add(int(sample.spin))
            cat_inputs = np.append(cat_inputs, (np.ones(n_events, dtype=i32) * sample.spin)[:, None], axis=1)

        # training and validation mask
        train_mask = (rec["EventNumber"] % train_valid_eventnumber_modulo) != train_valid_eventnumber_rest
        valid_mask = ~train_mask

        # fill dataset lists
        cont_inputs_train.append(cont_inputs[train_mask])
        cont_inputs_valid.append(cont_inputs[valid_mask])

        cat_inputs_train.append(cat_inputs[train_mask])
        cat_inputs_valid.append(cat_inputs[valid_mask])

        labels_train.append(labels[train_mask])
        labels_valid.append(labels[valid_mask])

        event_weights_train.append(event_weights[train_mask][..., None])
        event_weights_valid.append(event_weights[valid_mask][..., None])

    # determine contiuous input means and variances
    cont_input_means = (
        np.sum(np.concatenate([inp * bw / len(inp) for inp, bw in zip(cont_inputs_train, batch_weights)]), axis=0) /
        sum(batch_weights)
    )
    cont_input_vars = (
        np.sum(np.concatenate([inp**2 * bw / len(inp) for inp, bw in zip(cont_inputs_train, batch_weights)]), axis=0) /
        sum(batch_weights)
    ) - cont_input_means**2

    # handle spins
    spins = sorted(spins)
    if parameterize_spin:
        cat_input_names.append("spin")
        # add to possible embedding values
        possible_cont_input_values.append(embedding_expected_inputs["spin"])

    # handle masses
    masses = sorted(masses)
    if parameterize_mass:
        cont_input_names.append("mass")
        # replace mean and var with unweighted values
        cont_input_means[-1] = np.mean(masses)
        cont_input_vars[-1] = np.var(masses)

    # live transformation of inputs to inject spin and mass for backgrounds
    def transform(cont_inputs, cat_inputs, labels, weights):
        if parameterize_mass:
            random_masses = tf.gather(masses, tf.random.categorical([[1.0] * len(masses)], cont_inputs.shape[0]))[0]
            filled_masses = tf.where(cont_inputs[:, -1] < 0, random_masses, cont_inputs[:, -1])
            cont_inputs = tf.concat([cont_inputs[:, :-1], filled_masses[..., None]], axis=1)
        if parameterize_mass:
            random_spins = tf.gather(spins, tf.random.categorical([[1.0] * len(spins)], cat_inputs.shape[0]))[0]
            filled_spins = tf.where(cat_inputs[:, -1] < 0, random_spins, cat_inputs[:, -1])
            cat_inputs = tf.concat([cat_inputs[:, :-1], filled_spins[..., None]], axis=1)
        return cont_inputs, cat_inputs, labels, weights

    # build datasets
    dataset_train = MultiDataset(
        data=zip(zip(cont_inputs_train, cat_inputs_train, labels_train, event_weights_train), batch_weights),
        batch_size=batch_size,
        kind="train",
        transform_data=transform,
    )
    dataset_valid = MultiDataset(
        data=zip(zip(cont_inputs_valid, cat_inputs_valid, labels_valid, event_weights_valid), batch_weights),
        batch_size=batch_size,
        kind="valid",
        transform_data=transform,
    )

    with device:
        # create the model
        model = create_model(
            n_cont_inputs=len(cont_input_names),
            n_cat_inputs=len(cat_input_names),
            n_classes=labels_train[0].shape[1],
            embedding_expected_inputs=possible_cont_input_values,
            embedding_output_dim=embedding_output_dim,
            cont_input_means=cont_input_means,
            cont_input_vars=cont_input_vars,
            units=units,
            activation=activation,
            batch_norm=batch_norm,
            l2_norm=l2_norm,
            dropout_rate=dropout_rate,
        )
        model.summary()

        # compile
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
            metrics=[
                tf.keras.metrics.CategoricalCrossentropy(name="ce"),
                tf.keras.metrics.CategoricalAccuracy(name="acc"),
                tf.keras.metrics.AUC(name="auc"),
            ],
            jit_compile=False,
            run_eagerly=False,
        )

        # callbacks
        fit_callbacks = []
        if tensorboard_dir:
            fit_callbacks.append(tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(tensorboard_dir, model_name),
                write_graph=True,
                histogram_freq=1,
                profile_batch="500,520",
            ))

        model.fit(
            x=dataset_train.create_keras_generator(input_names=["cont_input", "cat_input"]),
            validation_data=dataset_valid.create_keras_generator(input_names=["cont_input", "cat_input"]),
            shuffle=False,
            epochs=30,
            steps_per_epoch=validate_every,
            validation_freq=1,
            validation_steps=dataset_valid.max_iter_valid,
            callbacks=fit_callbacks,
        )

    # TODOs (in order):
    # - l2 in metrics and tensorboard
    # - early stopping
    # - learning rate decay or cycle
    # - checkpointing (always save best weights after n-th step)
    # - other samples
    # - binary or multi-class?
    # - tauNN
    # - proper model names
    # - k-fold xvalidation style data usage (with the modulos above)
    # - seed change and ensembling encapuslation after trainings (but likely not in this script)
    # - skip connections, hyper-opt, symmetric CCE and group weight


# functional model builder for later use with hyperparameter optimization tools
# via https://www.tensorflow.org/tutorials/keras/keras_tuner
def create_model(
    *,
    n_cont_inputs: int,
    n_cat_inputs: int,
    n_classes: int,
    embedding_expected_inputs: list[list[int]],
    embedding_output_dim: int,
    cont_input_means: np.ndarray,
    cont_input_vars: np.ndarray,
    units: list[int],
    activation: str,
    batch_norm: bool,
    l2_norm: float,
    dropout_rate: float,
):
    # checks
    assert len(embedding_expected_inputs) == n_cat_inputs
    assert len(cont_input_means) == len(cont_input_vars) == n_cont_inputs
    assert len(units) > 0

    # get activation settings
    act_settings: ActivationSetting = activation_settings[activation]

    # prepare l2 regularization, use a dummy value as it is replaced after the model is built
    l2_reg = tf.keras.regularizers.l2(1.0) if l2_norm > 0 else None

    # input layers
    x_cont = tf.keras.Input(n_cont_inputs, name="cont_input")
    x_cat = tf.keras.Input(n_cat_inputs, name="cat_input")

    # normalize continuous inputs
    norm_layer = tf.keras.layers.Normalization(mean=cont_input_means, variance=cont_input_vars, name="norm")
    a = norm_layer(x_cont)

    # embedding layer
    if n_cat_inputs > 0:
        embedding_layer = CustomEmbeddingLayer(
            output_dim=embedding_output_dim,
            expected_inputs=embedding_expected_inputs,
            name="cat_embedding",
        )
        embed_cat = embedding_layer(x_cat)

        # combine with continuous inputs
        a = tf.keras.layers.Concatenate(name="concat")([a, embed_cat])

    # add layers programatically
    for i, n_units in enumerate(units, 1):
        # dense
        dense_layer = tf.keras.layers.Dense(
            n_units,
            use_bias=True,
            kernel_initializer=act_settings.weight_init,
            kernel_regularizer=l2_reg,
            name=f"dense_{i}")
        a = dense_layer(a)

        # batch norm before activation if requested
        batchnorm_layer = tf.keras.layers.BatchNormalization(dtype="float32", name=f"norm_{i}")
        if batch_norm and act_settings.batch_norm[0]:
            a = batchnorm_layer(a)

        # activation
        a = tf.keras.layers.Activation(act_settings.name, name=f"act_{i}")(a)

        # batch norm after activation if requested
        if batch_norm and act_settings.batch_norm[1]:
            a = batchnorm_layer(a)

        # add random unit dropout
        if dropout_rate:
            dropout_cls = getattr(tf.keras.layers, act_settings.dropout_name)
            a = dropout_cls(dropout_rate, name=f"do_{i}")(a)

    # add the output layer
    output_layer = tf.keras.layers.Dense(
        n_classes,
        activation="softmax",
        use_bias=True,
        kernel_initializer=activation_settings["softmax"].weight_init,
        kernel_regularizer=l2_reg,
        name="output",
    )
    y = output_layer(a)

    # build the model
    model = tf.keras.Model(inputs=[x_cont, x_cat], outputs=[y], name="bbtautau_classifier")

    # normalize the l2 regularization to the number of weights in dense layers
    if l2_norm > 0:
        n_weights = sum(map(
            tf.keras.backend.count_params,
            [layer.kernel for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)],
        ))
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense) and layer.kernel_regularizer is not None:
                layer.kernel_regularizer.l2[...] = l2_norm / n_weights

    return model


# # custom losses
# # TODO: still needed?
# def create_losses(regularization_weights, l2_norm=10.0):
#     # dictionary of losses to be returned
#     loss_dict = {}

#     # cross entropy
#     @tf.function(reduce_retracing=True)
#     def loss_ce_fn(**kwargs):
#         labels = kwargs["labels"]
#         predictions = kwargs["predictions"]
#         event_weights = kwargs["event_weights"]
#         with device:
#             # ensure proper prediction values before applying log's
#             predictions = tf.clip_by_value(predictions, 1e-6, 1 - 1e-6)
#             loss_ce = tf.reduce_mean(
#                 event_weights * -labels * tf.math.log(predictions))
#             return loss_ce

#     loss_dict["ce"] = loss_ce_fn

#     # l2 loss
#     if l2_norm > 0:
#         # total number of weights to be regularized
#         n_reg_weights = sum(functools.reduce(mul, w.shape) for w in regularization_weights)

#         @tf.function(reduce_retracing=True)
#         def loss_l2_fn(**kwargs):
#             with device:
#                 # accept labels and predictions although we don't need them
#                 # but this makes it easier to call all loss functions the same way
#                 loss_l2 = sum(tf.reduce_sum(w ** 2) for w in regularization_weights)
#                 return l2_norm / n_reg_weights * loss_l2

#         loss_dict["l2"] = loss_l2_fn

#     return loss_dict


# # TODO: still needed?
# def create_optimizer(initial_learning_rate):
#     with device:
#         learning_rate = tf.Variable(initial_learning_rate, dtype=tf.float32, trainable=False)
#         optimizer = tf.keras.optimizers.Adam(learning_rate)
#     return optimizer, learning_rate


if __name__ == "__main__":
    main()
