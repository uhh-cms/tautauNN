#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import numpy as np
import sys
import os
import functools
from operator import mul
from collections import defaultdict
from util import load_sample, phi_mpi_to_pi, split_train_validation_mask, calc_new_columns, create_tensorboard_callbacks, get_device
from custom_layers import CustomEmbeddingLayer, CustomOutputScalingLayer
from multi_dataset import MultiDataset

gpu = get_device(device="gpu", num_device=0)
tf.debugging.set_log_device_placement(True)


def main(model_name="no_singleH_add_bjetvars_3classification_massloss_simonesSelection",
         basepath="/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v4_02Mar23",
         tensorboard_dir="/tmp/tensorboard",
         # tensorboard_dir=None,
         samples={
             "SKIM_GGHH_SM": (1./35, 1., [1, 0, 0]),  # (batch fraction weight, event weight factor)
             "SKIM_ggF_Radion_m300": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_Radion_m350": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_Radion_m400": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_Radion_m450": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_Radion_m500": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_Radion_m550": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_Radion_m600": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_Radion_m650": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_Radion_m700": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_Radion_m750": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_Radion_m800": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_Radion_m850": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_Radion_m900": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_Radion_m1000": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_Radion_m1250": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_Radion_m1500": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_Radion_m1750": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m300": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m350": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m400": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m450": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m500": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m550": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m600": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m650": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m700": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m750": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m800": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m850": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m900": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m1000": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m1250": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m1500": (1./35, 1., [1, 0, 0]),
             "SKIM_ggF_BulkGraviton_m1750": (1./35, 1., [1, 0, 0]),
             "SKIM_DY_amc_incl": (1., 1., [0, 1, 0]),
             "SKIM_TT_fullyLep": (1., 1., [0, 0, 1]),
             # "SKIM_TT_semiLep": (1., 1., [0, 0, 1]),
             # "SKIM_GluGluHToTauTau": (1., 1.),
             # "SKIM_ttHToTauTau": (1., 1.),
         },
         columns_to_read=[
             "dau1_pt", "dau1_eta", "dau1_phi", "dau1_e", "dau1_dxy", "dau1_dz", "dau1_iso",
             "dau2_pt", "dau2_eta", "dau2_phi", "dau2_e", "dau2_dxy", "dau2_dz", "dau2_iso",
             "met_et", "met_phi", "met_cov00", "met_cov01", "met_cov11",
             "bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_e", "bjet1_btag_deepFlavor", "bjet1_cID_deepFlavor", "bjet1_pnet_bb", "bjet1_pnet_cc", "bjet1_pnet_b", "bjet1_pnet_c", "bjet1_pnet_g", "bjet1_pnet_uds", "bjet1_pnet_pu", "bjet1_pnet_undef", "bjet1_HHbtag",
             "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_e", "bjet2_btag_deepFlavor", "bjet2_cID_deepFlavor", "bjet2_pnet_bb", "bjet2_pnet_cc", "bjet2_pnet_b", "bjet2_pnet_c", "bjet2_pnet_g", "bjet2_pnet_uds", "bjet2_pnet_pu", "bjet2_pnet_undef", "bjet2_HHbtag",
             "pairType", "dau1_decayMode", "dau2_decayMode",
             "genNu1_pt", "genNu1_eta", "genNu1_phi",  # "genNu1_e",
             "genNu2_pt", "genNu2_eta", "genNu2_phi",  # "genNu2_e",
             # "npu", "npv",
             # "tauH_mass",
             "DeepMET_ResponseTune_px", "DeepMET_ResponseTune_py", "DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py",
             "recoGenTauH_mass",
         ],
         columns_to_add={
             "DeepMET_ResolutionTune_phi": (("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py"), (lambda a, b: np.arctan2(a, b))),
             "met_dphi": (("met_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
             "dmet_resp_px": (("DeepMET_ResponseTune_px", "DeepMET_ResponseTune_py", "DeepMET_ResolutionTune_phi"), (lambda a, b, c: b * np.cos(-c) - a * np.sin(-c))),
             "dmet_resp_py": (("DeepMET_ResponseTune_px", "DeepMET_ResponseTune_py", "DeepMET_ResolutionTune_phi"), (lambda a, b, c: a * np.cos(-c) + b * np.sin(-c))),
             "dmet_reso_px": (("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py", "DeepMET_ResolutionTune_phi"), (lambda a, b, c: b * np.cos(-c) - a * np.sin(-c))),
             "dmet_reso_py": (("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py", "DeepMET_ResolutionTune_phi"), (lambda a, b, c: a * np.cos(-c) + b * np.sin(-c))),
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
             # "dau1_mt": (("dau1_pz", "dau1_e"), (lambda z, e: np.sqrt(e**2-z**2))),
             # "dau2_mt": (("dau2_pz", "dau2_e"), (lambda z, e: np.sqrt(e**2-z**2))),
             # "mT_tau1": (("dau1_mt", "dau1_px", "dau1_py", "met_et"), (lambda e1, x1, y1, e2: np.sqrt((e1+e2)**2-(x1+e2)**2-(y1+0)**2))),
             # "mT_tau2": (("dau2_mt", "dau2_px", "dau2_py", "met_et"), (lambda e1, x1, y1, e2: np.sqrt((e1+e2)**2-(x1+e2)**2-(y1+0)**2))),
             # "mT_tautau": (("dau1_mt", "dau1_px", "dau1_py", "dau2_mt", "dau2_px", "dau2_py", "met_et"), (lambda e1, x1, y1, e2, x2, y2, e3: np.sqrt((e1+e2+e3)**2-(x1+x2+e3)**2-(y1+y2+0)**2))),
         },
         input_names=[
             # "met_et", "dmet_resp_px", "dmet_resp_py", "dmet_reso_px", "dmet_reso_py",
             "met_px", "met_py", "dmet_resp_px", "dmet_resp_py", "dmet_reso_px",
             "ditau_deltaphi", "ditau_deltaeta",
             "dau1_px", "dau1_py", "dau1_pz", "dau1_e", "dau1_dxy", "dau1_dz", "dau1_iso",
             "dau2_px", "dau2_py", "dau2_pz", "dau2_e", "dau2_dxy", "dau2_dz", "dau2_iso",
             "met_cov00", "met_cov01", "met_cov11",
             "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e", "bjet1_btag_deepFlavor", "bjet1_cID_deepFlavor", "bjet1_pnet_bb", "bjet1_pnet_cc", "bjet1_pnet_b", "bjet1_pnet_c", "bjet1_pnet_g", "bjet1_pnet_uds", "bjet1_pnet_pu", "bjet1_pnet_undef", "bjet1_HHbtag",
             "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e", "bjet2_btag_deepFlavor", "bjet2_cID_deepFlavor", "bjet2_pnet_bb", "bjet2_pnet_cc", "bjet2_pnet_b", "bjet2_pnet_c", "bjet2_pnet_g", "bjet2_pnet_uds", "bjet2_pnet_pu", "bjet2_pnet_undef", "bjet2_HHbtag",
             # "tauH_mass", "mT_tau1", "mT_tau2", "mT_tautau", "npu", "npv",
             # "met_px", "met_py",
         ],
         cat_input_names=[
             "pairType", "dau1_decayMode", "dau2_decayMode"
         ],
         target_names=[
             "genNu1_px", "genNu1_py", "genNu1_pz",  # "genNu1_e",
             "genNu2_px", "genNu2_py", "genNu2_pz",  # "genNu2_e",
         ],
         selections=[
             (("nbjetscand",), (lambda a: a > 1)),
             (("pairType",), (lambda a: a < 3)),
             # (("genLeptons_matched",), (lambda a: a == 1)),
             # (("genBQuarks_matched",),(lambda a: a == 1)),
             (("nleps",), (lambda a: a == 0)),
             (("isOS",), (lambda a: a == 1)),
             (("dau2_deepTauVsJet",), (lambda a: a >= 5)),
             (("pairType", "dau1_iso", "dau1_eleMVAiso", "dau1_deepTauVsJet"), (lambda a, b, c, d: (((a == 0) & (b < 0.15)) | ((a == 1) & (c == 1)) | ((a == 2) & (d >= 5))))),

         ],
         embedding_expected_inputs=[[0, 1, 2], [-1, 0, 1, 10, 11], [0, 1, 10, 11]],
         embedding_output_dim=5,
         units=((128,)*5, (128,) * 4),
         activation="elu",
         l2_norm=50.0,
         dropout_rate=0,
         batch_size=4096,
         train_valid_fraction=0.75,
         train_valid_seed=1,
         log_every=100,
         validate_every=1000,
         initial_learning_rate=0.0025,
         learning_rate_patience=2,
         max_learning_rate_reductions=5,
         early_stopping_patience=4,
         output_scaling=True,
         use_batch_composition=True,
         drop_quantile=0,
         gradient_clipping=False,
         classifier_weight=1.,
         mass_loss_weight=1./10000.,
         ):

    inputs_train = []
    inputs_valid = []

    cat_inputs_train = []
    cat_inputs_valid = []

    target_train = []
    target_valid = []

    classes_train = []
    classes_valid = []

    recoGenTauH_mass_train = []
    recoGenTauH_mass_valid = []

    mass_loss_input_indices = [input_names.index(x) for x in ["dau1_px", "dau1_py", "dau1_pz", "dau1_e", "dau2_px", "dau2_py", "dau2_pz", "dau2_e"]]
    mass_loss_inputs_train = []
    mass_loss_inputs_valid = []

    batch_weights = []

    nevents = []

    event_weights_train = []
    event_weights_valid = []

    input_means = []
    input_vars = []

    target_means = []
    target_stds = []

    for sample, (batch_weight, event_weight, target_classes) in samples.items():
        d, event_weights = load_sample(basepath, sample, event_weight, columns_to_read, selections)
        nev = len(event_weights)

        d = calc_new_columns(d, columns_to_add)

        inputs = d[input_names]
        cat_inputs = d[cat_input_names]
        targets = d[target_names]
        classes = [target_classes] * nev
        recoGenTauH_mass = d["recoGenTauH_mass"]

        inputs = inputs.astype([(name, np.float32) for name in inputs.dtype.names], copy=False).view(np.float32).reshape((-1, len(inputs.dtype)))
        cat_inputs = cat_inputs.astype([(name, np.float32) for name in cat_inputs.dtype.names], copy=False).view(np.float32).reshape((-1, len(cat_inputs.dtype)))
        targets = targets.astype([(name, np.float32) for name in targets.dtype.names], copy=False).view(np.float32).reshape((-1, len(targets.dtype)))

        mass_loss_inputs = inputs[:, mass_loss_input_indices]

        # input feature scaling
        input_means.append(np.mean(inputs, axis=0))
        input_vars.append(np.var(inputs, axis=0))

        # output scaling
        if output_scaling:
            target_means.append(np.mean(targets, axis=0))
            target_stds.append(np.std(targets, axis=0))

        train_mask = split_train_validation_mask(nev, fraction=train_valid_fraction, seed=train_valid_seed)

        nevents.append(nev)

        # remove top and bottom quantile per float input feature per sample from training events
        quantile_mask = [True] * len(np.where(train_mask)[0])
        if drop_quantile > 0:
            for i in range(inputs.shape[1]):
                quantile_mask = quantile_mask & ((inputs[train_mask][:, i] >= np.quantile(inputs[train_mask][:, i], drop_quantile)) &
                                                 (inputs[train_mask][:, i] <= np.quantile(inputs[train_mask][:, i], 1-drop_quantile)))

        print(f"Left for training: {np.where(quantile_mask)[0].shape[0]}")

        inputs_train.append(inputs[train_mask][quantile_mask])
        inputs_valid.append(inputs[~train_mask])

        cat_inputs_train.append(cat_inputs[train_mask][quantile_mask])
        cat_inputs_valid.append(cat_inputs[~train_mask])

        target_train.append(targets[train_mask][quantile_mask])
        target_valid.append(targets[~train_mask])

        classes_train.append(np.array(classes, dtype="float32")[train_mask][quantile_mask])
        classes_valid.append(np.array(classes, dtype="float32")[~train_mask])

        recoGenTauH_mass_train.append(recoGenTauH_mass[train_mask][quantile_mask])
        recoGenTauH_mass_valid.append(recoGenTauH_mass[~train_mask])

        mass_loss_inputs_train.append(mass_loss_inputs[train_mask][quantile_mask])
        mass_loss_inputs_valid.append(mass_loss_inputs[~train_mask])

        batch_weights.append(batch_weight)

        event_weights_train.append(event_weights[train_mask][quantile_mask][..., None])
        event_weights_valid.append(event_weights[~train_mask][..., None])

    if output_scaling:
        target_means = np.mean(target_means, axis=0)
        target_stds = np.mean(target_stds, axis=0)
        target_train = [(x - target_means) / target_stds for x in target_train]
        target_valid = [(x - target_means) / target_stds for x in target_valid]
    else:
        target_means = None
        target_stds = None

    input_means = np.mean(input_means, axis=0)
    input_vars = np.mean(input_vars, axis=0)

    if use_batch_composition:
        dataset_train = MultiDataset(zip(zip(inputs_train, cat_inputs_train, target_train, classes_train, event_weights_train,
                                     recoGenTauH_mass_train, mass_loss_inputs_train), batch_weights), batch_size, kind="train")
        dataset_valid = MultiDataset(zip(zip(inputs_valid, cat_inputs_valid, target_valid, classes_valid, event_weights_valid,
                                     recoGenTauH_mass_valid, mass_loss_inputs_valid), batch_weights), batch_size, kind="valid")
    else:
        inputs_train = np.concatenate(inputs_train, axis=0)
        inputs_valid = np.concatenate(inputs_valid, axis=0)
        cat_inputs_train = np.concatenate(cat_inputs_train, axis=0)
        cat_inputs_valid = np.concatenate(cat_inputs_valid, axis=0)
        target_train = np.concatenate(target_train, axis=0)
        target_valid = np.concatenate(target_valid, axis=0)
        classes_train = np.concatenate(classes_train, axis=0)
        classes_valid = np.concatenate(classes_valid, axis=0)
        event_weights_train = np.concatenate([event_weights_train[i]*batch_weights[i]/nevents[i] for i in range(len(batch_weights))], dtype="float32")
        event_weights_valid = np.concatenate([event_weights_valid[i]*batch_weights[i]/nevents[i] for i in range(len(batch_weights))], dtype="float32")
        avg_weight = np.mean(np.concatenate([event_weights_train, event_weights_valid]), dtype="float32")
        event_weights_train = event_weights_train/avg_weight
        event_weights_valid = event_weights_valid/avg_weight
        recoGenTauH_mass_train = np.concatenate(recoGenTauH_mass_train, axis=0)
        recoGenTauH_mass_valid = np.concatenate(recoGenTauH_mass_valid, axis=0)
        mass_loss_inputs_train = np.concatenate(mass_loss_inputs_train, axis=0)
        mass_loss_inputs_valid = np.concatenate(mass_loss_inputs_valid, axis=0)

        dataset_train = create_dataset(inputs_train, cat_inputs_train, target_train, classes_train, event_weights_train,
                                       recoGenTauH_mass_train, mass_loss_inputs_train, shuffle=True, repeat=-1, batch_size=batch_size, seed=None)
        dataset_valid = create_dataset(inputs_valid, cat_inputs_valid, target_valid, classes_valid, event_weights_valid,
                                       recoGenTauH_mass_valid, mass_loss_inputs_valid, shuffle=False, repeat=1, batch_size=batch_size, seed=None)
    nclasses = 0
    if classifier_weight > 0:
        nclasses = len(list(samples.items())[0][1][2])

    model, regularization_weights = create_model(len(input_names),
                                                 len(cat_input_names),
                                                 len(target_names),
                                                 nclasses,
                                                 embedding_expected_inputs,
                                                 embedding_output_dim,
                                                 input_means,
                                                 input_vars,
                                                 target_means=target_means,
                                                 target_stds=target_stds,
                                                 units=units,
                                                 activation=activation,
                                                 dropout_rate=dropout_rate)
    model.summary()

    loss_fns = create_losses(regularization_weights, l2_norm, classifier_weight, mass_loss_weight, mass_loss_input_indices)
    if target_means is not None and target_stds is not None:
        for name, loss_fn in loss_fns.items():
            if "mse" in name:
                loss_fn.prediction_index = 0
            if "mass" in name:
                loss_fn.prediction_index = 1
            if "ce" in name:
                loss_fn.prediction_index = -1

    optimizer, learning_rate = create_optimizer(initial_learning_rate)

    best_weights, _ = training_loop(dataset_train,
                                    dataset_valid,
                                    model,
                                    loss_fns,
                                    optimizer,
                                    learning_rate,
                                    log_every=log_every,
                                    validate_every=validate_every,
                                    tensorboard_dir=os.path.join(tensorboard_dir, model_name) if tensorboard_dir else None,
                                    early_stopping_patience=early_stopping_patience,
                                    learning_rate_patience=learning_rate_patience,
                                    max_learning_rate_reductions=max_learning_rate_reductions,
                                    gradient_clipping=gradient_clipping,
                                    )

    model.set_weights(best_weights)
    try:
        model.save(f"models/{model_name}")
    except:
        print("Couldn't save on afs. Trying /tmp")
        model.save(f"/tmp/{model_name}")


def create_dataset(inputs, cat_inputs, targets, event_weights, recoGenTauH_mass, mass_loss_inputs, shuffle=False, repeat=1, batch_size=1024, seed=None, **kwargs):
    nevents = inputs.shape[0]

    # create a tf dataset
    data = (inputs, cat_inputs, targets, event_weights, recoGenTauH_mass, mass_loss_inputs)

    ds = tf.data.Dataset.from_tensor_slices(data)

    # in the following, we amend the dataset object using methods
    # that return a new dataset object *without* copying the data

    # apply shuffeling
    if shuffle:
        ds = ds.shuffle(10 * nevents, reshuffle_each_iteration=True, seed=seed)

    # apply batching
    if batch_size < 1:
        batch_size = nevents
    ds = ds.batch(batch_size)

    # apply repetition, i.e. start iterating from the beginning when the
    # dataset is exhausted
    ds = ds.repeat(repeat)

    return ds


def create_model(input_shape, cat_input_shape, output_shape, nclasses, embedding_expected_inputs, embedding_output_dim, input_means, input_vars, target_means=None, target_stds=None, units=((128, 128, 128), (128, 128, 128)), activation="selu", dropout_rate=0.):
    activation_settings = {
        "elu": ("ELU", "he_uniform"),
        "relu": ("ReLU", "he_uniform"),
        "prelu": ("PReLU", "he_normal"),
        "selu": ("selu", "lecun_normal"),
        "tanh": ("tanh", "glorot_normal"),
        "softmax": ("softmax", "glorot_normal"),
    }

    if nclasses > 0:
        units2 = units[1]
        units = units[0]
    else:
        units = units[0] + units[1]
        units2 = []

    # track weights for later use
    weights = []

    with gpu:
        # input layers
        x1 = tf.keras.Input(input_shape)
        x2 = tf.keras.Input(cat_input_shape)

        norm_layer = tf.keras.layers.Normalization(mean=input_means, variance=input_vars)
        n = norm_layer(x1)

        # only add embedding layer if number of integer vars > 0
        if cat_input_shape > 0:
            custom_embedding_layer = CustomEmbeddingLayer(output_dim=embedding_output_dim, expected_inputs=embedding_expected_inputs)
            a = custom_embedding_layer(x2)
            a = tf.keras.layers.Concatenate()([n, a])
        else:
            a = n

        # add layers programatically
        for u in units:
            # build the layer
            dense_layer = tf.keras.layers.Dense(u, use_bias=True, kernel_initializer=activation_settings[activation][1])
            a = dense_layer(a)

            activation_layer = tf.keras.layers.Activation(activation_settings[activation][0])
            batchnorm_layer = tf.keras.layers.BatchNormalization(dtype="float32")

            if activation not in ["selu", "relu"]:
                a = batchnorm_layer(a)

            a = activation_layer(a)

            if activation == "relu":
                a = batchnorm_layer(a)

            # store the weight matrix for later use
            weights.append(dense_layer.kernel)

            # add random unit dropout
            if dropout_rate:
                if activation == "selu":
                    a = tf.keras.layers.AlphaDropout(dropout_rate)(a)
                else:
                    a = tf.keras.layers.Dropout(dropout_rate)(a)

        b = a
        c = a

        for u2 in units2:
            dense_layer1 = tf.keras.layers.Dense(u2, use_bias=True, kernel_initializer=activation_settings[activation][1])
            dense_layer2 = tf.keras.layers.Dense(u2, use_bias=True, kernel_initializer=activation_settings[activation][1])

            b = dense_layer1(b)
            c = dense_layer2(c)

            activation_layer1 = tf.keras.layers.Activation(activation_settings[activation][0])
            activation_layer2 = tf.keras.layers.Activation(activation_settings[activation][0])

            batchnorm_layer1 = tf.keras.layers.BatchNormalization(dtype="float32")
            batchnorm_layer2 = tf.keras.layers.BatchNormalization(dtype="float32")

            if activation not in ["selu", "relu"]:
                b = batchnorm_layer1(b)
                c = batchnorm_layer2(c)

            b = activation_layer1(b)
            c = activation_layer2(c)

            if activation == "relu":
                b = batchnorm_layer1(b)
                c = batchnorm_layer2(c)

            # store the weight matrix for later use
            weights.append(dense_layer1.kernel)
            weights.append(dense_layer2.kernel)

            # add random unit dropout
            if dropout_rate:
                if activation == "selu":
                    b = tf.keras.layers.AlphaDropout(dropout_rate)(b)
                    c = tf.keras.layers.AlphaDropout(dropout_rate)(c)
                else:
                    b = tf.keras.layers.Dropout(dropout_rate)(b)
                    c = tf.keras.layers.Dropout(dropout_rate)(c)

        a = b

        # add the output layer
        y1 = tf.keras.layers.Dense(output_shape, use_bias=True, kernel_initializer="he_uniform")(a)
        outputs = [y1]
        if target_means is not None and target_stds is not None:
            y2 = CustomOutputScalingLayer(target_means, target_stds)(y1)
            outputs.append(y2)

        if nclasses > 0:
            y3 = tf.keras.layers.Dense(nclasses, activation="softmax", use_bias=True, kernel_initializer="glorot_normal")(c)
            outputs.append(y3)

        # build the model
        model = tf.keras.Model(inputs=[x1, x2], outputs=outputs, name="htautau_regression")
    return model, weights


# define the losses
def create_losses(modelweights, l2_norm=10, classifier_weight=0., mass_loss_weight=0., mass_loss_input_indices=[]):
    n_modelweights = sum(functools.reduce(mul, w.shape) for w in modelweights)

    loss_dict = {}

    # MSE loss
    @tf.function
    def loss_mse_fn(**kwargs):
        labels = kwargs["labels"]
        predictions = kwargs["predictions"]
        event_weights = kwargs["event_weights"]
        with gpu:
            # compute the mse loss
            loss_mse = tf.reduce_mean(event_weights * (labels - predictions) ** 2.)
            return loss_mse

    loss_dict["mse"] = loss_mse_fn

    # l2 loss
    if l2_norm > 0:
        @tf.function
        def loss_l2_fn(**kwargs):
            with gpu:
                # accept labels and predictions although we don't need them
                # but this makes it easier to call all loss functions the same way
                loss_l2 = sum(tf.reduce_sum(w ** 2) for w in modelweights)

                return l2_norm / n_modelweights * loss_l2

        loss_dict["l2"] = loss_l2_fn

    # cross entropy
    if classifier_weight > 0.:
        @tf.function
        def loss_ce_fn(**kwargs):
            labels = kwargs["labels"]
            predictions = kwargs["predictions"]
            event_weights = kwargs["event_weights"]
            with gpu:
                # ensure proper prediction values before applying log's
                predictions = tf.clip_by_value(predictions, 1e-6, 1 - 1e-6)
                loss_ce = tf.reduce_mean(event_weights * -labels * tf.math.log(predictions))
                return classifier_weight * loss_ce

        loss_dict["ce"] = loss_ce_fn

    # mass loss
    if mass_loss_weight > 0.:
        @tf.function
        def loss_mass_fn(**kwargs):
            recoGenTauH_mass = kwargs["recoGenTauH_mass"]
            predictions = kwargs["predictions"]
            mass_loss_inputs = kwargs["mass_loss_inputs"]

            predicted_mass = (mass_loss_inputs[:, 3] + mass_loss_inputs[:, 7] + tf.sqrt(predictions[:, 0]**2+predictions[:, 1] **
                              2+predictions[:, 2]**2) + tf.sqrt(predictions[:, 3]**2+predictions[:, 4]**2+predictions[:, 5]**2))**2  # energy
            predicted_mass -= (mass_loss_inputs[:, 0] + mass_loss_inputs[:, 4] + predictions[:, 0] + predictions[:, 3])**2  # px
            predicted_mass -= (mass_loss_inputs[:, 1] + mass_loss_inputs[:, 5] + predictions[:, 1] + predictions[:, 4])**2  # py
            predicted_mass -= (mass_loss_inputs[:, 2] + mass_loss_inputs[:, 6] + predictions[:, 2] + predictions[:, 5])**2  # pz
            predicted_mass = tf.sqrt(predicted_mass)

            event_weights = kwargs["event_weights"]
            with gpu:
                # compute the mass loss
                loss_mass = tf.reduce_mean(event_weights * (recoGenTauH_mass - predicted_mass) ** 2.)
                return mass_loss_weight * loss_mass

        loss_dict["mass"] = loss_mass_fn

    # return a dict with all loss function components
    return loss_dict


def create_optimizer(initial_learning_rate):
    with gpu:
        learning_rate = tf.Variable(initial_learning_rate, dtype=tf.float32, trainable=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    return optimizer, learning_rate


def training_loop(
    dataset_train,
    dataset_valid,
    model,
    loss_fns,
    optimizer,
    learning_rate,
    log_every=10,
    validate_every=100,
    tensorboard_dir=None,
    early_stopping_patience=20,
    learning_rate_patience=10,
    max_learning_rate_reductions=5,
    gradient_clipping=False,
):
    early_stopping_counter = 0
    learning_rate_reduction_counter = 0
    message = ""
    losses_avg = defaultdict(list)
    loss_avg = []

    # store the best model, identified by the best validation accuracy
    best_weights = None

    # metrics to update during training
    metrics = dict(
        step=0, step_val=0,
        mse_valid_best=sys.maxsize,
        early_stopping_counter=0,
        learning_rate=learning_rate.numpy(),
        # initial_learning_rate=learning_rate.numpy(),
        start_validation_loss=None,
        total_validation_loss=None,
    )
    for name in loss_fns:
        for kind in ["train", "valid"]:
            metrics[f"loss_{name}_{kind}"] = 0

    if tensorboard_dir is not None:

        # helpers to add tensors and metrics to tensorboard for monitoring
        def tb_log_dir(kind):
            return tensorboard_dir and os.path.join(tensorboard_dir, kind)

        # tb_train_batch_add, tb_train_batch_flush = create_tensorboard_callbacks(tb_log_dir("train_batch"))
        tb_train_add, tb_train_flush = create_tensorboard_callbacks(tb_log_dir("train"))
        tb_valid_add, tb_valid_flush = create_tensorboard_callbacks(tb_log_dir("valid"))

    # helper to update metrics
    def update_metrics(kind, step, losses, total_loss):

        metrics["step"] = step
        for name, loss in losses.items():
            metrics[f"loss_{name}_{kind}"] = tf.reduce_mean(loss)

        metrics["early_stopping_counter"] = early_stopping_counter
        metrics["learning_rate"] = learning_rate.numpy()

        # validation specific
        if kind == "valid":
            metrics["step_val"] += 1
            metrics["mse_valid_best"] = min(metrics["mse_valid_best"], metrics[f"loss_mse_valid"])
            metrics["total_validation_loss"] = tf.reduce_mean(total_loss)
            if step == 0:
                metrics["start_validation_loss"] = metrics["total_validation_loss"]
            if tensorboard_dir is not None:
                tb_valid_add("scalar", "loss/total", metrics["total_validation_loss"], step=step)
                for key, l in losses.items():
                    tb_valid_add("scalar", "loss/" + key, tf.reduce_mean(l), step=step)
                tb_valid_flush()
            return metrics[f"loss_mse_valid"] == metrics["mse_valid_best"]
        else:
            if tensorboard_dir is not None:
                tb_train_add("scalar", "optimizer/learning_rate", learning_rate, step=step)
                tb_train_add("scalar", "loss/total", tf.reduce_mean(total_loss), step=step)
                for key, l in losses.items():
                    tb_train_add("scalar", "loss/" + key, tf.reduce_mean(l), step=step)
                for v in model.trainable_variables:
                    tb_train_add("histogram", "weight/{}".format(v.name), v, step=step)
                for v, g in zip(model.trainable_variables, gradients):
                    tb_train_add("histogram", "gradient/{}".format(v.name), g, step=step)
                tb_train_flush()

    # start the loop
    for step, (inputs, cat_inputs, targets, classes, event_weights, recoGenTauH_mass, mass_loss_inputs) in enumerate(dataset_train):

        # do a train step
        with tf.GradientTape() as tape:
            # get predictions
            if step == 0 and tensorboard_dir is not None:
                tb_train_add("trace_on", graph=True)

            predictions = model([inputs, cat_inputs], training=True)

            if step == 0 and tensorboard_dir is not None:
                tb_train_add("trace_export", "graph", step=step)
                tb_train_add("trace_off")

            # compute all losses and combine them into the total loss
            losses = {
                name: loss_fn(
                    labels=(classes if "ce" in name else targets),
                    predictions=(predictions[pred_i] if (pred_i := getattr(loss_fn, "prediction_index", None)) != None else predictions),
                    event_weights=event_weights,
                    recoGenTauH_mass=recoGenTauH_mass,
                    mass_loss_inputs=mass_loss_inputs,
                )
                for name, loss_fn in loss_fns.items()
            }

            loss = tf.add_n(list(losses.values()))

        # validation
        do_validate = step % validate_every == 0
        if do_validate:
            losses_valid_avg = defaultdict(list)
            loss_valid_avg = []
            for (inputs_valid, cat_inputs_valid, targets_valid, classes_valid, event_weights_valid, recoGenTauH_mass_valid, mass_loss_inputs_valid) in dataset_valid:
                predictions_valid = model([inputs_valid, cat_inputs_valid],  training=False)

                losses_valid = {
                    name: loss_fn(
                        labels=(classes_valid if "ce" in name else targets_valid),
                        predictions=(predictions_valid[pred_i] if (pred_i := getattr(loss_fn, "prediction_index", None)) != None else predictions_valid),
                        event_weights=event_weights_valid,
                        recoGenTauH_mass=recoGenTauH_mass_valid,
                        mass_loss_inputs=mass_loss_inputs_valid,
                    )
                    for name, loss_fn in loss_fns.items()
                }

                for name, loss_tensor in losses_valid.items():
                    losses_valid_avg[name].append(loss_tensor)
                loss_valid_avg.append(tf.add_n(list(losses_valid.values())))
            is_best = update_metrics("valid", step, losses_valid_avg, loss_valid_avg)

            # store the best model
            if is_best:
                best_weights = model.get_weights()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter == learning_rate_patience and learning_rate_reduction_counter < max_learning_rate_reductions:
                    learning_rate.assign(learning_rate / 2)
                    learning_rate_reduction_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    message = f"early stopping: validation loss did not improve within the last {early_stopping_patience} validation steps"
                    break

        # get and propagate gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        if gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        for name, loss_tensor in losses.items():
            losses_avg[name].append(loss_tensor)
        loss_avg.append(loss)
        # logging
        do_log = step % log_every == 0
        if do_log:
            # update_metrics("train_batch", step, losses, loss)
            update_metrics("train", step, losses_avg, loss_avg)
            losses_avg.clear()
            del loss_avg[:]

        update = [f"Step: {metrics['step']}, Validations: {metrics['step_val']}, Early stopping counter: {metrics['early_stopping_counter']}, Learning rate: {metrics['learning_rate']:.5f}"]
        for name in losses:
            if "mse" in name:
                update.append(f"Loss '{name}': {metrics[f'loss_{name}_train']:.4f} | {metrics[f'loss_{name}_valid']:.4f} | {metrics[f'mse_valid_best']:.4f}")
            else:
                update.append(f"Loss '{name}': {metrics[f'loss_{name}_train']:.4f} | {metrics[f'loss_{name}_valid']:.4f}")
        update = " --- ".join(update)
        print(update, end="\r")
    else:
        message = "dataset exhausted, stopping training"

    print(message)
    print("validation metrics of the best model:")
    print(f"MSE: {metrics['mse_valid_best']:.4f}")
    return best_weights, metrics


if __name__ == "__main__":
    main()
