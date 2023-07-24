#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import numpy as np
import sys
import os
import functools
import pickle
from operator import mul
from collections import defaultdict
from util import load_sample, phi_mpi_to_pi, calc_new_columns, create_tensorboard_callbacks, get_device
from custom_layers import CustomEmbeddingLayer, CustomOutputScalingLayer
from multi_dataset import MultiDataset

gpu = get_device(device="gpu", num_device=0)
tf.debugging.set_log_device_placement(True)


def main(model_name="reg_mass_class_para_l2n50_addCharge_incrMassLoss",
         basepath="/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v4_17Jul23",
         tensorboard_dir="/tmp/tensorboard",
         # tensorboard_dir=None,
         samples={
             # sample name: (batch fraction weight, event weight factor, [classes], spin, mass)
             "SKIM_GGHH_SM": (1./35, 1., [1, 0, 0], -1, -1.),
             # "SKIM_ggF_Radion_m250": (1./45, 1., [1, 0, 0], 0, 250.),
             # "SKIM_ggF_Radion_m260": (1./45, 1., [1, 0, 0], 0, 260.),
             # "SKIM_ggF_Radion_m270": (1./45, 1., [1, 0, 0], 0, 270.),
             # "SKIM_ggF_Radion_m280": (1./45, 1., [1, 0, 0], 0, 280.),
             "SKIM_ggF_Radion_m300": (1./35, 1., [1, 0, 0], 0, 300.),
             # "SKIM_ggF_Radion_m320": (1./45, 1., [1, 0, 0], 0, 320.),
             "SKIM_ggF_Radion_m350": (1./35, 1., [1, 0, 0], 0, 350.),
             "SKIM_ggF_Radion_m400": (1./35, 1., [1, 0, 0], 0, 400.),
             "SKIM_ggF_Radion_m450": (1./35, 1., [1, 0, 0], 0, 450.),
             "SKIM_ggF_Radion_m500": (1./35, 1., [1, 0, 0], 0, 500.),
             "SKIM_ggF_Radion_m550": (1./35, 1., [1, 0, 0], 0, 550.),
             "SKIM_ggF_Radion_m600": (1./35, 1., [1, 0, 0], 0, 600.),
             "SKIM_ggF_Radion_m650": (1./35, 1., [1, 0, 0], 0, 650.),
             "SKIM_ggF_Radion_m700": (1./35, 1., [1, 0, 0], 0, 700.),
             "SKIM_ggF_Radion_m750": (1./35, 1., [1, 0, 0], 0, 750.),
             "SKIM_ggF_Radion_m800": (1./35, 1., [1, 0, 0], 0, 800.),
             "SKIM_ggF_Radion_m850": (1./35, 1., [1, 0, 0], 0, 850.),
             "SKIM_ggF_Radion_m900": (1./35, 1., [1, 0, 0], 0, 900.),
             "SKIM_ggF_Radion_m1000": (1./35, 1., [1, 0, 0], 0, 1000.),
             "SKIM_ggF_Radion_m1250": (1./35, 1., [1, 0, 0], 0, 1250.),
             "SKIM_ggF_Radion_m1500": (1./35, 1., [1, 0, 0], 0, 1500.),
             "SKIM_ggF_Radion_m1750": (1./35, 1., [1, 0, 0], 0, 1750.),
             # "SKIM_ggF_BulkGraviton_m250": (1./45, 1., [1, 0, 0], 2, 250.),
             # "SKIM_ggF_BulkGraviton_m260": (1./45, 1., [1, 0, 0], 2, 260.),
             # "SKIM_ggF_BulkGraviton_m270": (1./45, 1., [1, 0, 0], 2, 270.),
             # "SKIM_ggF_BulkGraviton_m280": (1./45, 1., [1, 0, 0], 2, 280.),
             "SKIM_ggF_BulkGraviton_m300": (1./35, 1., [1, 0, 0], 2, 300.),
             # "SKIM_ggF_BulkGraviton_m320": (1./45, 1., [1, 0, 0], 2, 320.),
             "SKIM_ggF_BulkGraviton_m350": (1./35, 1., [1, 0, 0], 2, 350.),
             "SKIM_ggF_BulkGraviton_m400": (1./35, 1., [1, 0, 0], 2, 400.),
             "SKIM_ggF_BulkGraviton_m450": (1./35, 1., [1, 0, 0], 2, 450.),
             "SKIM_ggF_BulkGraviton_m500": (1./35, 1., [1, 0, 0], 2, 500.),
             "SKIM_ggF_BulkGraviton_m550": (1./35, 1., [1, 0, 0], 2, 550.),
             "SKIM_ggF_BulkGraviton_m600": (1./35, 1., [1, 0, 0], 2, 600.),
             "SKIM_ggF_BulkGraviton_m650": (1./35, 1., [1, 0, 0], 2, 650.),
             "SKIM_ggF_BulkGraviton_m700": (1./35, 1., [1, 0, 0], 2, 700.),
             "SKIM_ggF_BulkGraviton_m750": (1./35, 1., [1, 0, 0], 2, 750.),
             "SKIM_ggF_BulkGraviton_m800": (1./35, 1., [1, 0, 0], 2, 800.),
             "SKIM_ggF_BulkGraviton_m850": (1./35, 1., [1, 0, 0], 2, 850.),
             "SKIM_ggF_BulkGraviton_m900": (1./35, 1., [1, 0, 0], 2, 900.),
             "SKIM_ggF_BulkGraviton_m1000": (1./35, 1., [1, 0, 0], 2, 1000.),
             "SKIM_ggF_BulkGraviton_m1250": (1./35, 1., [1, 0, 0], 2, 1250.),
             "SKIM_ggF_BulkGraviton_m1500": (1./35, 1., [1, 0, 0], 2, 1500.),
             "SKIM_ggF_BulkGraviton_m1750": (1./35, 1., [1, 0, 0], 2, 1750.),
             "SKIM_DY_amc_incl": (1., 1., [0, 1, 0], -1, -1.),
             "SKIM_TT_fullyLep": (1., 1., [0, 0, 1], -1, -1.),
             # "SKIM_TT_semiLep": (1., 1., [0, 0, 1], -1, -1.),
             # "SKIM_GluGluHToTauTau": (1., 1., [0, 0, 0, 0, 1, 0], -1, -1.),
             # "SKIM_ttHToTauTau": (1., 1., [0, 0, 0, 1], -1, -1.),
         },
         columns_to_read=[  # variables to read from the input files
             "dau1_pt", "dau1_eta", "dau1_phi", "dau1_e", "dau1_dxy", "dau1_dz", "dau1_iso", "dau1_charge",
             "dau2_pt", "dau2_eta", "dau2_phi", "dau2_e", "dau2_dxy", "dau2_dz", "dau2_iso", "dau2_charge",
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
             "EventNumber",
         ],
         columns_to_add={  # new variables to calculate from the existing ones
             "DeepMET_ResolutionTune_phi": (("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py"), (lambda x, y: np.arctan2(y, x))),
             "met_dphi": (("met_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
             "dmet_resp_px": (("DeepMET_ResponseTune_px", "DeepMET_ResponseTune_py", "DeepMET_ResolutionTune_phi"), (lambda x, y, p: np.cos(-p)*x - np.sin(-p)*y)),
             "dmet_resp_py": (("DeepMET_ResponseTune_px", "DeepMET_ResponseTune_py", "DeepMET_ResolutionTune_phi"), (lambda x, y, p: np.sin(-p)*x + np.cos(-p)*y)),
             "dmet_reso_px": (("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py", "DeepMET_ResolutionTune_phi"), (lambda x, y, p: np.cos(-p)*x - np.sin(-p)*y)),
             "dmet_reso_py": (("DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py", "DeepMET_ResolutionTune_phi"), (lambda x, y, p: np.sin(-p)*x + np.cos(-p)*y)),
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
         },
         input_names=[  # continuous input features to the network
             "met_px", "met_py", "dmet_resp_px", "dmet_resp_py", "dmet_reso_px",
             "ditau_deltaphi", "ditau_deltaeta",
             "dau1_px", "dau1_py", "dau1_pz", "dau1_e", "dau1_dxy", "dau1_dz", "dau1_iso",
             "dau2_px", "dau2_py", "dau2_pz", "dau2_e", "dau2_dxy", "dau2_dz", "dau2_iso",
             "met_cov00", "met_cov01", "met_cov11",
             "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e", "bjet1_btag_deepFlavor", "bjet1_cID_deepFlavor", "bjet1_pnet_bb", "bjet1_pnet_cc", "bjet1_pnet_b", "bjet1_pnet_c", "bjet1_pnet_g", "bjet1_pnet_uds", "bjet1_pnet_pu", "bjet1_pnet_undef", "bjet1_HHbtag",
             "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e", "bjet2_btag_deepFlavor", "bjet2_cID_deepFlavor", "bjet2_pnet_bb", "bjet2_pnet_cc", "bjet2_pnet_b", "bjet2_pnet_c", "bjet2_pnet_g", "bjet2_pnet_uds", "bjet2_pnet_pu", "bjet2_pnet_undef", "bjet2_HHbtag",
         ],

         cat_input_names=[  # categorical input features for the network
             "pairType", "dau1_decayMode", "dau2_decayMode", "dau1_charge", "dau2_charge"
         ],
         target_names=[  # targets for the regression, mse loss will be calculated for these
             "genNu1_px", "genNu1_py", "genNu1_pz",  # "genNu1_e",
             "genNu2_px", "genNu2_py", "genNu2_pz",  # "genNu2_e",
         ],
         selections=[  # selections to apply before training
             (("nbjetscand",), (lambda a: a > 1)),
             (("pairType",), (lambda a: a < 3)),
             (("nleps",), (lambda a: a == 0)),
             (("isOS",), (lambda a: a == 1)),
             (("dau2_deepTauVsJet",), (lambda a: a >= 5)),
             (("pairType", "dau1_iso", "dau1_eleMVAiso", "dau1_deepTauVsJet"), (lambda a, b, c, d: (
                 ((a == 0) & (b < 0.15)) | ((a == 1) & (c == 1)) | ((a == 2) & (d >= 5))))),

         ],
         embedding_expected_inputs=[  # possible values for the categorical features
             [0, 1, 2],  # pairType
             [-1, 0, 1, 10, 11],  # dau1_decayMode, -1 for e/mu
             [0, 1, 10, 11],  # dau2_decayMode
             [-1, 1],  # dau1_charge
             [-1, 1],  # dau2_charge
         ],
         embedding_output_dim=5,  # dimension of the embedding layer output will be embedding_output_dim x N_categorical_features
         units=((128,)*5, (128,) * 4),  # number of layers and units, second entry determines the extra heads (if applicable, otherwise "concatenate")
         activation="elu",  # activation function after each hidden layer
         l2_norm=50.0,  # scale fot the l2 loss term (which is already normalized to the number of weights)
         dropout_rate=0,
         batch_size=4096,
         train_valid_eventnumber_modulo=4,  # divide events by this based on EventNumber
         train_valid_eventnumber_rest=0,  # assign event to validation dataset if the rest is this
         log_every=100,  # how frequently the terminal and tensorboard are updated
         validate_every=1000,  # how frequently to calulcate the validation loss
         initial_learning_rate=0.0025,
         learning_rate_patience=2,  # half the learning rate if the validation MSE loss hasn't improved in this many validation steps
         max_learning_rate_reductions=5,
         early_stopping_patience=4,  # stop training if the validation MSE loss hasn't improved since this many validation steps
         output_scaling=True,  # scale regression targets (and therefore also output) to have mean=0 and width=1
         use_batch_composition=True,  # control  sample importance via the composition of the batch instead of by weights
         drop_quantile=0,  # drop this percantage of outliers per input variable
         gradient_clipping=False,  # prevent gradients from becoming very large
         classifier_weight=1.,  # add classification head if non-zero; scale for the cross-entropy loss-term
         mass_loss_weight=1./10000.,  # scale for the loss-term based on the mass calculated from the reco taus and the predicted/generated neutrinos
         parameterize_spin=True,  # add the generator spin for the signal samples as categorical input -> network parameterized in spin
         parameterize_mass=True,  # add the generator mass for the signal samples as continuous input -> network parameterized in mass
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

    mass_loss_input_indices = [input_names.index(x) for x in [
        "dau1_px", "dau1_py", "dau1_pz", "dau1_e", "dau2_px", "dau2_py", "dau2_pz", "dau2_e"]]
    mass_loss_inputs_train = []
    mass_loss_inputs_valid = []

    batch_weights = []

    nevents = []

    event_weights_train = []
    event_weights_valid = []

    target_means = []
    target_stds = []

    spins = set()
    masses = set()

    for sample, (batch_weight, event_weight, target_classes, spin, mass) in samples.items():
        d, event_weights = load_sample(
            basepath, sample, event_weight, columns_to_read, selections)
        nev = len(event_weights)

        d = calc_new_columns(d, columns_to_add)

        inputs = d[input_names]
        cat_inputs = d[cat_input_names]
        targets = d[target_names]
        classes = [target_classes] * nev
        recoGenTauH_mass = d["recoGenTauH_mass"]

        inputs = inputs.astype([(name, np.float32) for name in inputs.dtype.names], copy=False).view(
            np.float32).reshape((-1, len(inputs.dtype)))

        cat_inputs = cat_inputs.astype([(name, np.float32) for name in cat_inputs.dtype.names], copy=False).view(
            np.float32).reshape((-1, len(cat_inputs.dtype)))

        targets = targets.astype([(name, np.float32) for name in targets.dtype.names], copy=False).view(
            np.float32).reshape((-1, len(targets.dtype)))

        mass_loss_inputs = inputs[:, mass_loss_input_indices]

        if parameterize_spin:
            if spin > -1:
                spins.add(np.float32(spin))
            cat_inputs = np.append(cat_inputs, [[np.float32(spin)]]*nev, axis=1)

        if parameterize_mass:
            if mass > -1:
                masses.add(mass)
            inputs = np.append(inputs, [[np.float32(mass)]]*nev, axis=1)

        # output scaling
        if output_scaling:
            target_means.append(np.mean(targets, axis=0))
            target_stds.append(np.std(targets, axis=0))

        train_mask = (d["EventNumber"] % train_valid_eventnumber_modulo) != train_valid_eventnumber_rest

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

        classes_train.append(np.array(classes, dtype="float32")[
                             train_mask][quantile_mask])
        classes_valid.append(np.array(classes, dtype="float32")[~train_mask])

        recoGenTauH_mass_train.append(
            recoGenTauH_mass[train_mask][quantile_mask])
        recoGenTauH_mass_valid.append(recoGenTauH_mass[~train_mask])

        mass_loss_inputs_train.append(
            mass_loss_inputs[train_mask][quantile_mask])
        mass_loss_inputs_valid.append(mass_loss_inputs[~train_mask])

        batch_weights.append(batch_weight)

        event_weights_train.append(
            event_weights[train_mask][quantile_mask][..., None])
        event_weights_valid.append(event_weights[~train_mask][..., None])

    spins = list(spins)
    masses = list(masses)

    if output_scaling:
        target_means = np.mean(target_means, axis=0)
        target_stds = np.mean(target_stds, axis=0)
        target_train = [(x - target_means) / target_stds for x in target_train]
        target_valid = [(x - target_means) / target_stds for x in target_valid]
    else:
        target_means = None
        target_stds = None

    input_means = np.sum(np.concatenate([i * b/len(i) for i, b in zip(inputs_train, batch_weights)]), axis=0)/np.sum([b for b in batch_weights], axis=0)
    input_vars = np.sum(np.concatenate([i**2 * b/len(i) for i, b in zip(inputs_train, batch_weights)]), axis=0)/np.sum([b for b in batch_weights], axis=0) - input_means**2

    if parameterize_spin:
        cat_input_names.append("spin")
        embedding_expected_inputs += [spins]

    if parameterize_mass:
        input_names.append("mass")
        input_means[-1] = np.mean(masses)
        input_vars[-1] = np.var(masses)

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
        event_weights_train = np.concatenate(
            [event_weights_train[i]*batch_weights[i]/nevents[i] for i in range(len(batch_weights))], dtype="float32")
        event_weights_valid = np.concatenate(
            [event_weights_valid[i]*batch_weights[i]/nevents[i] for i in range(len(batch_weights))], dtype="float32")
        avg_weight = np.mean(np.concatenate(
            [event_weights_train, event_weights_valid]), dtype="float32")
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

    loss_fns = create_losses(regularization_weights, l2_norm,
                             classifier_weight, mass_loss_weight)
    if target_means is not None and target_stds is not None:
        for name, loss_fn in loss_fns.items():
            if "mse" in name:
                loss_fn.prediction_index = 0
            if "mass" in name:
                loss_fn.prediction_index = 1
            if "ce" in name:
                loss_fn.prediction_index = -1

    optimizer, learning_rate = create_optimizer(initial_learning_rate)

    best_weights, training_steps, _ = training_loop(dataset_train,
                                                    dataset_valid,
                                                    model,
                                                    loss_fns,
                                                    optimizer,
                                                    learning_rate,
                                                    log_every=log_every,
                                                    validate_every=validate_every,
                                                    tensorboard_dir=os.path.join(
                                                        tensorboard_dir, model_name) if tensorboard_dir else None,
                                                    early_stopping_patience=early_stopping_patience,
                                                    learning_rate_patience=learning_rate_patience,
                                                    max_learning_rate_reductions=max_learning_rate_reductions,
                                                    gradient_clipping=gradient_clipping,
                                                    spins=spins,
                                                    masses=masses,
                                                    )

    model.set_weights(best_weights)
    modelweights = {}
    modelweights["training_steps"] = training_steps
    modelweights["rotate_phi"] = ["dmet_reso", (["met"] if "met_px" in input_names else []) + (["dmet_resp"] if "dmet_resp_px" in input_names else []) + (["dmet_reso"] if "dmet_reso_px" in input_names else []) + ["dau1", "dau2", "bjet1", "bjet2"]]
    modelweights["cont_features"] = input_names.copy()
    modelweights["cont_features"].insert(input_names.index(modelweights["rotate_phi"][0] + "_px") + 1, modelweights["rotate_phi"][0] + "_py")
    modelweights["cat_features"] = cat_input_names.copy()
    modelweights["embedding_choices"] = embedding_expected_inputs

    for layer in model.layers:
        layer_name = layer.name
        if layer_name is "norm":
            modelweights["input_mean"] = np.array(layer.mean)
            modelweights["input_variance"] = np.array(layer.variance)
        if layer_name is "embedding":
            modelweights["embedding_weight"] = np.array(layer.weights)
        head = layer_name.split("_")[0]
        if head in ["common", "regression", "classification"]:
            layer_type = str(type(layer))
            if "Dense" in layer_type:
                modelweights[f"{layer_name}_weight"] = np.array(layer.kernel)
                modelweights[f"{layer_name}_bias"] = np.array(layer.bias)
            if "BatchNormalization" in layer_type:
                modelweights[f"{layer_name}_mean"] = np.array(layer.moving_mean)
                modelweights[f"{layer_name}_variance"] = np.array(layer.moving_variance)
                modelweights[f"{layer_name}_beta"] = np.array(layer.beta)
                modelweights[f"{layer_name}_gamma"] = np.array(layer.gamma)
                modelweights[f"{layer_name}_epsilon"] = np.array(layer.epsilon)
        if layer_name is "regression_output_hep":
            modelweights["regression_output_mean"] = np.array(layer.get_config()["target_means"])
            modelweights["regression_output_std"] = np.array(layer.get_config()["target_stds"])

    htt_mass_sum = []
    htt_mass_sum2 = []
    htt_pt_sum = []
    htt_pt_sum2 = []
    htt_eta_sum = []
    htt_eta_sum2 = []
    htt_gamma_sum = []
    htt_gamma_sum2 = []
    hh_mass_sum = []
    hh_mass_sum2 = []

    if use_batch_composition:
        for idx, cont_input in enumerate(inputs_train):
            cat_input = cat_inputs_train[idx]
            batch_weight = batch_weights[idx]
            if masses:
                para_mass = tf.transpose(tf.where(cont_input[:, -1] == -1, tf.gather(masses, tf.random.categorical(tf.math.log([[1.]*len(masses)]), len(cont_input))), cont_input[:, -1]))
                cont_input = tf.concat([cont_input[:, 0:-1], para_mass], axis=1)
            if spins:
                para_spin = tf.transpose(tf.where(cat_input[:, -1] == -1, tf.gather(spins, tf.random.categorical(tf.math.log([[1.]*len(spins)]), len(cat_input))), cat_input[:, -1]))
                cat_input = tf.concat([cat_input[:, 0:-1], para_spin], axis=1)
            prediction = model.predict([cont_input, cat_input])
            nus = prediction[1]
            nu1 = np.concatenate([np.expand_dims(np.sqrt(np.sum(nus[:, [0, 1, 2]]**2, axis=1)), axis=1), nus[:, [0, 1, 2]]], axis=1)
            nu2 = np.concatenate([np.expand_dims(np.sqrt(np.sum(nus[:, [3, 4, 5]]**2, axis=1)), axis=1), nus[:, [3, 4, 5]]], axis=1)
            cont_input = np.array(cont_input)
            dau1 = cont_input[:, [input_names.index("dau1_e"), input_names.index("dau1_px"), input_names.index("dau1_py"), input_names.index("dau1_pz")]]
            dau2 = cont_input[:, [input_names.index("dau2_e"), input_names.index("dau2_px"), input_names.index("dau2_py"), input_names.index("dau2_pz")]]
            tau1 = dau1 + nu1
            tau2 = dau2 + nu2
            htt = tau1 + tau2
            htt_mass2 = htt[:, 0]**2 - htt[:, 1]**2 - htt[:, 2]**2 - htt[:, 3]**2
            htt_mass_sum.append(np.mean(np.sqrt(htt_mass2))*batch_weight)
            htt_mass_sum2.append(np.mean(htt_mass2, axis=0)*batch_weight)
            htt_pt2 = htt[:, 1]**2 + htt[:, 2]**2
            htt_pt_sum.append(np.mean(np.sqrt(htt_pt2))*batch_weight)
            htt_pt_sum2.append(np.mean(htt_pt2, axis=0)*batch_weight)
            htt_eta = np.arcsinh(htt[:, 3]/np.sqrt(htt_pt2))
            htt_eta_sum.append(np.mean(htt_eta)*batch_weight)
            htt_eta_sum2.append(np.mean(htt_eta**2)*batch_weight)
            htt_gamma = htt[:, 0]/np.sqrt(htt_mass2)
            htt_gamma_sum.append(np.mean(htt_gamma)*batch_weight)
            htt_gamma_sum2.append(np.mean(htt_gamma**2)*batch_weight)
            bjet1 = cont_input[:, [input_names.index("bjet1_e"), input_names.index("bjet1_px"), input_names.index("bjet1_py"), input_names.index("bjet1_pz")]]
            bjet2 = cont_input[:, [input_names.index("bjet2_e"), input_names.index("bjet2_px"), input_names.index("bjet2_py"), input_names.index("bjet2_pz")]]
            hbb = bjet1+bjet2
            hh = htt + hbb
            hh_mass2 = hh[:, 0]**2 - hh[:, 1]**2 - hh[:, 2]**2 - hh[:, 3]**2
            hh_mass_sum.append(np.mean(np.sqrt(hh_mass2))*batch_weight)
            hh_mass_sum2.append(np.mean(hh_mass2, axis=0)*batch_weight)

        for idx, cont_input in enumerate(inputs_valid):
            cat_input = cat_inputs_valid[idx]
            batch_weight = batch_weights[idx]
            if masses:
                para_mass = tf.transpose(tf.where(cont_input[:, -1] == -1, tf.gather(masses, tf.random.categorical(tf.math.log([[1.]*len(masses)]), len(cont_input))), cont_input[:, -1]))
                cont_input = tf.concat([cont_input[:, 0:-1], para_mass], axis=1)
            if spins:
                para_spin = tf.transpose(tf.where(cat_input[:, -1] == -1, tf.gather(spins, tf.random.categorical(tf.math.log([[1.]*len(spins)]), len(cat_input))), cat_input[:, -1]))
                cat_input = tf.concat([cat_input[:, 0:-1], para_spin], axis=1)
            prediction = model.predict([cont_input, cat_input])
    # else:
        # TO BE IMPLEMENTED

    htt_mass_mean = np.sum(htt_mass_sum)/np.sum(batch_weights)
    htt_mass_std = np.sqrt(np.sum(htt_mass_sum2, axis=0)/np.sum(batch_weights) - htt_mass_mean**2)

    htt_pt_mean = np.sum(htt_pt_sum)/np.sum(batch_weights)
    htt_pt_std = np.sqrt(np.sum(htt_pt_sum2, axis=0)/np.sum(batch_weights) - htt_pt_mean**2)

    htt_eta_mean = np.sum(htt_eta_sum)/np.sum(batch_weights)
    htt_eta_std = np.sqrt(np.sum(htt_eta_sum2, axis=0)/np.sum(batch_weights) - htt_eta_mean**2)

    htt_gamma_mean = np.sum(htt_gamma_sum)/np.sum(batch_weights)
    htt_gamma_std = np.sqrt(np.sum(htt_gamma_sum2, axis=0)/np.sum(batch_weights) - htt_gamma_mean**2)
    hh_mass_mean = np.sum(hh_mass_sum)/np.sum(batch_weights)
    hh_mass_std = np.sqrt(np.sum(hh_mass_sum2, axis=0)/np.sum(batch_weights) - hh_mass_mean**2)

    modelweights["custom_outputs"] = {
        "htt_mass": [htt_mass_mean, htt_mass_std],
        "htt_pt": [htt_pt_mean, htt_pt_std],
        "htt_eta": [htt_eta_mean, htt_eta_std],
        "htt_gamma": [htt_gamma_mean, htt_gamma_std],
        "htt_cos_phi": [0.0, 1.0],
        "hh_mass": [hh_mass_mean, hh_mass_std],
        "hh_cos_phi": [0.0, 1.0],
    }

    try:
        model.save(f"models/{model_name}")
        with open(f"models/{model_name}/modelfile.pkl", 'wb') as handle:
            pickle.dump(modelweights, handle, protocol=4)
    except:
        print("Couldn't save on afs. Trying /tmp")
        model.save(f"/tmp/{model_name}")
        with open(f"/tmp/{model_name}/modelfile.pkl", 'wb') as handle:
            pickle.dump(modelweights, handle, protocol=4)


def create_dataset(inputs, cat_inputs, targets, event_weights, recoGenTauH_mass, mass_loss_inputs, shuffle=False, repeat=1, batch_size=1024, seed=None, **kwargs):
    nevents = inputs.shape[0]

    # create a tf dataset
    data = (inputs, cat_inputs, targets, event_weights,
            recoGenTauH_mass, mass_loss_inputs)

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
        x1 = tf.keras.Input(input_shape, name="cont_input")
        x2 = tf.keras.Input(cat_input_shape, name="cat_input")

        norm_layer = tf.keras.layers.Normalization(
            mean=input_means, variance=input_vars, name="norm")
        n = norm_layer(x1)

        # only add embedding layer if number of integer vars > 0
        if cat_input_shape > 0:
            custom_embedding_layer = CustomEmbeddingLayer(
                output_dim=embedding_output_dim, expected_inputs=embedding_expected_inputs, name="embedding")
            a = custom_embedding_layer(x2)
            a = tf.keras.layers.Concatenate(name="concat")([n, a])
        else:
            a = n

        # add layers programatically
        for idx, u in enumerate(units):
            # build the layer
            dense_layer = tf.keras.layers.Dense(
                u, use_bias=True, kernel_initializer=activation_settings[activation][1], name=f"common_{idx+1}")
            a = dense_layer(a)

            activation_layer = tf.keras.layers.Activation(
                activation_settings[activation][0], name=f"common_{idx+1}_act")
            batchnorm_layer = tf.keras.layers.BatchNormalization(
                dtype="float32", name=f"common_{idx+1}_bn")

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
                    a = tf.keras.layers.AlphaDropout(dropout_rate, name=f"common_{idx+1}_do")(a)
                else:
                    a = tf.keras.layers.Dropout(dropout_rate, name=f"common_{idx+1}_do")(a)

        b = a
        c = a

        for idx2, u2 in enumerate(units2):
            dense_layer1 = tf.keras.layers.Dense(
                u2, use_bias=True, kernel_initializer=activation_settings[activation][1], name=f"regression_{idx2+1}")
            dense_layer2 = tf.keras.layers.Dense(
                u2, use_bias=True, kernel_initializer=activation_settings[activation][1], name=f"classification_{idx2+1}")

            b = dense_layer1(b)
            c = dense_layer2(c)

            activation_layer1 = tf.keras.layers.Activation(
                activation_settings[activation][0], name=f"regression_{idx2+1}_act")
            activation_layer2 = tf.keras.layers.Activation(
                activation_settings[activation][0], name=f"classification_{idx2+1}_act")

            batchnorm_layer1 = tf.keras.layers.BatchNormalization(
                dtype="float32", name=f"regression_{idx2+1}_bn")
            batchnorm_layer2 = tf.keras.layers.BatchNormalization(
                dtype="float32", name=f"classification_{idx2+1}_bn")

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
                    b = tf.keras.layers.AlphaDropout(dropout_rate, name=f"regression_{idx2+1}_do")(b)
                    c = tf.keras.layers.AlphaDropout(dropout_rate, name=f"classification_{idx2+1}_do")(c)
                else:
                    b = tf.keras.layers.Dropout(dropout_rate, name=f"regression_{idx2+1}_do")(b)
                    c = tf.keras.layers.Dropout(dropout_rate, name=f"classification_{idx2+1}_do")(c)

        a = b

        # add the output layer
        y1 = tf.keras.layers.Dense(
            output_shape, use_bias=True, kernel_initializer="he_uniform", name="regression_output")(a)
        outputs = [y1]
        if target_means is not None and target_stds is not None:
            y2 = CustomOutputScalingLayer(target_means, target_stds, name="regression_output_hep")(y1)
            outputs.append(y2)

        if nclasses > 0:
            y3 = tf.keras.layers.Dense(
                nclasses, activation="softmax", use_bias=True, kernel_initializer="glorot_normal", name="classification_output")(c)
            outputs.append(y3)

        # build the model
        model = tf.keras.Model(
            inputs=[x1, x2], outputs=outputs, name="htautau_regression")
    return model, weights


# define the losses
def create_losses(modelweights, l2_norm=10, classifier_weight=0., mass_loss_weight=0.):
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
            loss_mse = tf.reduce_mean(
                event_weights * (labels - predictions) ** 2.)
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
                loss_ce = tf.reduce_mean(
                    event_weights * -labels * tf.math.log(predictions))
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
            predicted_mass -= (mass_loss_inputs[:, 0] + mass_loss_inputs[:,
                               4] + predictions[:, 0] + predictions[:, 3])**2  # px
            predicted_mass -= (mass_loss_inputs[:, 1] + mass_loss_inputs[:,
                               5] + predictions[:, 1] + predictions[:, 4])**2  # py
            predicted_mass -= (mass_loss_inputs[:, 2] + mass_loss_inputs[:,
                               6] + predictions[:, 2] + predictions[:, 5])**2  # pz
            predicted_mass = tf.sqrt(predicted_mass)

            event_weights = kwargs["event_weights"]
            with gpu:
                # compute the mass loss
                loss_mass = tf.reduce_mean(
                    event_weights * (recoGenTauH_mass - predicted_mass) ** 2.)
                return mass_loss_weight * loss_mass

        loss_dict["mass"] = loss_mass_fn

    # return a dict with all loss function components
    return loss_dict


def create_optimizer(initial_learning_rate):
    with gpu:
        learning_rate = tf.Variable(
            initial_learning_rate, dtype=tf.float32, trainable=False)
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
    spins=[],
    masses=[],
):
    early_stopping_counter = 0
    learning_rate_reduction_counter = 0
    message = ""
    losses_avg = defaultdict(list)
    loss_avg = []

    # store the best model weights, identified by the best validation mse loss
    best_weights = None
    best_weights_steps = 0

    # metrics to update during training
    metrics = dict(
        step=0, step_val=0,
        loss_sum_train=sys.maxsize,
        loss_sum_valid=sys.maxsize,
        loss_sum_valid_best=sys.maxsize,
        early_stopping_counter=0,
        learning_rate=learning_rate.numpy(),
        # initial_learning_rate=learning_rate.numpy(),
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
        tb_train_add, tb_train_flush = create_tensorboard_callbacks(
            tb_log_dir("train"))
        tb_valid_add, tb_valid_flush = create_tensorboard_callbacks(
            tb_log_dir("valid"))

    # helper to update metrics
    def update_metrics(kind, step, losses, total_loss):

        metrics["step"] = step
        metrics[f"loss_sum_{kind}"] = 0
        for name, loss in losses.items():
            tmp_loss = tf.reduce_mean(loss)
            metrics[f"loss_{name}_{kind}"] = tmp_loss
            metrics[f"loss_sum_{kind}"] += tmp_loss

        metrics["early_stopping_counter"] = early_stopping_counter
        metrics["learning_rate"] = learning_rate.numpy()

        # validation specific
        if kind == "valid":
            metrics["step_val"] += 1
            metrics["loss_sum_valid_best"] = min(
                metrics["loss_sum_valid_best"], metrics["loss_sum_valid"])
            metrics["total_validation_loss"] = tf.reduce_mean(total_loss)
            if tensorboard_dir is not None:
                tb_valid_add("scalar", "loss/total",
                             metrics["total_validation_loss"], step=step)
                for key, l in losses.items():
                    tb_valid_add("scalar", "loss/" + key,
                                 tf.reduce_mean(l), step=step)
                tb_valid_flush()
            return metrics[f"loss_sum_valid"] == metrics["loss_sum_valid_best"]
        else:
            if tensorboard_dir is not None:
                tb_train_add("scalar", "optimizer/learning_rate",
                             learning_rate, step=step)
                tb_train_add("scalar", "loss/total",
                             tf.reduce_mean(total_loss), step=step)
                for key, l in losses.items():
                    tb_train_add("scalar", "loss/" + key,
                                 tf.reduce_mean(l), step=step)
                for v in model.trainable_variables:
                    tb_train_add(
                        "histogram", "weight/{}".format(v.name), v, step=step)
                for v, g in zip(model.trainable_variables, gradients):
                    tb_train_add(
                        "histogram", "gradient/{}".format(v.name), g, step=step)
                tb_train_flush()

    n_masses = len(masses)
    n_spins = len(spins)
    batch_size = 0
    # start the loop
    for step, (inputs, cat_inputs, targets, classes, event_weights, recoGenTauH_mass, mass_loss_inputs) in enumerate(dataset_train):
        if step == 0:
            batch_size = len(event_weights)

        if masses:
            para_mass = tf.transpose(tf.where(inputs[:, -1] == -1, tf.gather(masses, tf.random.categorical(tf.math.log([[1.]*n_masses]), batch_size)), inputs[:, -1]))
            inputs = tf.concat([inputs[:, 0:-1], para_mass], axis=1)

        if spins:
            para_spin = tf.transpose(tf.where(cat_inputs[:, -1] == -1, tf.gather(spins, tf.random.categorical(tf.math.log([[1.]*n_spins]), batch_size)), cat_inputs[:, -1]))
            cat_inputs = tf.concat([cat_inputs[:, 0:-1], para_spin], axis=1)

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
                    predictions=(predictions[pred_i] if (pred_i := getattr(
                        loss_fn, "prediction_index", None)) != None else predictions),
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

                if masses:
                    para_mass_valid = tf.transpose(tf.where(inputs_valid[:, -1] == -1, tf.gather(masses, tf.random.categorical(tf.math.log([[1.]*n_masses]), batch_size)), inputs_valid[:, -1]))
                    inputs_valid = tf.concat([inputs_valid[:, 0:-1], para_mass_valid], axis=1)

                if spins:

                    para_spin_valid = tf.transpose(tf.where(cat_inputs_valid[:, -1] == -1, tf.gather(spins, tf.random.categorical(tf.math.log([[1.]*n_spins]), batch_size)), cat_inputs_valid[:, -1]))
                    cat_inputs_valid = tf.concat([cat_inputs_valid[:, 0:-1], para_spin_valid], axis=1)

                predictions_valid = model(
                    [inputs_valid, cat_inputs_valid],  training=False)

                losses_valid = {
                    name: loss_fn(
                        labels=(classes_valid if "ce" in name else targets_valid),
                        predictions=(predictions_valid[pred_i] if (pred_i := getattr(
                            loss_fn, "prediction_index", None)) != None else predictions_valid),
                        event_weights=event_weights_valid,
                        recoGenTauH_mass=recoGenTauH_mass_valid,
                        mass_loss_inputs=mass_loss_inputs_valid,
                    )
                    for name, loss_fn in loss_fns.items()
                }

                for name, loss_tensor in losses_valid.items():
                    losses_valid_avg[name].append(loss_tensor)
                loss_valid_avg.append(tf.add_n(list(losses_valid.values())))
            is_best = update_metrics(
                "valid", step, losses_valid_avg, loss_valid_avg)

            # store the best model
            if is_best:
                best_weights = model.get_weights()
                best_weights_steps = step + 1
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter == learning_rate_patience and learning_rate_reduction_counter < max_learning_rate_reductions:
                    learning_rate.assign(learning_rate / 2)
                    learning_rate_reduction_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    message = f"\nearly stopping: validation loss did not improve within the last {early_stopping_patience} validation steps"
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
            update = [f"step: {metrics['step']}, n_val: {metrics['step_val']}, es: {metrics['early_stopping_counter']}, lr: {metrics['learning_rate']:.5f}"]
            for name in losses:
                update.append(
                    f"{name}: {metrics[f'loss_{name}_train']:.4f} | {metrics[f'loss_{name}_valid']:.4f}")
            update.append(f"sum-l2: {metrics[f'loss_sum_train']:.4f} | {metrics[f'loss_sum_valid']:.4f} | {metrics[f'loss_sum_valid_best']:.4f}")

            update = ", ".join(update)
            sys.stdout.write("\033[K")
            print(update, end="\r")
    else:
        message = "dataset exhausted, stopping training"

    print(message)
    print("validation metrics of the best model:")
    print(f"Loss sum - L2: {metrics['loss_sum_valid_best']:.4f}")
    return best_weights, best_weights_steps, metrics


if __name__ == "__main__":
    main()
