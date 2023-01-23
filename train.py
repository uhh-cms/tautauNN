#!/usr/bin/env python3
# coding: utf-8
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys
import os
import gc
import functools
from operator import mul
from collections import defaultdict
from util import load_sample, phi_mpi_to_pi, split_train_validation_mask, calc_new_columns, create_tensorboard_callbacks
from custom_layers import CustomEmbeddingLayer, CustomOutputScalingLayer
from multi_dataset import MultiDataset

def main(basepath                   =   "/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v2_31Aug22",
                                        # "/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_llr_2018_forTraining",
        tensorboard_dir            =   "/tmp/tensorboard",
         samples                    =   {
                                        "SKIM_GGHH_SM": (1./35, 1.), # (batch fraction weight, event weight factor)
                                        "SKIM_ggF_Radion_m300": (1./35, 1.),
                                        "SKIM_ggF_Radion_m350": (1./35, 1.),
                                        "SKIM_ggF_Radion_m400": (1./35, 1.),
                                        "SKIM_ggF_Radion_m450": (1./35, 1.),
                                        "SKIM_ggF_Radion_m500": (1./35, 1.),
                                        "SKIM_ggF_Radion_m550": (1./35, 1.),
                                        "SKIM_ggF_Radion_m600": (1./35, 1.),
                                        "SKIM_ggF_Radion_m650": (1./35, 1.),
                                        "SKIM_ggF_Radion_m700": (1./35, 1.),
                                        "SKIM_ggF_Radion_m750": (1./35, 1.),
                                        "SKIM_ggF_Radion_m800": (1./35, 1.),
                                        "SKIM_ggF_Radion_m850": (1./35, 1.),
                                        "SKIM_ggF_Radion_m900": (1./35, 1.),
                                        "SKIM_ggF_Radion_m1000": (1./35, 1.),
                                        "SKIM_ggF_Radion_m1250": (1./35, 1.),
                                        "SKIM_ggF_Radion_m1500": (1./35, 1.),
                                        "SKIM_ggF_Radion_m1750": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m300": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m350": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m400": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m450": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m500": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m550": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m600": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m650": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m700": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m750": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m800": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m850": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m900": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m1000": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m1250": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m1500": (1./35, 1.),
                                        "SKIM_ggF_BulkGraviton_m1750": (1./35, 1.),
                                        # "SKIM_DYmerged": (1., 1.),
                                        "SKIM_DY_NLO_incl": (1., 1.),
                                        "SKIM_TT_fullyLep": (1., 1.),
                                        # "SKIM_TT_semiLep": (1., 1.),
                                        },
         columns_to_read            =   [
                                        "dau1_pt", "dau1_eta", "dau1_phi", "dau1_e", "dau1_dxy", "dau1_dz", "dau1_iso", 
                                        "dau2_pt", "dau2_eta", "dau2_phi", "dau2_e", "dau2_dxy", "dau2_dz", "dau2_iso", 
                                        "met_et", "met_phi", "met_cov00", "met_cov01", "met_cov11", 
                                        "bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_e", "bjet1_btag_deepFlavor",
                                        "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_e", "bjet2_btag_deepFlavor",
                                        "tauH_mass", 
                                        "pairType", "dau1_decayMode", "dau2_decayMode",
                                        # "matchedGenLepton1_pt", "matchedGenLepton2_pt",
                                        "genNu1_pt", "genNu1_eta", "genNu1_phi", "genNu1_e", 
                                        "genNu2_pt", "genNu2_eta", "genNu2_phi", "genNu2_e",
                                        #"npu", "npv"
                                        ],
         columns_to_add             =   {
                                        "dau1_met_dphi": (("dau1_phi", "met_phi"),(lambda a,b: phi_mpi_to_pi(a-b))),
                                        "dau2_met_dphi": (("dau2_phi", "met_phi"),(lambda a,b: phi_mpi_to_pi(a-b))),
                                        "genNu1_met_dphi": (("genNu1_phi", "met_phi"),(lambda a,b: phi_mpi_to_pi(a-b))),
                                        "genNu2_met_dphi": (("genNu2_phi", "met_phi"),(lambda a,b: phi_mpi_to_pi(a-b))),
                                        # "dau1_pt_factor": (("dau1_pt", "matchedGenLepton1_pt"),(lambda a,b: a/b)),
                                        # "dau2_pt_factor": (("dau2_pt", "matchedGenLepton2_pt"),(lambda a,b: a/b)),
                                        "dau1_px": (("dau1_pt", "dau1_met_dphi"), (lambda a,b: a * np.cos(b))),
                                        "dau1_py": (("dau1_pt", "dau1_met_dphi"), (lambda a,b: a * np.sin(b))),
                                        "dau1_pz": (("dau1_pt", "dau1_eta"), (lambda a,b: a * np.sinh(b))),
                                        "dau1_m": (("dau1_px", "dau1_py", "dau1_pz", "dau1_e"), (lambda x,y,z,e: np.sqrt(e**2 - (x**2 + y**2 + z**2)))),
                                        # "dau1_mt": (("dau1_pz", "dau1_e"), (lambda z,e: np.sqrt(e**2-z**2))),
                                        "dau2_px": (("dau2_pt", "dau2_met_dphi"), (lambda a,b: a * np.cos(b))),
                                        "dau2_py": (("dau2_pt", "dau2_met_dphi"), (lambda a,b: a * np.sin(b))),
                                        "dau2_pz": (("dau2_pt", "dau2_eta"), (lambda a,b: a * np.sinh(b))),
                                        "dau2_m": (("dau2_px", "dau2_py", "dau2_pz", "dau2_e"), (lambda x,y,z,e: np.sqrt(e**2 - (x**2 + y**2 + z**2)))),
                                        # "dau2_mt": (("dau2_pz", "dau2_e"), (lambda z,e: np.sqrt(e**2-z**2))),
                                        "ditau_deltaphi": (("dau1_met_dphi", "dau2_met_dphi"), (lambda a,b: np.abs(phi_mpi_to_pi(a-b)))),
                                        "ditau_deltaeta": (("dau1_eta", "dau2_eta"), (lambda a,b: np.abs(a-b))),
                                        "genNu1_px": (("genNu1_pt", "genNu1_met_dphi"), (lambda a,b: a * np.cos(b))),
                                        "genNu1_py": (("genNu1_pt", "genNu1_met_dphi"), (lambda a,b: a * np.sin(b))),
                                        "genNu1_pz": (("genNu1_pt", "genNu1_eta"), (lambda a,b: a * np.sinh(b))),
                                        "genNu2_px": (("genNu2_pt", "genNu2_met_dphi"), (lambda a,b: a * np.cos(b))),
                                        "genNu2_py": (("genNu2_pt", "genNu2_met_dphi"), (lambda a,b: a * np.sin(b))),
                                        "genNu2_pz": (("genNu2_pt", "genNu2_eta"), (lambda a,b: a * np.sinh(b))),
                                        # "met_px": (("met_et", "met_phi"), (lambda a,b: a * np.cos(b))),
                                        # "met_py": (("met_et", "met_phi"), (lambda a,b: a * np.sin(b))),
                                        # "mT_tau1": (("dau1_mt", "dau1_px", "dau1_py", "met_et"), (lambda e1, x1, y1, e2: np.sqrt((e1+e2)**2-(x1+e2)**2-(y1+0)**2))),
                                        # "mT_tau2": (("dau2_mt", "dau2_px", "dau2_py", "met_et"), (lambda e1, x1, y1, e2: np.sqrt((e1+e2)**2-(x1+e2)**2-(y1+0)**2))),
                                        # "mT_tautau": (("dau1_mt", "dau1_px", "dau1_py", "dau2_mt", "dau2_px", "dau2_py", "met_et"), (lambda e1, x1, y1, e2, x2, y2, e3: np.sqrt((e1+e2+e3)**2-(x1+x2+e3)**2-(y1+y2+0)**2))),
                                        "bjet1_met_dphi": (("bjet1_phi", "met_phi"),(lambda a,b: phi_mpi_to_pi(a-b))),
                                        "bjet1_px": (("bjet1_pt", "bjet1_met_dphi"), (lambda a,b: a * np.cos(b))),
                                        "bjet1_py": (("bjet1_pt", "bjet1_met_dphi"), (lambda a,b: a * np.sin(b))),
                                        "bjet1_pz": (("bjet1_pt", "bjet1_eta"), (lambda a,b: a * np.sinh(b))),
                                        "bjet2_met_dphi": (("bjet2_phi", "met_phi"),(lambda a,b: phi_mpi_to_pi(a-b))),
                                        "bjet2_px": (("bjet2_pt", "bjet2_met_dphi"), (lambda a,b: a * np.cos(b))),
                                        "bjet2_py": (("bjet2_pt", "bjet2_met_dphi"), (lambda a,b: a * np.sin(b))),
                                        "bjet2_pz": (("bjet2_pt", "bjet2_eta"), (lambda a,b: a * np.sinh(b))),
                                        },
         float_inputs               =   [
                                        #"dau1_pt", "dau1_eta", "dau1_phi", "dau1_e", "dau1_dxy", "dau1_dz", "dau1_iso",
                                        #"dau2_pt", "dau2_eta", "dau2_phi", "dau2_e", "dau2_dxy", "dau2_dz", "dau2_iso",
                                        "met_et", #"met_phi", "met_cov00", "met_cov01", "met_cov11",
                                        "ditau_deltaphi", "ditau_deltaeta",
                                        "dau1_px", "dau1_py", "dau1_pz", "dau1_e", "dau1_dxy", "dau1_dz", "dau1_iso",
                                        "dau2_px", "dau2_py", "dau2_pz", "dau2_e", "dau2_dxy", "dau2_dz", "dau2_iso", 
                                        #"met_px", "met_py", 
                                        "met_cov00", "met_cov01", "met_cov11",
                                        "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_e", "bjet1_btag_deepFlavor",
                                        "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_e", "bjet2_btag_deepFlavor",
                                        #"tauH_mass", "mT_tau1", "mT_tau2", "mT_tautau", "npu", "npv",
                                        ],
         int_inputs                 =   [
                                        "pairType", "dau1_decayMode", "dau2_decayMode"
                                        ],
         targets                    =   [
                                        # "dau1_pt_factor", 
                                        # "dau2_pt_factor",
                                        "genNu1_px", "genNu1_py", "genNu1_pz", "genNu1_e", 
                                        "genNu2_px", "genNu2_py", "genNu2_pz", "genNu2_e",
                                        ],
         selections                 =   [
                                        # (("nleps",), (lambda a: a==0)),
                                        # (("nbjetscand",), (lambda a: a>1)),
                                        # (("isOS",), (lambda a: a==1)),
                                        # (("dau2_deepTauVsJet",), (lambda a: a>=5)),
                                        # (("pairType", "dau1_iso", "dau1_deepTauVsJet"), (lambda a, b, c: (((a==0) & (b < 0.15)) | (a==1) | ((a==2) & (c>=5))))),
                                        (("pairType",),(lambda a: a < 3)),
                                        # (("genLeptons_matched",),(lambda a: a == 1)),
                                        # (("genBQuarks_matched",),(lambda a: a == 1)),
                                        ],
         embedding_expected_inputs  =   [[0, 1, 2], [-1, 0, 1, 10, 11], [0, 1, 10, 11]],
         embedding_output_dim       =   5,
         units                      =   (128, 128, 128, 128, 128, 128, 128, 128, 128),
         activation                 =   "selu",
         l2_norm                    =   50.0,
         dropout_rate               =   0,
         batch_size                 =   2048,
         epochs                     =   2000,
         train_valid_fraction       =   0.75,
         train_valid_seed           =   0,
         initial_learning_rate      =   0.0025,
         learning_rate_patience     =   10,
         early_stopping_patience    =   20,
         output_scaling             =   True,
         ):
    
    device = get_device(device="gpu", num_device= 0)
    
    float_input_train = []
    float_input_valid = []

    int_input_train = []
    int_input_valid = []

    target_train = []
    target_valid = []

    batch_weights = []

    event_weights_train = []
    event_weights_valid = []

    float_input_means = []
    float_input_vars = []

    target_means = []
    target_stds = []

    for sample, (batch_weight, event_weight) in samples.items():
        d, event_weights = load_sample(basepath, sample, event_weight, columns_to_read, selections)

        d = calc_new_columns(d, columns_to_add)

        float_input_vecs = d[float_inputs]
        int_input_vecs = d[int_inputs]
        target_vecs = d[targets]

        float_input_vecs = float_input_vecs.astype([(name, np.float32) for name in float_input_vecs.dtype.names], copy = False).view(np.float32).reshape((-1, len(float_input_vecs.dtype)))
        int_input_vecs = int_input_vecs.astype([(name, np.float32) for name in int_input_vecs.dtype.names], copy = False).view(np.float32).reshape((-1, len(int_input_vecs.dtype)))
        target_vecs = target_vecs.astype([(name, np.float32) for name in target_vecs.dtype.names], copy = False).view(np.float32).reshape((-1, len(target_vecs.dtype)))

        # input feature scaling
        float_input_means.append(np.mean(float_input_vecs, axis = 0))
        float_input_vars.append(np.var(float_input_vecs, axis = 0))

        # output scaling
        if output_scaling:
            target_means.append(np.mean(target_vecs, axis = 0))
            target_stds.append(np.std(target_vecs, axis = 0))

        train_mask = split_train_validation_mask(len(event_weights), fraction = train_valid_fraction, seed = train_valid_seed)

        float_input_train.append(float_input_vecs[train_mask])
        float_input_valid.append(float_input_vecs[~train_mask])

        int_input_train.append(int_input_vecs[train_mask])
        int_input_valid.append(int_input_vecs[~train_mask])

        target_train.append(target_vecs[train_mask])
        target_valid.append(target_vecs[~train_mask])

        batch_weights.append(batch_weight)

        event_weights_train.append(np.expand_dims(event_weights[train_mask], axis=1))
        event_weights_valid.append(np.expand_dims(event_weights[~train_mask], axis=1))

    if output_scaling:
        target_means = np.mean(target_means, axis=0)
        target_stds = np.mean(target_stds, axis=0)
        target_train = [(x - target_means)/target_stds for x in target_train]
        target_valid = [(x - target_means)/target_stds for x in target_valid]
    else:
        target_means = None
        target_stds = None

    dataset_train = MultiDataset(zip(zip(float_input_train, int_input_train, target_train, event_weights_train), batch_weights), batch_size, True, True)
    dataset_valid = MultiDataset(zip(zip(float_input_valid, int_input_valid, target_valid, event_weights_valid), batch_weights), batch_size, False, False)

    float_input_means = np.mean(float_input_means, axis=0)
    float_input_vars = np.mean(float_input_vars, axis=0)

    with device:
        model, regularization_weights = create_model(len(float_inputs),
                                                     len(int_inputs),
                                                     len(targets),
                                                     embedding_expected_inputs,
                                                     embedding_output_dim,
                                                     float_input_means,
                                                     float_input_vars,
                                                     target_means = target_means,
                                                     target_stds = target_stds,
                                                     units = units,
                                                     activation=activation,
                                                     dropout_rate=dropout_rate)
        model.summary()

        loss_fns = create_losses(regularization_weights, l2_norm=l2_norm)
        if target_means is not None and target_stds is not None:
            for name, loss_fn in loss_fns.items():
                if "mse" in name:
                    loss_fn.prediction_index = 0
        optimizer, learning_rate = create_optimizer(initial_learning_rate)

        best_weights, _ = training_loop(dataset_train,
                                        dataset_valid,
                                        model,
                                        loss_fns,
                                        optimizer,
                                        learning_rate,
                                        log_every=10,
                                        validate_every=100,
                                        tensorboard_dir = tensorboard_dir,
                                        early_stopping_patience = early_stopping_patience,
                                        learning_rate_patience = learning_rate_patience,
                                        )

        model.set_weights(best_weights)
        model.save("models/best_model")

def create_dataset(float_input_vecs, int_input_vecs, target_vecs, event_weights, shuffle=False, repeat=1, batch_size=1024, seed=None, **kwargs):
    nevents =  float_input_vecs.shape[0]

    # create a tf dataset
    data = (float_input_vecs, int_input_vecs, target_vecs, np.expand_dims(event_weights, axis=1))
    ds = tf.data.Dataset.from_tensor_slices(data)

    # in the following, we amend the dataset object using methods
    # that return a new dataset object *without* copying the data

    # apply shuffeling
    if shuffle:
        ds = ds.shuffle(10 * nevents, reshuffle_each_iteration=True, seed=seed)

    # apply repetition, i.e. start iterating from the beginning when the dataset is exhausted
    ds = ds.repeat(repeat)

    # apply batching
    if batch_size < 1:
        batch_size = nevents
    ds = ds.batch(batch_size)

    return ds

def create_model(float_input_shape, int_input_shape, output_shape, embedding_expected_inputs, embedding_output_dim, float_input_means, float_input_vars, target_means = None, target_stds = None, units=(128, 128, 128), activation="selu", dropout_rate=0.):
    # track weights for later use
    weights = []

    # input layers
    x1 = tf.keras.Input(float_input_shape)
    x2 = tf.keras.Input(int_input_shape)

    norm_layer = tf.keras.layers.Normalization(mean=float_input_means, variance=float_input_vars)
    n = norm_layer(x1)

    # only add embedding layer if number of integer vars > 0
    if int_input_shape > 0:
        custom_embedding_layer = CustomEmbeddingLayer(output_dim = embedding_output_dim, expected_inputs=embedding_expected_inputs)
        a = custom_embedding_layer(x2)
        a = tf.keras.layers.Concatenate()([n, a])
    else:
        a = n

    # add layers programatically
    for n in units:
        # build the layer    
        dense_layer = tf.keras.layers.Dense(n, use_bias=True)
        a = dense_layer(a)

        activation_layer = tf.keras.layers.Activation(activation)
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

    # add the output layer
    y1 = tf.keras.layers.Dense(output_shape, use_bias=True)(a)
    outputs = [y1]
    if target_means is not None and target_stds is not None:
        y2 = CustomOutputScalingLayer(target_means, target_stds)(y1)
        outputs.append(y2)

    # build the model
    model = tf.keras.Model(inputs=[x1, x2], outputs=outputs, name="htautau_regression")
    return model, weights

# define the losses
def create_losses(modelweights, l2_norm=10):
    n_modelweights = sum(functools.reduce(mul, w.shape) for w in modelweights)

    # cross entropy
    @tf.function
    def loss_ce_fn(labels, predictions, event_weights):
        # ensure proper prediction values before applying log's
        predictions = tf.clip_by_value(predictions, 1e-6, 1 - 1e-6)
        loss_ce = tf.reduce_mean(event_weights * -labels * tf.math.log(predictions))
        return loss_ce

    # l2 loss
    @tf.function
    def loss_l2_fn(labels, predictions, event_weights):
        # accept labels and predictions although we don't need them
        # but this makes it easier to call all loss functions the same way
        loss_l2 = sum(tf.reduce_sum(w**2) for w in modelweights)

        return l2_norm/n_modelweights * loss_l2

    # MSE loss
    @tf.function
    def loss_mse_fn(labels, predictions, event_weights):
        # compute the mse loss
        loss_mse = tf.reduce_mean(event_weights * (labels - predictions)**2.)
        return loss_mse

    # return a dict with all loss function components
    return {"mse": loss_mse_fn, "l2": loss_l2_fn}

def create_optimizer(initial_learning_rate):
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
    max_steps=10000,
    log_every=10,
    validate_every=100,
    tensorboard_dir = None,
    early_stopping_patience = 20,
    learning_rate_patience = 10,
):    
    # store the best model, identified by the best validation accuracy
    best_weights = None

    # metrics to update during training
    metrics = dict(
        step=0, step_val=0,
        mse_valid_best=sys.maxsize,
        early_stopping_counter=0,
        learning_rate=learning_rate.numpy(),
    )
    for name in loss_fns:
        for kind in ["train", "valid"]:
            metrics[f"loss_{name}_{kind}"] = 0

    # progress bar format
    fmt = ["{percentage:3.0f}% {bar} Step: {postfix[0][step]}/{total}, Validations: {postfix[0][step_val]}, Early stopping counter: {postfix[0][early_stopping_counter]}, Learning rate: {postfix[0][learning_rate]:.5f}"]
    for name in loss_fns:
        if "mse" in name:
            fmt.append(f"Loss '{name}': {{postfix[0][loss_{name}_train]:.4f}} | {{postfix[0][loss_{name}_valid]:.4f}} | {{postfix[0][mse_valid_best]:.4f}}")
        else:
            fmt.append(f"Loss '{name}': {{postfix[0][loss_{name}_train]:.4f}} | {{postfix[0][loss_{name}_valid]:.4f}}")
    fmt = " --- ".join(fmt)

    if tensorboard_dir is not None:
        # helpers to add tensors and metrics to tensorboard for monitoring
        tb_log_dir = lambda kind: tensorboard_dir and os.path.join(tensorboard_dir, kind)
        tb_train_batch_add, tb_train_batch_flush = create_tensorboard_callbacks(tb_log_dir("train_batch"))
        tb_train_add, tb_train_flush = create_tensorboard_callbacks(tb_log_dir("train"))
        tb_valid_add, tb_valid_flush = create_tensorboard_callbacks(tb_log_dir("valid"))
        tb_best_add, tb_best_flush = create_tensorboard_callbacks(tb_log_dir("best"))

    # helper to update metrics
    def update_metrics(kind, step, losses, total_loss):

        # update bar data
        metrics["step"] = step + 1
        for name, loss in losses.items():
            metrics[f"loss_{name}_{kind}"] = tf.reduce_mean(loss)

        metrics["early_stopping_counter"] = early_stopping_counter
        metrics["learning_rate"] = learning_rate.numpy()

        # validation specific
        if kind == "valid":
            metrics["step_val"] += 1
            metrics["mse_valid_best"] = min(metrics["mse_valid_best"], metrics[f"loss_mse_valid"])

            if tensorboard_dir is not None:
                add_funcs, flush_funcs = [tb_valid_add], [tb_valid_flush]
                if is_best:=metrics[f"loss_mse_valid"] == metrics["mse_valid_best"]:
                    add_funcs.append(tb_best_add)
                    flush_funcs.append(tb_best_flush)
                for add, flush in zip(add_funcs, flush_funcs):
                    add("scalar", "loss/total", total_loss, step=step)
                    for key, l in losses.items():
                        add("scalar", "loss/" + key, tf.reduce_mean(l), step=step)
                    flush()
                return is_best
            return metrics[f"loss_mse_valid"] == metrics["mse_valid_best"]
        elif kind == "train_batch":
            if tensorboard_dir is not None:
                tb_train_batch_add("scalar", "optimizer/learning_rate", learning_rate, step=step)
                tb_train_batch_add("scalar", "loss/total", tf.reduce_mean(total_loss), step=step)
                for key, l in losses.items():
                    tb_train_batch_add("scalar", "loss/" + key, tf.reduce_mean(l), step=step)
                for v in model.trainable_variables:
                    tb_train_batch_add("histogram", "weight/{}".format(v.name), v, step=step)
                for v, g in zip(model.trainable_variables, gradients):
                    tb_train_batch_add("histogram", "gradient/{}".format(v.name), g, step=step)
                tb_train_batch_flush()
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
    early_stopping_counter = 0
    message = ""
    with tqdm(total=max_steps, bar_format=fmt, postfix=[metrics]) as bar:
        losses_avg = defaultdict(list)
        loss_avg = []
        for step, (float_inputs, int_inputs, targets, event_weights) in enumerate(dataset_train):
            
            if step == 0 and tensorboard_dir is not None:
                tb_train_add("trace_on", graph=True)
                tb_train_add("trace_export", "graph", step=step)
                tb_train_add("trace_off")

            # do a train step
            with tf.GradientTape() as tape:
                # get predictions
                predictions = model([float_inputs, int_inputs], training=True)

                # compute all losses and combine them into the total loss

                losses = {
                    name: loss_fn(
                        targets,
                        predictions[pred_i] if (pred_i := getattr(loss_fn, "prediction_index", None)) != None else predictions,
                        event_weights,
                    )
                    for name, loss_fn in loss_fns.items()
                } 
                
                loss = tf.add_n(list(losses.values()))

            # validation
            do_validate = step % validate_every == 0
            if do_validate:
                float_inputs_valid, int_inputs_valid, targets_valid, event_weights_valid = next(iter(dataset_valid))

                predictions_valid = model([float_inputs_valid, int_inputs_valid], training=False)
                losses_valid = {
                    name: loss_fn(
                        targets_valid,
                        predictions_valid[pred_i] if (pred_i := getattr(loss_fn, "prediction_index", None)) != None else predictions_valid,
                        event_weights_valid,
                    )
                    for name, loss_fn in loss_fns.items()
                }
                total_loss_valid = tf.add_n(list(losses_valid.values()))
                is_best = update_metrics("valid", step, losses_valid, total_loss_valid)

                # store the best model
                if is_best:
                    best_weights = model.get_weights()
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter == learning_rate_patience:
                        learning_rate.assign(learning_rate/2)
                    if early_stopping_counter > early_stopping_patience:
                        message = f"early stopping: validation loss did not improve within the last {early_stopping_patience} validation steps"
                        break

            # get and propagate gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            for name, loss_tensor in losses.items():
                losses_avg[name].append(loss_tensor)

            loss_avg.append(loss)
            # logging
            do_log = step % log_every == 0
            if do_log:
                update_metrics("train_batch", step,  losses, loss)
                update_metrics("train", step,  losses_avg, loss_avg)
                losses_avg.clear()
                del loss_avg[:]

            bar.update()

        else:
            message = "dataset exhausted, stopping training"
    
    print(message)
    print("validation metrics of the best model:")
    print(f"MSE: {metrics['mse_valid_best']:.4f}")
    return best_weights, metrics

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

if __name__ == "__main__":
    main()