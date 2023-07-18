#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import mplhep as hep
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
from util import load_sample, phi_mpi_to_pi, calc_new_columns


def main(
        basepath="/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v4_02Mar23",
        outputdir="plots/no_singleH_add_bjetvars_3classification_massloss_simonesSelection/",
        modelpath="models/no_singleH_add_bjetvars_3classification_massloss_simonesSelection/",
        samples={
            "SKIM_GGHH_SM": (1./35, 1.),  # (batch fraction weight, event weight factor)
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
            "SKIM_DY_amc_incl": (1., 1.),
            "SKIM_TT_fullyLep": (1., 1.),
            "SKIM_TT_semiLep": (1., 1.),
            # "SKIM_GluGluHToTauTau": (1., 1.),
            "SKIM_ttHToTauTau": (1., 1.),
        },
        columns_to_read=[
            "dau1_pt", "dau1_eta", "dau1_phi", "dau1_e", "dau1_dxy", "dau1_dz", "dau1_iso",
            "dau2_pt", "dau2_eta", "dau2_phi", "dau2_e", "dau2_dxy", "dau2_dz", "dau2_iso",
            "met_et", "met_phi", "met_cov00", "met_cov01", "met_cov11",
            "bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_e", "bjet1_btag_deepFlavor", "bjet1_cID_deepFlavor", "bjet1_pnet_bb", "bjet1_pnet_cc", "bjet1_pnet_b", "bjet1_pnet_c", "bjet1_pnet_g", "bjet1_pnet_uds", "bjet1_pnet_pu", "bjet1_pnet_undef", "bjet1_HHbtag",
            "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_e", "bjet2_btag_deepFlavor", "bjet2_cID_deepFlavor", "bjet2_pnet_bb", "bjet2_pnet_cc", "bjet2_pnet_b", "bjet2_pnet_c", "bjet2_pnet_g", "bjet2_pnet_uds", "bjet2_pnet_pu", "bjet2_pnet_undef", "bjet2_HHbtag",
            "pairType", "dau1_decayMode", "dau2_decayMode",
            "genNu1_pt", "genNu1_eta", "genNu1_phi", "genNu1_e",
            "genNu2_pt", "genNu2_eta", "genNu2_phi", "genNu2_e",
            "tauH_SVFIT_mass", "tauH_SVFIT_pt", "tauH_SVFIT_eta", "tauH_SVFIT_phi",
            "tauH_mass", "tauH_pt", "tauH_eta", "tauH_phi",
            "recoGenTauH_mass", "recoGenTauH_pt", "recoGenTauH_eta", "recoGenTauH_phi",
            "matchedGenLepton1_pt", "matchedGenLepton1_eta", "matchedGenLepton1_phi", "matchedGenLepton1_e",
            "matchedGenLepton2_pt", "matchedGenLepton2_eta", "matchedGenLepton2_phi", "matchedGenLepton2_e",
            "DeepMET_ResponseTune_px", "DeepMET_ResponseTune_py", "DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py",
            "bH_pt", "bH_eta", "bH_phi", "bH_e", "bH_mass", "HH_pt", "HH_mass", "HHKin_mass",
            "EventNumber",  # "lumi_weight", "plot_weight",
        ],
        columns_to_add={
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
            "genNu1_dphi": (("genNu1_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a-b))),
            "genNu2_dphi": (("genNu2_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a-b))),
            # "matchedGenLepton1_dphi": (("matchedGenLepton1_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
            # "matchedGenLepton2_dphi": (("matchedGenLepton2_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
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
            # "matchedGenLepton1_px": (("matchedGenLepton1_pt", "matchedGenLepton1_dphi"), (lambda a, b: a * np.cos(b))),
            # "matchedGenLepton1_py": (("matchedGenLepton1_pt", "matchedGenLepton1_dphi"), (lambda a, b: a * np.sin(b))),
            # "matchedGenLepton1_pz": (("matchedGenLepton1_pt", "matchedGenLepton1_eta"), (lambda a, b: a * np.sinh(b))),
            # "matchedGenLepton2_px": (("matchedGenLepton2_pt", "matchedGenLepton2_dphi"), (lambda a, b: a * np.cos(b))),
            # "matchedGenLepton2_py": (("matchedGenLepton2_pt", "matchedGenLepton2_dphi"), (lambda a, b: a * np.sin(b))),
            # "matchedGenLepton2_pz": (("matchedGenLepton2_pt", "matchedGenLepton2_eta"), (lambda a, b: a * np.sinh(b))),
            "bjet1_dphi": (("bjet1_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
            "bjet1_px": (("bjet1_pt", "bjet1_dphi"), (lambda a, b: a * np.cos(b))),
            "bjet1_py": (("bjet1_pt", "bjet1_dphi"), (lambda a, b: a * np.sin(b))),
            "bjet1_pz": (("bjet1_pt", "bjet1_eta"), (lambda a, b: a * np.sinh(b))),
            "bjet2_dphi": (("bjet2_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
            "bjet2_px": (("bjet2_pt", "bjet2_dphi"), (lambda a, b: a * np.cos(b))),
            "bjet2_py": (("bjet2_pt", "bjet2_dphi"), (lambda a, b: a * np.sin(b))),
            "bjet2_pz": (("bjet2_pt", "bjet2_eta"), (lambda a, b: a * np.sinh(b))),
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

        selections=[(("pairType",), (lambda a: a < 3)),
                    (("nbjetscand",), (lambda a: a > 1)),
                    # (("genLeptons_matched",), (lambda a: a == 1)),
                    (("nleps",), (lambda a: a == 0)),
                    (("isOS",), (lambda a: a == 1)),
                    (("dau2_deepTauVsJet",), (lambda a: a >= 5)),
                    (("pairType", "dau1_iso", "dau1_eleMVAiso", "dau1_deepTauVsJet"), (lambda a, b, c, d: (((a == 0) & (b < 0.15)) | ((a == 1) & (c == 1)) | ((a == 2) & (d >= 5))))),
                    ],
        train_valid_eventnumber_modulo=4,
        plot_only="valid",
):
    hep.style.use("CMS")

    model = tf.keras.models.load_model(modelpath, compile=False)

    data = {}

    for sample, (batch_weight, event_weight) in samples.items():

        d, event_weights = load_sample(basepath, sample, event_weight, columns_to_read, selections)
        nevents = len(event_weights)
        weights = event_weights*batch_weight/nevents
        d = rfn.rec_append_fields(d, "weight", weights)

        if plot_only in ["train", "valid"]:
            train_mask = (d["EventNumber"] % train_valid_eventnumber_modulo) != 0
            d = calc_new_columns(
                d[train_mask if plot_only == "train" else ~train_mask], columns_to_add)
        else:
            d = calc_new_columns(d, columns_to_add)

        inputs = d[input_names]
        cat_inputs = d[cat_input_names]

        inputs = inputs.astype([(name, np.float32) for name in inputs.dtype.names], copy=False).view(np.float32).reshape((-1, len(inputs.dtype)))
        cat_inputs = cat_inputs.astype([(name, np.float32) for name in cat_inputs.dtype.names], copy=False).view(np.float32).reshape((-1, len(cat_inputs.dtype)))

        predictions = model.predict([inputs, cat_inputs])

        d = rfn.rec_append_fields(d, ["reg_nu1_px", "reg_nu1_py", "reg_nu1_pz", "reg_nu2_px", "reg_nu2_py", "reg_nu2_pz"], [
                                  predictions[1][:, i] for i in range(predictions[1].shape[1])], dtypes=["<f4"] * predictions[1].shape[1])

        reg_H = {
            "reg_H_e": (("dau1_e", "reg_nu1_px", "reg_nu1_py", "reg_nu1_pz", "dau2_e", "reg_nu2_px", "reg_nu2_py", "reg_nu2_pz"), (lambda a, b, c, d, e, f, g, h: a + np.sqrt(b**2+c**2+d**2) + e + np.sqrt(f**2+g**2+h**2))),
            "reg_H_px": (("dau1_px", "reg_nu1_px", "dau2_px", "reg_nu2_px"), (lambda a, b, c, d: a + b + c + d)),
            "reg_H_py": (("dau1_py", "reg_nu1_py", "dau2_py", "reg_nu2_py"), (lambda a, b, c, d: a + b + c + d)),
            "reg_H_pz": (("dau1_pz", "reg_nu1_pz", "dau2_pz", "reg_nu2_pz"), (lambda a, b, c, d: a + b + c + d)),
            "reg_H_m": (("reg_H_e", "reg_H_px", "reg_H_py", "reg_H_pz"), (lambda a, b, c, d: np.sqrt(a**2-b**2-c**2-d**2))),
            "reg_H_pt": (("reg_H_px", "reg_H_py"), (lambda a, b: np.sqrt(a**2 + b**2))),
            "reg_H_eta": (("reg_H_pt", "reg_H_pz"), (lambda a, b: np.arcsinh(b/a))),
            "reg_H_phi": (("reg_H_px", "reg_H_py"), (lambda a, b: np.arctan2(a, b))),
        }

        d = calc_new_columns(d, reg_H)

        reg_HH = {
            "reg_HH_e": (("reg_H_e", "bH_e"), (lambda a, b: a + b)),
            "reg_HH_px": (("reg_H_px", "bH_pt", "bH_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b, c, d: a + b * np.cos(phi_mpi_to_pi(c-d)))),
            "reg_HH_py": (("reg_H_py", "bH_pt", "bH_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b, c, d: a + b * np.sin(phi_mpi_to_pi(c-d)))),
            "reg_HH_pz": (("reg_H_pz", "bH_pt", "bH_eta"), (lambda a, b, c: a + b * np.sinh(c))),
            "reg_HH_m": (("reg_HH_e", "reg_HH_px", "reg_HH_py", "reg_HH_pz"), (lambda a, b, c, d: np.sqrt(a**2-b**2-c**2-d**2))),
            "reg_HH_pt": (("reg_HH_px", "reg_HH_py"), (lambda a, b: np.sqrt(a**2 + b**2))),
        }

        d = calc_new_columns(d, reg_HH)

        svfit_HH = {
            "svfit_HH_e": (("tauH_SVFIT_mass", "tauH_SVFIT_pt", "tauH_SVFIT_eta", "bH_e"), (lambda a, b, c, d: np.sqrt(a**2 + (b * np.cosh(c))**2) + d)),
            "svfit_HH_px": (("tauH_SVFIT_pt", "tauH_SVFIT_phi", "bH_pt", "bH_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b, c, d, e: a * np.cos(phi_mpi_to_pi(b-e)) + c * np.cos(phi_mpi_to_pi(d-e)))),
            "svfit_HH_py": (("tauH_SVFIT_pt", "tauH_SVFIT_phi", "bH_pt", "bH_phi", "DeepMET_ResolutionTune_phi"), (lambda a, b, c, d, e: a * np.sin(phi_mpi_to_pi(b-e)) + c * np.sin(phi_mpi_to_pi(d-e)))),
            "svfit_HH_pz": (("tauH_SVFIT_pt", "tauH_SVFIT_eta", "bH_pt", "bH_eta"), (lambda a, b, c, d: a * np.sinh(b) + c * np.sinh(d))),
            "svfit_HH_m": (("svfit_HH_e", "svfit_HH_px", "svfit_HH_py", "svfit_HH_pz"), (lambda a, b, c, d: np.sqrt(a**2-b**2-c**2-d**2))),
            "svfit_HH_pt": (("svfit_HH_px", "svfit_HH_py"), (lambda a, b: np.sqrt(a**2 + b**2))),
        }

        d = calc_new_columns(d, svfit_HH)

        d = rfn.rec_append_fields(d, ["class_HH", "class_DY", "class_TT"], [
                                  predictions[2][:, i] for i in range(predictions[2].shape[1])], dtypes=["<f4"] * predictions[2].shape[1])
        data[sample] = d

    # plot_ROC(outputdir,
    #          "ROC_classHH_HHvsDY",
    #          "HH class (HH vs DY)",
    #          np.concatenate([[1]*len(data[f"{s}"]) for s in samples.keys() if "ggF" in s or "GGHH" in s] + [[0]*len(data["SKIM_DY_amc_incl"])]),
    #          np.concatenate([data[f"{s}"]["class_HH"] for s in samples.keys() if "ggF" in s or "GGHH" in s] + [data["SKIM_DY_amc_incl"]["class_HH"]]),
    #          weights=np.concatenate([data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s] +
    #                                 [data["SKIM_DY_amc_incl"]["weight"]])
    #          # weights=np.concatenate([data[f"{s}"]["lumi_weight"] * data[f"{s}"]["plot_weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s] +
    #          #                        [data["SKIM_DY_amc_incl"]["lumi_weight"] * data["SKIM_DY_amc_incl"]["plot_weight"]])
    #          )
    #
    # plot_ROC(outputdir,
    #          "ROC_classHH_HHvsTTdl",
    #          r"HH class (HH vs $t\overline{t}$ dl)",
    #          np.concatenate([[1]*len(data[f"{s}"]) for s in samples.keys() if "ggF" in s or "GGHH" in s] + [[0]*len(data["SKIM_TT_fullyLep"])]),
    #          np.concatenate([data[f"{s}"]["class_HH"] for s in samples.keys() if "ggF" in s or "GGHH" in s] + [data["SKIM_TT_fullyLep"]["class_HH"]]),
    #          weights=np.concatenate([data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s] +
    #                                 [data["SKIM_TT_fullyLep"]["weight"]])
    #          # weights=np.concatenate([data[f"{s}"]["lumi_weight"] * data[f"{s}"]["plot_weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s] +
    #          #                        [data["SKIM_DY_amc_incl"]["lumi_weight"] * data["SKIM_DY_amc_incl"]["plot_weight"]])
    #          )
    #
    # plot_ROC(outputdir,
    #          "ROC_classHH_HHvsTTsl",
    #          r"HH class (HH vs $t\overline{t}$ sl)",
    #          np.concatenate([[1]*len(data[f"{s}"]) for s in samples.keys() if "ggF" in s or "GGHH" in s] + [[0]*len(data["SKIM_TT_semiLep"])]),
    #          np.concatenate([data[f"{s}"]["class_HH"] for s in samples.keys() if "ggF" in s or "GGHH" in s] + [data["SKIM_TT_semiLep"]["class_HH"]]),
    #          weights=np.concatenate([data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s] +
    #                                 [data["SKIM_TT_semiLep"]["weight"]])
    #          # weights=np.concatenate([data[f"{s}"]["lumi_weight"] * data[f"{s}"]["plot_weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s] +
    #          #                        [data["SKIM_DY_amc_incl"]["lumi_weight"] * data["SKIM_DY_amc_incl"]["plot_weight"]])
    #          )
    #
    # plot_ROC(outputdir,
    #          "ROC_classHH_HHvsTTH",
    #          r"HH class (HH vs $t\overline{t}$H*)",
    #          np.concatenate([[1]*len(data[f"{s}"]) for s in samples.keys() if "ggF" in s or "GGHH" in s] + [[0]*len(data["SKIM_ttHToTauTau"])]),
    #          np.concatenate([data[f"{s}"]["class_HH"] for s in samples.keys() if "ggF" in s or "GGHH" in s] + [data["SKIM_ttHToTauTau"]["class_HH"]]),
    #          weights=np.concatenate([data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s] +
    #                                 [data["SKIM_ttHToTauTau"]["weight"]])
    #          # weights=np.concatenate([data[f"{s}"]["lumi_weight"] * data[f"{s}"]["plot_weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s] +
    #          #                        [data["SKIM_DY_amc_incl"]["lumi_weight"] * data["SKIM_DY_amc_incl"]["plot_weight"]])
    #          )
    #
    # exit()

    for var, xlabel, xmin, xmax in zip(["reg_H_m", "reg_H_pt", "tauH_SVFIT_mass", "tauH_SVFIT_pt", "tauH_mass", "tauH_pt", "recoGenTauH_mass", "recoGenTauH_pt", "bH_mass", "bH_pt", "reg_HH_m", "reg_HH_pt", "HH_mass", "HH_pt", "HHKin_mass", "svfit_HH_m", "class_HH", "class_DY", "class_TT", "class_H"],
                                       [r"$m_{\tau\tau,reg}$ [GeV]", r"$p_T^{\tau\tau,reg}$ [GeV]", r"$m_{\tau\tau,SVfit}$ [GeV]", r"$p_T^{\tau\tau,SVfit}$ [GeV]", r"$m_{\tau\tau,vis}$ [GeV]", r"$p_T^{\tau\tau,vis}$ [GeV]", r"$m_{\tau\tau,gen\ \nu}$ [GeV]", r"$p_T^{\tau\tau,gen\ \nu}$ [GeV]", r"$m_{bb,reco}$ [GeV]", r"$p_T^{bb,reco}$ [GeV]", r"$m_{HH,reg}$ [GeV]", r"$p_T^{HH,reg}$ [GeV]", r"$m_{HH,reco}$ [GeV]", r"$p_T^{HH,reco}$ [GeV]", r"$m_{HH,kinfit}$ [GeV]", r"$m_{HH,SVfit}$ [GeV]",
                                        "Classifier HH output", "Classifier DY output", "Classifier TT output"],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [200, 800, 200, 800, 200, 800, 200, 800, 200, 800, 2000, 800, 2000, 800, 2000, 2000, 1, 1, 1]):

        # for var, xlabel in zip(["reg_H_m",],
        #                        [r"$m_{\tau\tau,reg}$ [GeV]"]):
        plot_1Dhist(outputdir,
                    f"{var}",
                    [
                        [data[f"{s}"][f"{var}"] for s in samples.keys() if "ggF" in s or "GGHH" in s],
                        # [data["SKIM_GluGluHToTauTau"][f"{var}"]],
                        [data["SKIM_DY_amc_incl"][f"{var}"]],
                        [data["SKIM_TT_fullyLep"][f"{var}"]],
                        [data["SKIM_TT_semiLep"][f"{var}"]],
                        [data["SKIM_ttHToTauTau"][f"{var}"]]
                    ],
                    weights=[
                        [data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s],
                        # [data["SKIM_GluGluHToTauTau"]["weight"]],
                        [data["SKIM_DY_amc_incl"]["weight"]],
                        [data["SKIM_TT_fullyLep"]["weight"]],
                        [data["SKIM_TT_semiLep"]["weight"]],
                        [data["SKIM_ttHToTauTau"]["weight"]]
                    ],
                    labels=[
                        "HH",
                        # "H*",
                        "DY",
                        r"$t\overline{t}$ dl",
                        r"$t\overline{t}$ sl*",
                        r"$t\overline{t}$H*"
                    ],
                    nbins=50,
                    xmin=xmin,
                    xmax=xmax,
                    xlabel=xlabel,
                    normalized=True,
                    )

    for var, xlabel, xmin, xmax in zip(["reg_H_m", "reg_H_pt", "tauH_SVFIT_mass", "tauH_SVFIT_pt", "tauH_mass", "tauH_pt", "recoGenTauH_mass", "recoGenTauH_pt", "bH_mass", "bH_pt", "reg_HH_m", "reg_HH_pt", "HH_mass", "HH_pt", "HHKin_mass", "svfit_HH_m", "class_HH", "class_DY", "class_TT", "class_H"],
                                       [r"$m_{\tau\tau,reg}$ [GeV]", r"$p_T^{\tau\tau,reg}$ [GeV]", r"$m_{\tau\tau,SVfit}$ [GeV]", r"$p_T^{\tau\tau,SVfit}$ [GeV]", r"$m_{\tau\tau,vis}$ [GeV]", r"$p_T^{\tau\tau,vis}$ [GeV]", r"$m_{\tau\tau,gen\ \nu}$ [GeV]", r"$p_T^{\tau\tau,gen\ \nu}$ [GeV]", r"$m_{bb,reco}$ [GeV]", r"$p_T^{bb,reco}$ [GeV]", r"$m_{HH,reg}$ [GeV]", r"$p_T^{HH,reg}$ [GeV]", r"$m_{HH,reco}$ [GeV]", r"$p_T^{HH,reco}$ [GeV]", r"$m_{HH,kinfit}$ [GeV]", r"$m_{HH,SVfit}$ [GeV]",
                                        "Classifier HH output", "Classifier DY output", "Classifier TT output"],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [200, 800, 200, 800, 200, 800, 200, 800, 200, 800, 2000, 800, 2000, 800, 2000, 2000, 1, 1, 1]):
        plot_1Dhist(outputdir + "/signals",
                    f"{var}",
                    [[data["SKIM_GGHH_SM"][f"{var}"]], [data["SKIM_ggF_Radion_m300"][f"{var}"]], [data["SKIM_ggF_Radion_m700"][f"{var}"]], [data["SKIM_ggF_Radion_m1750"][f"{var}"]]],
                    weights=[[data["SKIM_GGHH_SM"]["weight"]], [data["SKIM_ggF_Radion_m300"]["weight"]], [data["SKIM_ggF_Radion_m700"]["weight"]], [data["SKIM_ggF_Radion_m1750"]["weight"]]],
                    labels=["HH SM", "HH m=300", "HH m=700", "HH m=1750"],
                    nbins=50, xmin=xmin, xmax=xmax,
                    xlabel=xlabel,
                    normalized=True)

    # plot_1Dhist(outputdir,
    #      "diff_to_truth_mass",
    #      [[data[f"{s}"]["recoGenTauH_mass"] - data[f"{s}"]["reg_H_m"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["recoGenTauH_mass"] - data["SKIM_GluGluHToTauTau"]["reg_H_m"]], [data["SKIM_DY_amc_incl"]["recoGenTauH_mass"] - data["SKIM_DY_amc_incl"]["reg_H_m"]],
    #       [data["SKIM_TT_fullyLep"]["recoGenTauH_mass"] - data["SKIM_TT_fullyLep"]["reg_H_m"]], [data["SKIM_TT_semiLep"]["recoGenTauH_mass"] - data["SKIM_TT_semiLep"]["reg_H_m"]]],
    #      weights=[[data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["weight"]], [data["SKIM_DY_amc_incl"]["weight"]],
    #               [data["SKIM_TT_fullyLep"]["weight"]], [data["SKIM_TT_semiLep"]["weight"]]],
    #      labels=["HH", "H", "DY", "tt_dl", "tt_sl*"],
    #      nbins=51,
    #      xmin=-100,
    #      xmax=100,
    #      xlabel=r"$m_{\tau\tau,gen}$ - $m_{\tau\tau,reg}$ [GeV]",
    #      normalized=True,
    #      )

    # for var in float_inputs + int_inputs + targets:
    #     print(f"Plotting variable {var}")
    #     plot_1Dhist(outputdir,
    #          f"{var}",
    #          [[data[f"{s}"][f"{var}"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_DY_NLO_incl"][f"{var}"]], [data["SKIM_TT_fullyLep"][f"{var}"]]],
    #          weights=[[data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_DY_NLO_incl"]["weight"]], [data["SKIM_TT_fullyLep"]["weight"]]],
    #          labels=["Signal", "DY", "tt_dl"],
    #          nbins=50,
    #          xmin=min([min(d[f"{var}"])for d in data.values()]),
    #          xmax=max([max(d[f"{var}"])for d in data.values()]),
    #          xlabel=f"{var}",
    #          normalized=True,
    #          )


def plot_1Dhist(outputdir, title, hists, weights=None, labels=["reco", "gen", "reg", "svfit"], nbins=10, xmin=None, xmax=None, xlabel="di tau mass [GeV]", ylabel="Events", log=False, normalized=True, legend=True, legend_loc="best"):
    hep.cms.text("Simulation, private work")
    nphists = False
    if isinstance(hists, list):
        for idx, hist in enumerate(hists):
            if isinstance(hist, list):
                if xmin is None or xmax is None:
                    print("xmin, xmax range must be specified when trying to add lists of hists")
                    quit()
                nphists = True
                for i, h in enumerate(hist):
                    if i == 0:
                        histsum, bins = np.histogram(h, bins=nbins, range=(xmin, xmax), weights=weights[idx][i])
                    else:
                        histsum += np.histogram(h, bins=nbins, range=(xmin, xmax), weights=weights[idx][i])[0]
                if normalized:
                    histsum /= sum(histsum) * np.diff(bins)
                    # histsum *= 100
                hists[idx] = histsum

    if nphists:
        ymin = min([min(h[np.nonzero(h)]) for h in hists])
        ymax = max([max(h) for h in hists]) * 1.1
        for hist, label in zip(hists, labels):
            plt.stairs(hist, bins, label=label, linewidth=3)
    else:
        opts = {"weights": weights, "histtype": "step", "density": normalized}
        means = [np.mean(hist) for hist in hists]
        stds = [np.std(hist) for hist in hists]
        quantiles = [(np.quantile(hist, 0.84) - np.quantile(hist, 0.16)) / 2 for hist in hists]
        labels = [l + r", $\mu$=" + f"{m:.1f}" + r", $\sigma_{std}$=" + f"{s:.1f}" + r", $\sigma_{68\%}$=" + f"{q:.1f}" for l, m, s, q in zip(labels, means, stds, quantiles)]
        # labels = [l + r", $\mu$=" + f"{m:.1f}" + r", $\sigma_{68\%}$=" + f"{q:.1f}" for l, m, q in zip(labels, means, quantiles)]
        plt.hist(hists, bins=nbins, label=labels, **opts)
        ymin = plt.ylim[0]
        ymax = plt.ylim[1]

    plt.ylim([0, ymax])
    # plt.title(title)
    plt.xlabel(xlabel)
    # if log:
    #     plt.yscale('log')
    if normalized:
        plt.ylabel(ylabel + " (normalized) [a.u.]")
    else:
        plt.ylabel(ylabel)
    if legend:
        plt.legend(loc=legend_loc)
    if xmin is not None and xmax is not None:
        plt.xlim([xmin, xmax])

    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1)
    Path(outputdir).mkdir(parents=True, exist_ok=True)
    plt.savefig(outputdir + "/" + title + ".pdf")
    plt.yscale('log')
    plt.ylim([ymin/10, ymax*10])
    plt.savefig(outputdir + "/" + title + "_log.pdf")
    plt.clf()


def plot_ROC(outputdir, title, label, y_true, y_score,  weights=None):
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=weights)
    plt.plot(fpr, tpr,  label=f"{label} (AUC={roc_auc_score(y_true, y_score):.3f})")
    plt.xlim([0, 1])
    plt.xlabel("False positive rate")
    plt.ylim([0, 1])
    plt.ylabel("True positive rate")
    plt.legend()
    Path(outputdir).mkdir(parents=True, exist_ok=True)
    plt.savefig(outputdir + "/" + title + ".pdf")
    plt.clf()


if __name__ == "__main__":
    main()
