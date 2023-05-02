#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import mplhep as hep
from pathlib import Path
from util import load_sample, phi_mpi_to_pi, split_train_validation_mask, calc_new_columns


def main(
        basepath="/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v4_02Mar23",
        # basepath="/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v2_forTim",
        outputdir="plots/no_singleH_add_bjetvars/",
        modelpath="models/no_singleH_add_bjetvars/",
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
            # "SKIM_DY_NLO_incl": (1., 1.),
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
        # target_names=[
        #     # "dau1_pt_factor",
    #     # "dau2_pt_factor",
    #     "genNu1_px", "genNu1_py", "genNu1_pz", "genNu1_e",
    #     "genNu2_px", "genNu2_py", "genNu2_pz", "genNu2_e",
    # ],
    selections=[(("pairType",), (lambda a: a < 3)),
                (("nbjetscand",), (lambda a: a > 1)),
                # (("genLeptons_matched",), (lambda a: a == 1)),
                (("nleps",), (lambda a: a == 0)),
                ],
    train_valid_fraction=0.75,
    train_valid_seed=0,
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
            train_mask = split_train_validation_mask(nevents, fraction=train_valid_fraction, seed=train_valid_seed)
            d = calc_new_columns(d[train_mask if plot_only == "train" else ~train_mask], columns_to_add)
        else:
            d = calc_new_columns(d, columns_to_add)

        inputs = d[input_names]
        cat_inputs = d[cat_input_names]

        inputs = inputs.astype([(name, np.float32) for name in inputs.dtype.names], copy=False).view(np.float32).reshape((-1, len(inputs.dtype)))
        cat_inputs = cat_inputs.astype([(name, np.float32) for name in cat_inputs.dtype.names], copy=False).view(np.float32).reshape((-1, len(cat_inputs.dtype)))

        predictions = model.predict([inputs, cat_inputs])
        # d = rfn.rec_append_fields(d, ["reg_nu1_px", "reg_nu1_py", "reg_nu1_pz", "reg_nu1_e", "reg_nu2_px", "reg_nu2_py", "reg_nu2_pz", "reg_nu2_e"], [
        #                           predictions[1][:, i] for i in range(predictions[1].shape[1])], dtypes=["<f4"] * predictions[1].shape[1])

        d = rfn.rec_append_fields(d, ["reg_nu1_px", "reg_nu1_py", "reg_nu1_pz", "reg_nu2_px", "reg_nu2_py", "reg_nu2_pz"], [
                                  predictions[1][:, i] for i in range(predictions[1].shape[1])], dtypes=["<f4"] * predictions[1].shape[1])

        # d = rfn.rec_append_fields(d, ["reg_tau1_px", "reg_tau1_py", "reg_tau1_pz", "reg_tau2_px", "reg_tau2_py", "reg_tau2_pz"], [
        #                           predictions[1][:, i] for i in range(predictions[1].shape[1])], dtypes=["<f4"] * predictions[1].shape[1])

        reg_H = {
            # "reg_H_e": (("dau1_e", "reg_nu1_e", "dau2_e", "reg_nu2_e"), (lambda a, b, c, d: a + b + c + d)),
            "reg_H_e": (("dau1_e", "reg_nu1_px", "reg_nu1_py", "reg_nu1_pz", "dau2_e", "reg_nu2_px", "reg_nu2_py", "reg_nu2_pz"), (lambda a, b, c, d, e, f, g, h: a + np.sqrt(b**2+c**2+d**2) + e + np.sqrt(f**2+g**2+h**2))),
            "reg_H_px": (("dau1_px", "reg_nu1_px", "dau2_px", "reg_nu2_px"), (lambda a, b, c, d: a + b + c + d)),
            "reg_H_py": (("dau1_py", "reg_nu1_py", "dau2_py", "reg_nu2_py"), (lambda a, b, c, d: a + b + c + d)),
            "reg_H_pz": (("dau1_pz", "reg_nu1_pz", "dau2_pz", "reg_nu2_pz"), (lambda a, b, c, d: a + b + c + d)),
            "reg_H_m": (("reg_H_e", "reg_H_px", "reg_H_py", "reg_H_pz"), (lambda a, b, c, d: np.sqrt(a**2-b**2-c**2-d**2))),
            "reg_H_pt": (("reg_H_px", "reg_H_py"), (lambda a, b: np.sqrt(a**2 + b**2))),
            "reg_H_eta": (("reg_H_pt", "reg_H_pz"), (lambda a, b: np.arcsinh(b/a))),
            "reg_H_phi": (("reg_H_px", "reg_H_py"), (lambda a, b: np.arctan2(a, b))),
        }

        # reg_H = {
        #     "reg_H_e": (("reg_tau1_px", "reg_tau1_py", "reg_tau1_pz", "reg_tau2_px", "reg_tau2_py", "reg_tau2_pz"), (lambda a, b, c, d, e, f: np.sqrt(a**2 + b**2 + c**2 + 1.77686**2) + np.sqrt(d**2 + e**2 + f**2 + 1.77686**2))),
        #     "reg_H_px": (("reg_tau1_px", "reg_tau2_px"), (lambda a, b: a + b)),
        #     "reg_H_py": (("reg_tau1_py", "reg_tau2_py"), (lambda a, b: a + b)),
        #     "reg_H_pz": (("reg_tau1_pz", "reg_tau2_pz"), (lambda a, b: a + b)),
        #     "reg_H_m": (("reg_H_e", "reg_H_px", "reg_H_py", "reg_H_pz"), (lambda a, b, c, d: np.sqrt(a**2-b**2-c**2-d**2))),
        # }

        d = calc_new_columns(d, reg_H)
        data[sample] = d

    for var, xlabel, xmin, xmax in zip(["reg_H_m", "reg_H_pt", "reg_H_eta", "reg_H_phi", "tauH_SVFIT_mass", "tauH_SVFIT_pt", "tauH_SVFIT_eta", "tauH_SVFIT_phi", "tauH_mass", "tauH_pt", "tauH_eta", "tauH_phi", "recoGenTauH_mass", "recoGenTauH_pt", "recoGenTauH_eta", "recoGenTauH_phi"],
                                       [r"$m_{\tau\tau,reg}$ [GeV]", r"$p_T^{\tau\tau,reg}$ [GeV]", r"$\eta_{\tau\tau,reg}$", r"$\phi_{\tau\tau,reg}$", r"$m_{\tau\tau,SVfit}$ [GeV]", r"$p_T^{\tau\tau,SVfit}$ [GeV]", r"$\eta_{\tau\tau,SVfit}$", r"$\phi_{\tau\tau,SVfit}$",
                                        r"$m_{\tau\tau,vis}$ [GeV]", r"$p_T^{\tau\tau,vis}$ [GeV]", r"$\eta_{\tau\tau,vis}$", r"$\phi_{\tau\tau,vis}$", r"$m_{\tau\tau,gen\ \nu}$ [GeV]", r"$p_T^{\tau\tau,gen\ \nu}$ [GeV]", r"$\eta_{\tau\tau,gen\ \nu}$", r"$\phi_{\tau\tau,gen\ \nu}$"],
                                       [0, 0, -5, -3.5, 0, 0, -5, -3.5, 0, 0, -5, -3.5, 0, 0, -5, -3.5],
                                       [200, 800, 5, 3.5, 200, 800, 5, 3.5, 200, 800, 5, 3.5, 200, 800, 5, 3.5]):

        # for var, xlabel in zip(["reg_H_m",],
        #                        [r"$m_{\tau\tau,reg}$ [GeV]"]):
        plot(outputdir,
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
                 [data["SKIM_ttHToTauTau"]["weight"]]],
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

    # for var in ["reg_H_m", "tauH_SVFIT_mass", "tauH_mass", "recoGenTauH_mass"]:
    #     plot(outputdir,
    #          f"{var}",
    #          [[data[f"{s}"][f"{var}"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"][f"{var}"]], [data["SKIM_DY_NLO_incl"][f"{var}"]],
    #           [data["SKIM_TT_fullyLep"][f"{var}"]], [data["SKIM_TT_semiLep"][f"{var}"]]],
    #          weights=[[data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["weight"]], [data["SKIM_DY_NLO_incl"]["weight"]],
    #                   [data["SKIM_TT_fullyLep"]["weight"]], [data["SKIM_TT_semiLep"]["weight"]]],
    #          labels=["HH", "H*", "DY", "tt_dl", "tt_sl*"],
    #          nbins=50,
    #          xmin=0,
    #          xmax=200,
    #          xlabel=f"{var}",
    #          normalized=True,
    #          )
    #
    # plot(outputdir,
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
    #
    # plot(outputdir,
    #      "diff_to_truth_genNu1_px",
    #      [[data[f"{s}"]["genNu1_px"] - data[f"{s}"]["reg_nu1_px"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["genNu1_px"] - data["SKIM_GluGluHToTauTau"]["reg_nu1_px"]], [data["SKIM_DY_amc_incl"]["genNu1_px"] - data["SKIM_DY_amc_incl"]["reg_nu1_px"]],
    #       [data["SKIM_TT_fullyLep"]["genNu1_px"] - data["SKIM_TT_fullyLep"]["reg_nu1_px"]], [data["SKIM_TT_semiLep"]["genNu1_px"] - data["SKIM_TT_semiLep"]["reg_nu1_px"]]],
    #      weights=[[data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["weight"]], [data["SKIM_DY_amc_incl"]["weight"]],
    #               [data["SKIM_TT_fullyLep"]["weight"]], [data["SKIM_TT_semiLep"]["weight"]]],
    #      labels=["HH", "H", "DY", "tt_dl", "tt_sl*"],
    #      nbins=51,
    #      xmin=-100,
    #      xmax=100,
    #      xlabel=r"$p_x^{gen}(\nu_1)$ - $p_x^{reg}(\nu_1)$ [GeV]",
    #      normalized=True,
    #      )
    #
    # plot(outputdir,
    #      "diff_to_truth_genNu1_py",
    #      [[data[f"{s}"]["genNu1_py"] - data[f"{s}"]["reg_nu1_py"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["genNu1_py"] - data["SKIM_GluGluHToTauTau"]["reg_nu1_py"]], [data["SKIM_DY_amc_incl"]["genNu1_py"] - data["SKIM_DY_amc_incl"]["reg_nu1_py"]],
    #       [data["SKIM_TT_fullyLep"]["genNu1_py"] - data["SKIM_TT_fullyLep"]["reg_nu1_py"]], [data["SKIM_TT_semiLep"]["genNu1_py"] - data["SKIM_TT_semiLep"]["reg_nu1_py"]]],
    #      weights=[[data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["weight"]], [data["SKIM_DY_amc_incl"]["weight"]],
    #               [data["SKIM_TT_fullyLep"]["weight"]], [data["SKIM_TT_semiLep"]["weight"]]],
    #      labels=["HH", "H", "DY", "tt_dl", "tt_sl*"],
    #      nbins=51,
    #      xmin=-100,
    #      xmax=100,
    #      xlabel=r"$p_y^{gen}(\nu_1)$ - $p_y^{reg}(\nu_1)$ [GeV]",
    #      normalized=True,
    #      )
    #
    # plot(outputdir,
    #      "diff_to_truth_genNu1_pz",
    #      [[data[f"{s}"]["genNu1_pz"] - data[f"{s}"]["reg_nu1_pz"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["genNu1_pz"] - data["SKIM_GluGluHToTauTau"]["reg_nu1_pz"]], [data["SKIM_DY_amc_incl"]["genNu1_pz"] - data["SKIM_DY_amc_incl"]["reg_nu1_pz"]],
    #       [data["SKIM_TT_fullyLep"]["genNu1_pz"] - data["SKIM_TT_fullyLep"]["reg_nu1_pz"]], [data["SKIM_TT_semiLep"]["genNu1_pz"] - data["SKIM_TT_semiLep"]["reg_nu1_pz"]]],
    #      weights=[[data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["weight"]], [data["SKIM_DY_amc_incl"]["weight"]],
    #               [data["SKIM_TT_fullyLep"]["weight"]], [data["SKIM_TT_semiLep"]["weight"]]],
    #      labels=["HH", "H", "DY", "tt_dl", "tt_sl*"],
    #      nbins=51,
    #      xmin=-100,
    #      xmax=100,
    #      xlabel=r"$p_z^{gen}(\nu_1)$ - $p_z^{reg}(\nu_1)$ [GeV]",
    #      normalized=True,
    #      )
    #
    # plot(outputdir,
    #      "diff_to_truth_genNu2_px",
    #      [[data[f"{s}"]["genNu2_px"] - data[f"{s}"]["reg_nu2_px"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["genNu2_px"] - data["SKIM_GluGluHToTauTau"]["reg_nu2_px"]], [data["SKIM_DY_amc_incl"]["genNu2_px"] - data["SKIM_DY_amc_incl"]["reg_nu2_px"]],
    #       [data["SKIM_TT_fullyLep"]["genNu2_px"] - data["SKIM_TT_fullyLep"]["reg_nu2_px"]], [data["SKIM_TT_semiLep"]["genNu2_px"] - data["SKIM_TT_semiLep"]["reg_nu2_px"]]],
    #      weights=[[data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["weight"]], [data["SKIM_DY_amc_incl"]["weight"]],
    #               [data["SKIM_TT_fullyLep"]["weight"]], [data["SKIM_TT_semiLep"]["weight"]]],
    #      labels=["HH", "H", "DY", "tt_dl", "tt_sl*"],
    #      nbins=51,
    #      xmin=-100,
    #      xmax=100,
    #      xlabel=r"$p_x^{gen}(\nu_2)$ - $p_x^{reg}(\nu_2)$ [GeV]",
    #      normalized=True,
    #      )
    #
    # plot(outputdir,
    #      "diff_to_truth_genNu2_py",
    #      [[data[f"{s}"]["genNu2_py"] - data[f"{s}"]["reg_nu2_py"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["genNu2_py"] - data["SKIM_GluGluHToTauTau"]["reg_nu2_py"]], [data["SKIM_DY_amc_incl"]["genNu2_py"] - data["SKIM_DY_amc_incl"]["reg_nu2_py"]],
    #       [data["SKIM_TT_fullyLep"]["genNu2_py"] - data["SKIM_TT_fullyLep"]["reg_nu2_py"]], [data["SKIM_TT_semiLep"]["genNu2_py"] - data["SKIM_TT_semiLep"]["reg_nu2_py"]]],
    #      weights=[[data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["weight"]], [data["SKIM_DY_amc_incl"]["weight"]],
    #               [data["SKIM_TT_fullyLep"]["weight"]], [data["SKIM_TT_semiLep"]["weight"]]],
    #      labels=["HH", "H", "DY", "tt_dl", "tt_sl*"],
    #      nbins=51,
    #      xmin=-100,
    #      xmax=100,
    #      xlabel=r"$p_y^{gen}(\nu_2)$ - $p_y^{reg}(\nu_2)$ [GeV]",
    #      normalized=True,
    #      )
    #
    # plot(outputdir,
    #      "diff_to_truth_genNu2_pz",
    #      [[data[f"{s}"]["genNu2_pz"] - data[f"{s}"]["reg_nu2_pz"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["genNu2_pz"] - data["SKIM_GluGluHToTauTau"]["reg_nu2_pz"]], [data["SKIM_DY_amc_incl"]["genNu2_pz"] - data["SKIM_DY_amc_incl"]["reg_nu2_pz"]],
    #       [data["SKIM_TT_fullyLep"]["genNu2_pz"] - data["SKIM_TT_fullyLep"]["reg_nu2_pz"]], [data["SKIM_TT_semiLep"]["genNu2_pz"] - data["SKIM_TT_semiLep"]["reg_nu2_pz"]]],
    #      weights=[[data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["weight"]], [data["SKIM_DY_amc_incl"]["weight"]],
    #               [data["SKIM_TT_fullyLep"]["weight"]], [data["SKIM_TT_semiLep"]["weight"]]],
    #      labels=["HH", "H", "DY", "tt_dl", "tt_sl*"],
    #      nbins=51,
    #      xmin=-100,
    #      xmax=100,
    #      xlabel=r"$p_z^{gen}(\nu_2)$ - $p_z^{reg}(\nu_2)$ [GeV]",
    #      normalized=True,
    #      )
    #
    # plot(outputdir,
    #      "diff_to_truth_mass_norm",
    #      [[(data[f"{s}"]["recoGenTauH_mass"] - data[f"{s}"]["reg_H_m"])/data[f"{s}"]["recoGenTauH_mass"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [(data["SKIM_GluGluHToTauTau"]["recoGenTauH_mass"] - data["SKIM_GluGluHToTauTau"]["reg_H_m"])/data["SKIM_GluGluHToTauTau"]["recoGenTauH_mass"]], [(data["SKIM_DY_amc_incl"]["recoGenTauH_mass"] - data["SKIM_DY_amc_incl"]["reg_H_m"])/data["SKIM_DY_amc_incl"]["recoGenTauH_mass"]],
    #       [(data["SKIM_TT_fullyLep"]["recoGenTauH_mass"] - data["SKIM_TT_fullyLep"]["reg_H_m"])/data["SKIM_TT_fullyLep"]["recoGenTauH_mass"]], [(data["SKIM_TT_semiLep"]["recoGenTauH_mass"] - data["SKIM_TT_semiLep"]["reg_H_m"])/data["SKIM_TT_semiLep"]["recoGenTauH_mass"]]],
    #      weights=[[data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["weight"]], [data["SKIM_DY_amc_incl"]["weight"]],
    #               [data["SKIM_TT_fullyLep"]["weight"]], [data["SKIM_TT_semiLep"]["weight"]]],
    #      labels=["HH", "H", "DY", "tt_dl", "tt_sl*"],
    #      nbins=51,
    #      xmin=-1,
    #      xmax=1,
    #      xlabel=r"($m_{\tau\tau,gen}$ - $m_{\tau\tau,reg}$)/$m_{\tau\tau,gen}$",
    #      normalized=True,
    #      )
    #
    # plot(outputdir,
    #      "diff_to_truth_genNu1_px_norm",
    #      [[(data[f"{s}"]["genNu1_px"] - data[f"{s}"]["reg_nu1_px"])/data[f"{s}"]["genNu1_px"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [(data["SKIM_GluGluHToTauTau"]["genNu1_px"] - data["SKIM_GluGluHToTauTau"]["reg_nu1_px"])/data["SKIM_GluGluHToTauTau"]["genNu1_px"]], [(data["SKIM_DY_amc_incl"]["genNu1_px"] - data["SKIM_DY_amc_incl"]["reg_nu1_px"])/data["SKIM_DY_amc_incl"]["genNu1_px"]],
    #       [(data["SKIM_TT_fullyLep"]["genNu1_px"] - data["SKIM_TT_fullyLep"]["reg_nu1_px"])/data["SKIM_TT_fullyLep"]["genNu1_px"]], [(data["SKIM_TT_semiLep"]["genNu1_px"] - data["SKIM_TT_semiLep"]["reg_nu1_px"])/data["SKIM_TT_semiLep"]["genNu1_px"]]],
    #      weights=[[data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["weight"]], [data["SKIM_DY_amc_incl"]["weight"]],
    #               [data["SKIM_TT_fullyLep"]["weight"]], [data["SKIM_TT_semiLep"]["weight"]]],
    #      labels=["HH", "H", "DY", "tt_dl", "tt_sl*"],
    #      nbins=51,
    #      xmin=-1,
    #      xmax=1,
    #      xlabel=r"($p_x^{gen}(\nu_1)$ - $p_x^{reg}(\nu_1)$)/$p_x^{gen}(\nu_1)$",
    #      normalized=True,
    #      )
    #
    # plot(outputdir,
    #      "diff_to_truth_genNu1_py_norm",
    #      [[(data[f"{s}"]["genNu1_py"] - data[f"{s}"]["reg_nu1_py"])/data[f"{s}"]["genNu1_py"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [(data["SKIM_GluGluHToTauTau"]["genNu1_py"] - data["SKIM_GluGluHToTauTau"]["reg_nu1_py"])/data["SKIM_GluGluHToTauTau"]["genNu1_py"]], [(data["SKIM_DY_amc_incl"]["genNu1_py"] - data["SKIM_DY_amc_incl"]["reg_nu1_py"])/data["SKIM_DY_amc_incl"]["genNu1_py"]],
    #       [(data["SKIM_TT_fullyLep"]["genNu1_py"] - data["SKIM_TT_fullyLep"]["reg_nu1_py"])/data["SKIM_TT_fullyLep"]["genNu1_py"]], [(data["SKIM_TT_semiLep"]["genNu1_py"] - data["SKIM_TT_semiLep"]["reg_nu1_py"])/data["SKIM_TT_semiLep"]["genNu1_py"]]],
    #      weights=[[data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["weight"]], [data["SKIM_DY_amc_incl"]["weight"]],
    #               [data["SKIM_TT_fullyLep"]["weight"]], [data["SKIM_TT_semiLep"]["weight"]]],
    #      labels=["HH", "H", "DY", "tt_dl", "tt_sl*"],
    #      nbins=51,
    #      xmin=-1,
    #      xmax=1,
    #      xlabel=r"($p_y^{gen}(\nu_1)$ - $p_y^{reg}(\nu_1)$)/$p_y^{gen}(\nu_1)$",
    #      normalized=True,
    #      )
    #
    # plot(outputdir,
    #      "diff_to_truth_genNu1_pz_norm",
    #      [[(data[f"{s}"]["genNu1_pz"] - data[f"{s}"]["reg_nu1_pz"])/data[f"{s}"]["genNu1_pz"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [(data["SKIM_GluGluHToTauTau"]["genNu1_pz"] - data["SKIM_GluGluHToTauTau"]["reg_nu1_pz"])/data["SKIM_GluGluHToTauTau"]["genNu1_pz"]], [(data["SKIM_DY_amc_incl"]["genNu1_pz"] - data["SKIM_DY_amc_incl"]["reg_nu1_pz"])/data["SKIM_DY_amc_incl"]["genNu1_pz"]],
    #       [(data["SKIM_TT_fullyLep"]["genNu1_pz"] - data["SKIM_TT_fullyLep"]["reg_nu1_pz"])/data["SKIM_TT_fullyLep"]["genNu1_pz"]], [(data["SKIM_TT_semiLep"]["genNu1_pz"] - data["SKIM_TT_semiLep"]["reg_nu1_pz"])/data["SKIM_TT_semiLep"]["genNu1_pz"]]],
    #      weights=[[data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["weight"]], [data["SKIM_DY_amc_incl"]["weight"]],
    #               [data["SKIM_TT_fullyLep"]["weight"]], [data["SKIM_TT_semiLep"]["weight"]]],
    #      labels=["HH", "H", "DY", "tt_dl", "tt_sl*"],
    #      nbins=51,
    #      xmin=-1,
    #      xmax=1,
    #      xlabel=r"($p_z^{gen}(\nu_1)$ - $p_z^{reg}(\nu_1)$)/$p_z^{gen}(\nu_1)$",
    #      normalized=True,
    #      )
    #
    # plot(outputdir,
    #      "diff_to_truth_genNu2_px_norm",
    #      [[(data[f"{s}"]["genNu2_px"] - data[f"{s}"]["reg_nu2_px"])/data[f"{s}"]["genNu2_px"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [(data["SKIM_GluGluHToTauTau"]["genNu2_px"] - data["SKIM_GluGluHToTauTau"]["reg_nu2_px"])/data["SKIM_GluGluHToTauTau"]["genNu2_px"]], [(data["SKIM_DY_amc_incl"]["genNu2_px"] - data["SKIM_DY_amc_incl"]["reg_nu2_px"])/data["SKIM_DY_amc_incl"]["genNu2_px"]],
    #       [(data["SKIM_TT_fullyLep"]["genNu2_px"] - data["SKIM_TT_fullyLep"]["reg_nu2_px"])/data["SKIM_TT_fullyLep"]["genNu2_px"]], [(data["SKIM_TT_semiLep"]["genNu2_px"] - data["SKIM_TT_semiLep"]["reg_nu2_px"])/data["SKIM_TT_semiLep"]["genNu2_px"]]],
    #      weights=[[data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["weight"]], [data["SKIM_DY_amc_incl"]["weight"]],
    #               [data["SKIM_TT_fullyLep"]["weight"]], [data["SKIM_TT_semiLep"]["weight"]]],
    #      labels=["HH", "H", "DY", "tt_dl", "tt_sl*"],
    #      nbins=51,
    #      xmin=-1,
    #      xmax=1,
    #      xlabel=r"($p_x^{gen}(\nu_2)$ - $p_x^{reg}(\nu_2)$)/$p_x^{gen}(\nu_2)$",
    #      normalized=True,
    #      )
    #
    # plot(outputdir,
    #      "diff_to_truth_genNu2_py_norm",
    #      [[(data[f"{s}"]["genNu2_py"] - data[f"{s}"]["reg_nu2_py"])/data[f"{s}"]["genNu2_py"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [(data["SKIM_GluGluHToTauTau"]["genNu2_py"] - data["SKIM_GluGluHToTauTau"]["reg_nu2_py"])/data["SKIM_GluGluHToTauTau"]["genNu2_py"]], [(data["SKIM_DY_amc_incl"]["genNu2_py"] - data["SKIM_DY_amc_incl"]["reg_nu2_py"])/data["SKIM_DY_amc_incl"]["genNu2_py"]],
    #       [(data["SKIM_TT_fullyLep"]["genNu2_py"] - data["SKIM_TT_fullyLep"]["reg_nu2_py"])/data["SKIM_TT_fullyLep"]["genNu2_py"]], [(data["SKIM_TT_semiLep"]["genNu2_py"] - data["SKIM_TT_semiLep"]["reg_nu2_py"])/data["SKIM_TT_semiLep"]["genNu2_py"]]],
    #      weights=[[data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["weight"]], [data["SKIM_DY_amc_incl"]["weight"]],
    #               [data["SKIM_TT_fullyLep"]["weight"]], [data["SKIM_TT_semiLep"]["weight"]]],
    #      labels=["HH", "H", "DY", "tt_dl", "tt_sl*"],
    #      nbins=51,
    #      xmin=-1,
    #      xmax=1,
    #      xlabel=r"($p_y^{gen}(\nu_2)$ - $p_y^{reg}(\nu_2)$)/$p_y^{gen}(\nu_2)$",
    #      normalized=True,
    #      )
    #
    # plot(outputdir,
    #      "diff_to_truth_genNu2_pz_norm",
    #      [[(data[f"{s}"]["genNu2_pz"] - data[f"{s}"]["reg_nu2_pz"])/data[f"{s}"]["genNu2_pz"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [(data["SKIM_GluGluHToTauTau"]["genNu2_pz"] - data["SKIM_GluGluHToTauTau"]["reg_nu2_pz"])/data["SKIM_GluGluHToTauTau"]["genNu2_pz"]], [(data["SKIM_DY_amc_incl"]["genNu2_pz"] - data["SKIM_DY_amc_incl"]["reg_nu2_pz"])/data["SKIM_DY_amc_incl"]["genNu2_pz"]],
    #       [(data["SKIM_TT_fullyLep"]["genNu2_pz"] - data["SKIM_TT_fullyLep"]["reg_nu2_pz"])/data["SKIM_TT_fullyLep"]["genNu2_pz"]], [(data["SKIM_TT_semiLep"]["genNu2_pz"] - data["SKIM_TT_semiLep"]["reg_nu2_pz"])/data["SKIM_TT_semiLep"]["genNu2_pz"]]],
    #      weights=[[data[f"{s}"]["weight"] for s in samples.keys() if "ggF" in s or "GGHH" in s], [data["SKIM_GluGluHToTauTau"]["weight"]], [data["SKIM_DY_amc_incl"]["weight"]],
    #               [data["SKIM_TT_fullyLep"]["weight"]], [data["SKIM_TT_semiLep"]["weight"]]],
    #      labels=["HH", "H", "DY", "tt_dl", "tt_sl*"],
    #      nbins=51,
    #      xmin=-1,
    #      xmax=1,
    #      xlabel=r"($p_z^{gen}(\nu_2)$ - $p_z^{reg}(\nu_2)$)/$p_z^{gen}(\nu_2)$",
    #      normalized=True,
    #      )
    #
    # plot(outputdir,
    #      "reg_H_m_masses",
    #      [[data["SKIM_ggF_Radion_m300"]["reg_H_m"]], [data["SKIM_ggF_Radion_m700"]["reg_H_m"]], [data["SKIM_ggF_Radion_m1750"]["reg_H_m"]]],
    #      weights=[[data["SKIM_ggF_Radion_m300"]["weight"]], [data["SKIM_ggF_Radion_m700"]["weight"]], [data["SKIM_ggF_Radion_m1750"]["weight"]]],
    #      labels=["300", "700", "1750"],
    #      nbins=50,
    #      xmin=0,
    #      xmax=200,
    #      xlabel=r"$m_{\tau\tau,reg}$ [GeV]",
    #      normalized=True,
    #      legend=True,
    #      )

    # for var in float_inputs + int_inputs + targets:
    #     print(f"Plotting variable {var}")
    #     plot(outputdir,
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

    # plot(outputdir,
    #      "reg_h_mass",
    #      [data["SKIM_GGHH_SM"]["reg_H_m"], data["SKIM_GluGluHToTauTau"]["reg_H_m"], data["SKIM_DY_NLO_incl"]["reg_H_m"], data["SKIM_TT_semiLep"]["reg_H_m"], data["SKIM_TT_fullyLep"]["reg_H_m"]],
    #      labels=["hh", "h*", "dy", "tt_sl*", "tt_dl"],
    #      nbins=50, xmin=0, xmax=200,
    #      xlabel="regressed ditau mass [GeV]", ylabel="Events",
    #      log=False, normalized=True)

    # plot(outputdir,
    #      "reg_h_mass_signals",
    #      [data["SKIM_GGHH_SM"]["reg_H_m"], data["SKIM_ggF_Radion_m300"]["reg_H_m"], data["SKIM_ggF_Radion_m700"]["reg_H_m"], data["SKIM_ggF_Radion_m1750"]["reg_H_m"]],
    #      labels=["hh sm", "hh m300", "hh m700", "hh m1750"],
    #      nbins=50, xmin=0, xmax=200,
    #      xlabel="regressed ditau mass [GeV]", ylabel="Events",
    #      log=False, normalized=True)


def plot(outputdir, title, hists, weights=None, labels=["reco", "gen", "reg", "svfit"], nbins=10, xmin=None, xmax=None, xlabel="di tau mass [GeV]", ylabel="Events", log=False, normalized=True, legend=True, legend_loc="best"):
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


if __name__ == "__main__":
    main()
