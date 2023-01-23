#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
from util import load_sample, phi_mpi_to_pi, split_train_validation_mask, calc_new_columns


def main(basepath="/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v2_31Aug22",
         outputdir="plots",
         samples={
             "SKIM_GGHH_SM": 1.0,
             "SKIM_GluGluHToTauTau": 1.0,
             "SKIM_DY_NLO_incl": 1.0,
             "SKIM_TT_semiLep": 1.0,
             "SKIM_TT_fullyLep": 1.0,
         },
         columns_to_read=[
             "dau1_pt", "dau1_eta", "dau1_phi", "dau1_e", "dau1_dxy", "dau1_dz", "dau1_iso",
             "dau2_pt", "dau2_eta", "dau2_phi", "dau2_e", "dau2_dxy", "dau2_dz", "dau2_iso",
             "met_et", "met_phi", "met_cov00", "met_cov01", "met_cov11",
             "bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_e", "bjet1_btag_deepFlavor",
             "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_e", "bjet2_btag_deepFlavor",
             "tauH_mass",
             "pairType", "dau1_decayMode", "dau2_decayMode",
             # "matchedGenLepton1_pt", "matchedGenLepton2_pt",
             # "genNu1_pt", "genNu1_eta", "genNu1_phi", "genNu1_e",
             # "genNu2_pt", "genNu2_eta", "genNu2_phi", "genNu2_e",
             "tauH_SVFIT_mass",
             "matchedGenLepton1_pt", "matchedGenLepton1_eta", "matchedGenLepton1_phi", "matchedGenLepton1_e",
             "matchedGenLepton2_pt", "matchedGenLepton2_eta", "matchedGenLepton2_phi", "matchedGenLepton2_e",
         ],
         columns_to_add={
             "dau1_met_dphi": (("dau1_phi", "met_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
             "dau2_met_dphi": (("dau2_phi", "met_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
             # "genNu1_met_dphi": (("genNu1_phi", "met_phi"),(lambda a,b: phi_mpi_to_pi(a-b))),
             # "genNu2_met_dphi": (("genNu2_phi", "met_phi"),(lambda a,b: phi_mpi_to_pi(a-b))),
             # "dau1_pt_factor": (("dau1_pt", "matchedGenLepton1_pt"),(lambda a,b: a/b)),
             # "dau2_pt_factor": (("dau2_pt", "matchedGenLepton2_pt"),(lambda a,b: a/b)),
             "dau1_px": (("dau1_pt", "dau1_met_dphi"), (lambda a, b: a * np.cos(b))),
             "dau1_py": (("dau1_pt", "dau1_met_dphi"), (lambda a, b: a * np.sin(b))),
             "dau1_pz": (("dau1_pt", "dau1_eta"), (lambda a, b: a * np.sinh(b))),
             "dau1_m": (("dau1_px", "dau1_py", "dau1_pz", "dau1_e"), (lambda x, y, z, e: np.sqrt(e ** 2 - (x ** 2 + y ** 2 + z ** 2)))),
             # "dau1_mt": (("dau1_pz", "dau1_e"), (lambda z,e: np.sqrt(e**2-z**2))),
             "dau2_px": (("dau2_pt", "dau2_met_dphi"), (lambda a, b: a * np.cos(b))),
             "dau2_py": (("dau2_pt", "dau2_met_dphi"), (lambda a, b: a * np.sin(b))),
             "dau2_pz": (("dau2_pt", "dau2_eta"), (lambda a, b: a * np.sinh(b))),
             "dau2_m": (("dau2_px", "dau2_py", "dau2_pz", "dau2_e"), (lambda x, y, z, e: np.sqrt(e ** 2 - (x ** 2 + y ** 2 + z ** 2)))),
             # "dau2_mt": (("dau2_pz", "dau2_e"), (lambda z,e: np.sqrt(e**2-z**2))),
             "ditau_deltaphi": (("dau1_met_dphi", "dau2_met_dphi"), (lambda a, b: np.abs(phi_mpi_to_pi(a - b)))),
             "ditau_deltaeta": (("dau1_eta", "dau2_eta"), (lambda a, b: np.abs(a - b))),
             # "genNu1_px": (("genNu1_pt", "genNu1_met_dphi"), (lambda a,b: a * np.cos(b))),
             # "genNu1_py": (("genNu1_pt", "genNu1_met_dphi"), (lambda a,b: a * np.sin(b))),
             # "genNu1_pz": (("genNu1_pt", "genNu1_eta"), (lambda a,b: a * np.sinh(b))),
             # "genNu2_px": (("genNu2_pt", "genNu2_met_dphi"), (lambda a,b: a * np.cos(b))),
             # "genNu2_py": (("genNu2_pt", "genNu2_met_dphi"), (lambda a,b: a * np.sin(b))),
             # "genNu2_pz": (("genNu2_pt", "genNu2_eta"), (lambda a,b: a * np.sinh(b))),
             # "met_px": (("met_et", "met_phi"), (lambda a,b: a * np.cos(b))),
             # "met_py": (("met_et", "met_phi"), (lambda a,b: a * np.sin(b))),
             # "mT_tau1": (("dau1_mt", "dau1_px", "dau1_py", "met_et"), (lambda e1, x1, y1, e2: np.sqrt((e1+e2)**2-(x1+e2)**2-(y1+0)**2))),
             # "mT_tau2": (("dau2_mt", "dau2_px", "dau2_py", "met_et"), (lambda e1, x1, y1, e2: np.sqrt((e1+e2)**2-(x1+e2)**2-(y1+0)**2))),
             # "mT_tautau": (("dau1_mt", "dau1_px", "dau1_py", "dau2_mt", "dau2_px", "dau2_py", "met_et"), (lambda e1, x1, y1, e2, x2, y2, e3: np.sqrt((e1+e2+e3)**2-(x1+x2+e3)**2-(y1+y2+0)**2))),
             "bjet1_met_dphi": (("bjet1_phi", "met_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
             "bjet1_px": (("bjet1_pt", "bjet1_met_dphi"), (lambda a, b: a * np.cos(b))),
             "bjet1_py": (("bjet1_pt", "bjet1_met_dphi"), (lambda a, b: a * np.sin(b))),
             "bjet1_pz": (("bjet1_pt", "bjet1_eta"), (lambda a, b: a * np.sinh(b))),
             "bjet2_met_dphi": (("bjet2_phi", "met_phi"), (lambda a, b: phi_mpi_to_pi(a - b))),
             "bjet2_px": (("bjet2_pt", "bjet2_met_dphi"), (lambda a, b: a * np.cos(b))),
             "bjet2_py": (("bjet2_pt", "bjet2_met_dphi"), (lambda a, b: a * np.sin(b))),
             "bjet2_pz": (("bjet2_pt", "bjet2_eta"), (lambda a, b: a * np.sinh(b))),
         },
         float_inputs=[
             # "dau1_pt", "dau1_eta", "dau1_phi", "dau1_e", "dau1_dxy", "dau1_dz", "dau1_iso",
             # "dau2_pt", "dau2_eta", "dau2_phi", "dau2_e", "dau2_dxy", "dau2_dz", "dau2_iso",
             "met_et",  # "met_phi", "met_cov00", "met_cov01", "met_cov11",
             "ditau_deltaphi", "ditau_deltaeta",
             "dau1_px", "dau1_py", "dau1_pz", "dau1_e", "dau1_dxy", "dau1_dz", "dau1_iso",
             "dau2_px", "dau2_py", "dau2_pz", "dau2_e", "dau2_dxy", "dau2_dz", "dau2_iso",
             # "met_px", "met_py",
             "met_cov00", "met_cov01", "met_cov11",
             "bjet1_pt", "bjet1_eta", "bjet1_phi", "bjet1_e", "bjet1_btag_deepFlavor",
             "bjet2_pt", "bjet2_eta", "bjet2_phi", "bjet2_e", "bjet2_btag_deepFlavor",
             # "tauH_mass",
         ],
         int_inputs=[
             "pairType", "dau1_decayMode", "dau2_decayMode"
         ],
         selections=[(("pairType",), (lambda a: a < 3)), ],
         modelpath="models/best_model",
         train_valid_fraction=0.75,
         train_valid_seed=0,
         ):

    model = tf.keras.models.load_model(modelpath, compile=False)

    data = {}

    for sample, weight in samples.items():

        d, weights = load_sample(basepath, sample, weight, columns_to_read, selections)

        train_mask = split_train_validation_mask(len(weights), fraction=train_valid_fraction, seed=train_valid_seed)

        d = calc_new_columns(d[~train_mask], columns_to_add)

        float_input_valid = d[float_inputs]
        int_input_valid = d[int_inputs]

        float_input_valid = float_input_valid.astype([(name, np.float32) for name in float_input_valid.dtype.names], copy=False).view(np.float32).reshape((-1, len(float_input_valid.dtype)))
        int_input_valid = int_input_valid.astype([(name, np.float32) for name in int_input_valid.dtype.names], copy=False).view(np.float32).reshape((-1, len(int_input_valid.dtype)))

        predictions = model.predict([float_input_valid, int_input_valid])
        d = rfn.rec_append_fields(d, ["reg_nu1_px", "reg_nu1_py", "reg_nu1_pz", "reg_nu1_e", "reg_nu2_px", "reg_nu2_py", "reg_nu2_pz", "reg_nu2_e"], [
                                  predictions[1][:, i] for i in range(predictions[1].shape[1])], dtypes=["<f4"] * predictions[1].shape[1])

        reg_H = {"reg_H_e": (("dau1_e", "reg_nu1_e", "dau2_e", "reg_nu2_e"), (lambda a, b, c, d: a + b + c + d)),
                 "reg_H_px": (("dau1_px", "reg_nu1_px", "dau2_px", "reg_nu2_px"), (lambda a, b, c, d: a + b + c + d)),
                 "reg_H_py": (("dau1_py", "reg_nu1_py", "dau2_py", "reg_nu2_py"), (lambda a, b, c, d: a + b + c + d)),
                 "reg_H_pz": (("dau1_pz", "reg_nu1_pz", "dau2_pz", "reg_nu2_pz"), (lambda a, b, c, d: a + b + c + d)),
                 "reg_H_m": (("reg_H_e", "reg_H_px", "reg_H_py", "reg_H_pz"), (lambda a, b, c, d: np.sqrt(a ** 2 - b ** 2 - c ** 2 - d ** 2))),
                 }

        d = calc_new_columns(d, reg_H)
        data[sample] = d

    plot(outputdir,
         "reg_h_mass_test",
         [data["SKIM_GGHH_SM"]["reg_H_m"], data["SKIM_GluGluHToTauTau"]["reg_H_m"], data["SKIM_DY_NLO_incl"]["reg_H_m"], data["SKIM_TT_semiLep"]["reg_H_m"], data["SKIM_TT_fullyLep"]["reg_H_m"]],
         labels=["hh", "h", "dy", "tt_sl", "tt_dl"],
         nbins=50, xmin=0, xmax=200,
         xlabel="regressed ditau mass [GeV]", ylabel="Events",
         log=False, normalized=True)


def plot(outputdir, title, hists, labels=["reco", "gen", "reg", "svfit"], nbins=50, xmin=0, xmax=200, xlabel="di tau mass [GeV]", ylabel="Events", log=False, normalized=True):
    opts = {"histtype": "step", "bins": nbins, "range": [
        xmin, xmax], "log": log, "density": normalized}
    means = [np.mean(hist) for hist in hists]
    stds = [np.std(hist) for hist in hists]
    quantiles = [(np.quantile(hist, 0.84) - np.quantile(hist, 0.16)) / 2 for hist in hists]
    labels = [l + r", $\mu$=" + f"{m:.2f}" + r", $\sigma_{std}$=" + f"{s:.2f}" + r", $\sigma_{68\%}$=" + f"{q:.2f}" for l, m, s, q in zip(labels, means, stds, quantiles)]
    plt.hist(hists, **opts, label=labels)
    plt.title(title)
    plt.xlabel(xlabel)
    if normalized:
        plt.ylabel(ylabel + " (normalized)")
    else:
        plt.ylabel(ylabel)
    plt.legend()
    plt.xlim([xmin, xmax])
    plt.ylim([plt.ylim()[0], plt.ylim()[1] * 1.33])
    plt.savefig(outputdir + "/" + title + ".pdf")
    plt.clf()


if __name__ == "__main__":
    main()
