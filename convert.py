#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np
import awkward as ak
import glob
import uproot4
from multiprocessing import Pool
from extract_klub_info import delta_r_from_names
from pathlib import Path


variables = ["pairType",
             "met_et",
             "met_phi",
             "met_cov00",
             "met_cov01",
             "met_cov11",
             "METx",
             "METy",
             "DeepMET_ResponseTune_px",
             "DeepMET_ResponseTune_py",
             "DeepMET_ResolutionTune_px",
             "DeepMET_ResolutionTune_py",
             "npv",
             "npu",
             "isOS",
             "nleps",
             "rho",
             "dau1_pt",
             "dau1_eta",
             "dau1_phi",
             "dau1_e",
             "dau1_dxy",
             "dau1_dz",
             "dau1_decayMode",
             "dau1_iso",
             "dau1_eleMVAiso",
             "dau1_MVAisoNew",
             "dau1_MVAisoNewdR0p3",
             "dau1_CUTiso",
             "dau1_deepTauVsJet",
             "dau1_deepTauVsEle",
             "dau1_deepTauVsMu",
             "dau2_pt",
             "dau2_eta",
             "dau2_phi",
             "dau2_e",
             "dau2_dxy",
             "dau2_dz",
             "dau2_decayMode",
             "dau2_iso",
             "dau2_MVAisoNew",
             "dau2_MVAisoNewdR0p3",
             "dau2_CUTiso",
             "dau2_deepTauVsJet",
             "dau2_deepTauVsEle",
             "dau2_deepTauVsMu",
             "nbjetscand",
             "bjet1_pt",
             "bjet1_eta",
             "bjet1_phi",
             "bjet1_e",
             "bjet1_btag_deepFlavor",
             # "bjet1_bID_deepFlavor",
             "bjet1_cID_deepFlavor",
             "bjet1_pnet_bb",
             "bjet1_pnet_cc",
             "bjet1_pnet_b",
             "bjet1_pnet_c",
             "bjet1_pnet_g",
             "bjet1_pnet_uds",
             "bjet1_pnet_pu",
             "bjet1_pnet_undef",
             "bjet1_HHbtag",
             "bjet1_PUjetIDupdated",
             "bjet2_pt",
             "bjet2_eta",
             "bjet2_phi",
             "bjet2_e",
             "bjet2_btag_deepFlavor",
             # "bjet2_bID_deepFlavor",
             "bjet2_cID_deepFlavor",
             "bjet2_pnet_bb",
             "bjet2_pnet_cc",
             "bjet2_pnet_b",
             "bjet2_pnet_c",
             "bjet2_pnet_g",
             "bjet2_pnet_uds",
             "bjet2_pnet_pu",
             "bjet2_pnet_undef",
             "bjet2_HHbtag",
             "bjet2_PUjetIDupdated",
             "VBFjet1_pt",
             "VBFjet1_eta",
             "VBFjet1_phi",
             "VBFjet1_e",
             "VBFjet1_btag_deepFlavor",
             "VBFjet1_ctag_deepFlavor",
             "VBFjet1_pnet_bb",
             "VBFjet1_pnet_cc",
             "VBFjet1_pnet_b",
             "VBFjet1_pnet_c",
             "VBFjet1_pnet_g",
             "VBFjet1_pnet_uds",
             "VBFjet1_pnet_pu",
             "VBFjet1_pnet_undef",
             "VBFjet1_HHbtag",
             "VBFjet1_PUjetIDupdated",
             "VBFjet2_pt",
             "VBFjet2_eta",
             "VBFjet2_phi",
             "VBFjet2_e",
             "VBFjet2_btag_deepFlavor",
             "VBFjet2_ctag_deepFlavor",
             "VBFjet2_pnet_bb",
             "VBFjet2_pnet_cc",
             "VBFjet2_pnet_b",
             "VBFjet2_pnet_c",
             "VBFjet2_pnet_g",
             "VBFjet2_pnet_uds",
             "VBFjet2_pnet_pu",
             "VBFjet2_pnet_undef",
             "VBFjet2_HHbtag",
             "VBFjet2_PUjetIDupdated",
             "tauH_pt",
             "tauH_eta",
             "tauH_phi",
             "tauH_e",
             "tauH_mass",
             "tauH_SVFIT_mass",
             "tauH_SVFIT_pt",
             "tauH_SVFIT_eta",
             "tauH_SVFIT_phi",
             "tauH_SVFIT_METphi",
             "tauH_SVFIT_METrho",
             "bH_pt",
             "bH_eta",
             "bH_phi",
             "bH_e",
             "bH_mass",
             "HH_pt",
             "HH_eta",
             "HH_phi",
             "HH_e",
             "HH_mass",
             "HH_mass_raw",
             "HH_deltaPhi",
             "HH_deltaR",
             "HH_deltaEta",
             "HHKin_mass",
             "HHKin_chi2",
             "HT20",
             "HT50",
             "HT20Full",
             "plot_weight",
             "lumi_weight",
             "genNu1_pt",
             "genNu1_eta",
             "genNu1_phi",
             "genNu1_e",
             "genNu2_pt",
             "genNu2_eta",
             "genNu2_phi",
             "genNu2_e",
             "genNuTot_pt",
             "genNuTot_eta",
             "genNuTot_phi",
             "genNuTot_e",
             "recoGenTauH_pt",
             "recoGenTauH_eta",
             "recoGenTauH_phi",
             "recoGenTauH_e",
             "recoGenTauH_mass",
             "genH1_pt",
             "genH1_eta",
             "genH1_phi",
             "genH1_e",
             "genH2_pt",
             "genH2_eta",
             "genH2_phi",
             "genH2_e",
             "genB1_pt",
             "genB1_eta",
             "genB1_phi",
             "genB1_e",
             "genB2_pt",
             "genB2_eta",
             "genB2_phi",
             "genB2_e",
             "genBQuark1_pt",
             "genBQuark1_eta",
             "genBQuark1_phi",
             "genBQuark1_e",
             "genBQuark1_motherPdgId",
             "genBQuark2_pt",
             "genBQuark2_eta",
             "genBQuark2_phi",
             "genBQuark2_e",
             "genBQuark2_motherPdgId",
             "genBQuark3_pt",
             "genBQuark3_eta",
             "genBQuark3_phi",
             "genBQuark3_e",
             "genBQuark3_motherPdgId",
             "genBQuark4_pt",
             "genBQuark4_eta",
             "genBQuark4_phi",
             "genBQuark4_e",
             "genBQuark4_motherPdgId",
             "genLepton1_pt",
             "genLepton1_eta",
             "genLepton1_phi",
             "genLepton1_e",
             "genLepton1_pdgId",
             "genLepton1_motherPdgId",
             "genLepton2_pt",
             "genLepton2_eta",
             "genLepton2_phi",
             "genLepton2_e",
             "genLepton2_pdgId",
             "genLepton2_motherPdgId",
             "genLepton3_pt",
             "genLepton3_eta",
             "genLepton3_phi",
             "genLepton3_e",
             "genLepton3_pdgId",
             "genLepton3_motherPdgId",
             "genLepton4_pt",
             "genLepton4_eta",
             "genLepton4_phi",
             "genLepton4_e",
             "genLepton4_pdgId",
             "genLepton4_motherPdgId",
             "matchedGenLepton1_pt",
             "matchedGenLepton1_eta",
             "matchedGenLepton1_phi",
             "matchedGenLepton1_e",
             "matchedGenLepton1_pdgId",
             "matchedGenLepton1_motherPdgId",
             "matchedGenLepton2_pt",
             "matchedGenLepton2_eta",
             "matchedGenLepton2_phi",
             "matchedGenLepton2_e",
             "matchedGenLepton2_pdgId",
             "matchedGenLepton2_motherPdgId",
             "matchedGenBQuark1_pt",
             "matchedGenBQuark1_eta",
             "matchedGenBQuark1_phi",
             "matchedGenBQuark1_e",
             "matchedGenBQuark1_motherPdgId",
             "matchedGenBQuark2_pt",
             "matchedGenBQuark2_eta",
             "matchedGenBQuark2_phi",
             "matchedGenBQuark2_e",
             "matchedGenBQuark2_motherPdgId",
             ]

ak_variables = ["jets_pt",
                "jets_eta",
                "jets_phi",
                "jets_e",
                "jets_btag_deepFlavor",
                "jets_pnet_bb",
                "jets_pnet_cc",
                "jets_pnet_b",
                "jets_pnet_c",
                "jets_pnet_g",
                "jets_pnet_uds",
                "jets_pnet_pu",
                "jets_pnet_undef",
                "jets_HHbtag",
                ]

aliasdict = {"lumi_weight": "MC_weight",
             "plot_weight": "(PUjetID_SF * L1pref_weight * prescaleWeight * trigSF * IdAndIsoAndFakeSF_deep_pt * bTagweightReshape)",
             "bjet1_btag_deepFlavor": "bjet1_bID_deepFlavor",
             "bjet2_btag_deepFlavor": "bjet2_bID_deepFlavor",
             "matchedGenLepton1_pt": "genLepton1_pt",
             "matchedGenLepton1_eta": "genLepton1_eta",
             "matchedGenLepton1_phi": "genLepton1_phi",
             "matchedGenLepton1_e": "genLepton1_e",
             "matchedGenLepton1_pdgId": "genLepton1_pdgId",
             "matchedGenLepton1_motherPdgId": "genLepton1_motherPdgId",
             "matchedGenLepton2_pt": "genLepton2_pt",
             "matchedGenLepton2_eta": "genLepton2_eta",
             "matchedGenLepton2_phi": "genLepton2_phi",
             "matchedGenLepton2_e": "genLepton2_e",
             "matchedGenLepton2_pdgId": "genLepton2_pdgId",
             "matchedGenLepton2_motherPdgId": "genLepton2_motherPdgId",
             "matchedGenBQuark1_pt": "genBQuark1_pt",
             "matchedGenBQuark1_eta": "genBQuark1_eta",
             "matchedGenBQuark1_phi": "genBQuark1_phi",
             "matchedGenBQuark1_e": "genBQuark1_e",
             "matchedGenBQuark1_motherPdgId": "genBQuark1_motherPdgId",
             "matchedGenBQuark2_pt": "genBQuark2_pt",
             "matchedGenBQuark2_eta": "genBQuark2_eta",
             "matchedGenBQuark2_phi": "genBQuark2_phi",
             "matchedGenBQuark2_e": "genBQuark2_e",
             "matchedGenBQuark2_motherPdgId": "genBQuark2_motherPdgId",
             }

ak_aliasdict = {
}

dtype = [(v, np.float32) for v in variables]
nentries_total = {}
nentries_selected = {}


def root2npz(filename):
    variables2 = []
    dtype2 = []
    outputfilename = filename.replace(".root", ".npz")
    outputfilename = outputfilename.replace("riegerma", "kramerto").replace("02Mar", "22May")
    if os.path.exists(outputfilename):
        print("skipping existing file")
        return
    # global nentries_total
    # global nentries_selected
    sample = filename.split("/")[-2]
    print(sample)
    f = uproot4.open(filename)
    try:
        tree = f["HTauTauTree"]
    except uproot4.exceptions.KeyInFileError:
        print("Key 'HTauTauTree' not found, skipped file " + filename)
        return

    weightsum = f["h_eff"].values()[0]

    # sel_baseline_emutau = "(((pairType == 0) | (pairType == 1)) & (dau1_pt > 20) & (abs(dau1_eta) < 2.1) & (dau2_pt > 20) & (abs(dau1_eta) < 2.3) & (nleps == 0) & (nbjetscand > 1))"
    # sel_baseline_tautau = "((pairType == 2) & (dau1_pt > 40) & (abs(dau1_eta) < 2.1) & (dau2_pt > 40) & (abs(dau1_eta) < 2.1) & (nleps == 0) & (nbjetscand > 1))"
    # sel_baseline = sel_baseline_emutau + " | " + sel_baseline_tautau

    sel_baseline = "(nbjetscand >= 0)"

    arrays = tree.arrays(variables, sel_baseline, library="np", aliases=aliasdict)
    ak_arrays = tree.arrays(ak_variables, sel_baseline, library="ak", aliases=ak_aliasdict)

    arrays["weightsum"] = [weightsum]
    variables2.append("weightsum")
    dtype2.append(("weightsum", np.float32))

    # btag_sorted = {}
    # for discr in ["btag_deepFlavor", "pnet_b", "pnet_bb"]:
    #     bjet1 = arrays[f"bjet1_{discr}"]
    #     bjet2 = arrays[f"bjet2_{discr}"]
    #     jets = ak_arrays[f"jets_{discr}"]
    #     jets = ak.concatenate((ak.unflatten(bjet1, [1]*len(bjet1)), ak.unflatten(bjet2, [1]*len(bjet2)), jets), axis=1)
    #     btag_sorted[discr] = ak.argsort(jets, axis=1, ascending=False)
    #
    # for var in ["pt", "eta", "phi", "e", "btag_deepFlavor", "pnet_b", "pnet_bb", "HHbtag"]:
    #     concat = ak.concatenate((ak.unflatten(arrays[f"bjet1_{var}"], [1]*len(arrays[f"bjet1_{var}"])),
    #                             ak.unflatten(arrays[f"bjet2_{var}"], [1]*len(arrays[f"bjet2_{var}"])), ak_arrays[f"jets_{var}"]), axis=1)
    #     for discr in ["btag_deepFlavor", "pnet_b", "pnet_bb"]:
    #         for i in range(2):
    #             variable_name = f"{discr}_sorted{i+1}_{var}"
    #             arrays[variable_name] = concat[btag_sorted[discr]][:, i]
    #             variables2.append(variable_name)
    #             dtype2.append((variable_name, np.float32))

    # dau1_px = arrays["dau1_pt"] * np.cos(arrays["dau1_phi"])
    # dau1_py = arrays["dau1_pt"] * np.sin(arrays["dau1_phi"])
    # dau1_pz = arrays["dau1_pt"] * np.sinh(arrays["dau1_eta"])
    # dau2_px = arrays["dau2_pt"] * np.cos(arrays["dau2_phi"])
    # dau2_py = arrays["dau2_pt"] * np.sin(arrays["dau2_phi"])
    # dau2_pz = arrays["dau2_pt"] * np.sinh(arrays["dau2_eta"])
    #
    # arrays["m_rec_vec"] = np.sqrt(np.square(arrays["dau1_e"] + arrays["dau2_e"] + np.sqrt(np.square(dau1_px - dau2_px) + np.square(dau1_py - dau2_py))) - np.square(dau1_px + dau2_px + (dau1_px - dau2_px)) - np.square(dau1_py + dau2_py + (dau1_py - dau2_py)) - np.square(dau1_pz + dau2_pz))
    #
    # arrays["m_rec_vec_abs"] = np.sqrt(np.square(arrays["dau1_e"] + arrays["dau2_e"] + np.sqrt(np.square(dau1_px - dau2_px) + np.square(dau1_py - dau2_py))) - np.square(dau1_px + dau2_px + np.abs(dau1_px - dau2_px)) - np.square(dau1_py + dau2_py + np.abs(dau1_py - dau2_py)) - np.square(dau1_pz + dau2_pz))

    # variables2.append("m_rec_vec")
    # dtype2.append(("m_rec_vec", np.float32))
    # variables2.append("m_rec_vec_abs")
    # dtype2.append(("m_rec_vec_abs", np.float32))

    lepton_delta_r_mask = delta_r_from_names(arrays, "genLepton1", "dau1") > delta_r_from_names(arrays, "genLepton2", "dau1")

    arrays["matchedGenLepton1_pt"][lepton_delta_r_mask] = arrays["genLepton2_pt"][lepton_delta_r_mask]
    arrays["matchedGenLepton1_eta"][lepton_delta_r_mask] = arrays["genLepton2_eta"][lepton_delta_r_mask]
    arrays["matchedGenLepton1_phi"][lepton_delta_r_mask] = arrays["genLepton2_phi"][lepton_delta_r_mask]
    arrays["matchedGenLepton1_e"][lepton_delta_r_mask] = arrays["genLepton2_e"][lepton_delta_r_mask]
    arrays["matchedGenLepton1_pdgId"][lepton_delta_r_mask] = arrays["genLepton2_pdgId"][lepton_delta_r_mask]
    arrays["matchedGenLepton1_motherPdgId"][lepton_delta_r_mask] = arrays["genLepton2_motherPdgId"][lepton_delta_r_mask]

    arrays["matchedGenLepton2_pt"][lepton_delta_r_mask] = arrays["genLepton1_pt"][lepton_delta_r_mask]
    arrays["matchedGenLepton2_eta"][lepton_delta_r_mask] = arrays["genLepton1_eta"][lepton_delta_r_mask]
    arrays["matchedGenLepton2_phi"][lepton_delta_r_mask] = arrays["genLepton1_phi"][lepton_delta_r_mask]
    arrays["matchedGenLepton2_e"][lepton_delta_r_mask] = arrays["genLepton1_e"][lepton_delta_r_mask]
    arrays["matchedGenLepton2_pdgId"][lepton_delta_r_mask] = arrays["genLepton1_pdgId"][lepton_delta_r_mask]
    arrays["matchedGenLepton2_motherPdgId"][lepton_delta_r_mask] = arrays["genLepton1_motherPdgId"][lepton_delta_r_mask]

    arrays["genLeptons_matched"] = (delta_r_from_names(arrays, "matchedGenLepton1", "dau1") < 0.2) & (delta_r_from_names(arrays, "matchedGenLepton2", "dau2") < 0.2)

    variables2.append("genLeptons_matched")
    dtype2.append(("genLeptons_matched", np.float32))

    b_delta_r_mask = delta_r_from_names(arrays, "genBQuark1", "bjet1") > delta_r_from_names(arrays, "genBQuark2", "bjet1")

    arrays["matchedGenBQuark1_pt"][b_delta_r_mask] = arrays["genBQuark2_pt"][b_delta_r_mask]
    arrays["matchedGenBQuark1_eta"][b_delta_r_mask] = arrays["genBQuark2_eta"][b_delta_r_mask]
    arrays["matchedGenBQuark1_phi"][b_delta_r_mask] = arrays["genBQuark2_phi"][b_delta_r_mask]
    arrays["matchedGenBQuark1_e"][b_delta_r_mask] = arrays["genBQuark2_e"][b_delta_r_mask]
    arrays["matchedGenBQuark1_motherPdgId"][b_delta_r_mask] = arrays["genBQuark2_motherPdgId"][b_delta_r_mask]

    arrays["matchedGenBQuark2_pt"][b_delta_r_mask] = arrays["genBQuark1_pt"][b_delta_r_mask]
    arrays["matchedGenBQuark2_eta"][b_delta_r_mask] = arrays["genBQuark1_eta"][b_delta_r_mask]
    arrays["matchedGenBQuark2_phi"][b_delta_r_mask] = arrays["genBQuark1_phi"][b_delta_r_mask]
    arrays["matchedGenBQuark2_e"][b_delta_r_mask] = arrays["genBQuark1_e"][b_delta_r_mask]
    arrays["matchedGenBQuark2_motherPdgId"][b_delta_r_mask] = arrays["genBQuark1_motherPdgId"][b_delta_r_mask]

    arrays["genBQuarks_matched"] = (delta_r_from_names(arrays, "matchedGenBQuark1", "bjet1") < 0.4) & (delta_r_from_names(arrays, "matchedGenBQuark2", "bjet2") < 0.4)

    variables2.append("genBQuarks_matched")
    dtype2.append(("genBQuarks_matched", np.float32))

    records = list(zip(*(arrays[v] for v in variables + variables2)))
    r = np.array(records, dtype=dtype + dtype2)

    os.makedirs(os.path.dirname(outputfilename), exist_ok=True)
    np.savez(outputfilename, events=r)


inputpath = "/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v4_02Mar23/"
dirnames = glob.glob(inputpath + "*")
for dirname in dirnames:
    if ".txt" in dirname or ".sh" in dirname:
        continue
    Path(dirname.replace("riegerma", "kramerto").replace("02Mar", "22May")).mkdir(parents=True, exist_ok=True)

filenames = glob.glob(inputpath + "*/*.root")

print(len(filenames))

pool = Pool(12)

pool.map(root2npz, filenames)

# root2npz(inputpath + "SKIM_GGHH_SM/output_0.root")
