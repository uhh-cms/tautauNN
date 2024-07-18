# coding: utf-8

"""
Notes:

1. Statistical model taken from [1].
2. Process contributions in KLUB are normalized to 1/pb so they have to be scaled by the luminostiy.
3. Signals are to be normalized to 1pb times the analysis branching ratio according to [2]. The
   latter is not taken into account by KLUB yet and therefore applied below.

[1] https://gitlab.cern.ch/hh/naming-conventions#systematic-uncertainties
[2] https://gitlab.cern.ch/hh/naming-conventions#2-resonant-interpretation
"""

from __future__ import annotations

import os
import gc
import re
import json
import itertools
import hashlib
import pickle
import tempfile
import shutil
import time
from functools import reduce, wraps
from operator import mul
from collections import OrderedDict, defaultdict
from fnmatch import fnmatch
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Sequence, Any, Callable

from tqdm import tqdm
import numpy as np
import awkward as ak
import uproot
import hist

from tautaunn.util import transform_data_dir_cache
from tautaunn.config import masses, spins, klub_index_columns, luminosities, btag_wps, pnet_wps


#
# configurations
#

br_hh_bbtt = 0.073056256
channels = {
    "mutau": 0,
    "etau": 1,
    "tautau": 2,
}
klub_weight_columns = [
    "MC_weight",
    "PUReweight",
    "L1pref_weight",
    "trigSF",
    "dauSFs",
    "PUjetID_SF",
    "bTagweightReshape",
]
klub_extra_columns = [
    # "DNNoutSM_kl_1",
]
# "years" in all structures above actually mean "era", so define "datacard year" as the actual year of an era
# for datacard purposes, as, for instance, eras "2016APV" and "2016" are both considered as datacard year "2016"
datacard_years = {
    "2016APV": "2016",
    "2016": "2016",
    "2017": "2017",
    "2018": "2018",
}
shape_nuisances = {}


@dataclass
class ShapeNuisance:
    name: str
    combine_name: str = ""
    processes: list[str] = field(default_factory=lambda: ["*"])
    weights: dict[str, tuple[str, str]] = field(default_factory=dict)  # original name mapped to (up, down) variations
    discriminator_suffix: tuple[str, str] = ("", "")  # name suffixes for (up, down) variations
    channels: set[str] = field(default_factory=set)
    skip: bool = False

    @classmethod
    def new(cls, *args, **kwargs):
        inst = cls(*args, **kwargs)
        shape_nuisances[inst.name] = inst
        return inst

    @classmethod
    def create_full_name(cls, name: str, *, year: str) -> str:
        return name.format(year=year)

    def __post_init__(self):
        # never skip nominal
        if self.is_nominal:
            self.skip = False

        # default combine name
        if not self.combine_name:
            self.combine_name = self.name

    @property
    def is_nominal(self) -> bool:
        return self.name == "nominal"

    def get_combine_name(self, *, year: str) -> str:
        return self.create_full_name(self.combine_name, year=year)

    def get_directions(self) -> list[str]:
        return [""] if self.is_nominal else ["up", "down"]

    def get_varied_weight(self, nominal_weight: str, direction: str) -> str:
        assert direction in ("", "up", "down")
        if direction:
            for nom, (up, down) in self.weights.items():
                if nom == nominal_weight:
                    return up if direction == "up" else down
        return nominal_weight

    def get_varied_full_weight(self, direction: str) -> str:
        assert direction in ("", "up", "down")
        # use the default weight field in case the nuisance is nominal or has no dedicated weight variations
        if not direction or not self.weights:
            return "full_weight_nominal"
        # compose the full weight field name
        return f"full_weight_{self.name}_{direction}"

    def get_varied_discriminator(self, nominal_discriminator: str, direction: str) -> str:
        assert direction in ("", "up", "down")
        suffix = ""
        if direction and (suffix := self.discriminator_suffix[direction == "down"]):
            suffix = f"_{suffix}"
        return nominal_discriminator + suffix

    def applies_to_process(self, process_name: str) -> bool:
        return any(fnmatch(process_name, pattern) for pattern in self.processes)

    def applies_to_channel(self, channel_name: str) -> bool:
        return not self.channels or channel_name in self.channels


ShapeNuisance.new(
    name="nominal",
)
ShapeNuisance.new(
    name="btag_hf",
    combine_name="CMS_btag_HF_2016_2017_2018",
    weights={"bTagweightReshape": ("bTagweightReshape_hf_up", "bTagweightReshape_hf_down")},
)
ShapeNuisance.new(
    name="btag_lf",
    combine_name="CMS_btag_LF_2016_2017_2018",
    weights={"bTagweightReshape": ("bTagweightReshape_lf_up", "bTagweightReshape_lf_down")},
)
ShapeNuisance.new(
    name="btag_lfstats1",
    combine_name="CMS_btag_lfstats1_{year}",
    weights={"bTagweightReshape": ("bTagweightReshape_lfstats1_up", "bTagweightReshape_lfstats1_down")},
)
ShapeNuisance.new(
    name="btag_lfstats2",
    combine_name="CMS_btag_lfstats2_{year}",
    weights={"bTagweightReshape": ("bTagweightReshape_lfstats2_up", "bTagweightReshape_lfstats2_down")},
)
ShapeNuisance.new(
    name="btag_hfstats1",
    combine_name="CMS_btag_hfstats1_{year}",
    weights={"bTagweightReshape": ("bTagweightReshape_hfstats1_up", "bTagweightReshape_hfstats1_down")},
)
ShapeNuisance.new(
    name="btag_hfstats2",
    combine_name="CMS_btag_hfstats2_{year}",
    weights={"bTagweightReshape": ("bTagweightReshape_hfstats2_up", "bTagweightReshape_hfstats2_down")},
)
ShapeNuisance.new(
    name="btag_cferr1",
    combine_name="CMS_btag_cfeff1_{year}",
    weights={"bTagweightReshape": ("bTagweightReshape_cferr1_up", "bTagweightReshape_cferr1_down")},
)
ShapeNuisance.new(
    name="btag_cferr2",
    combine_name="CMS_btag_cfeff2_{year}",
    weights={"bTagweightReshape": ("bTagweightReshape_cferr2_up", "bTagweightReshape_cferr2_down")},
)

for (name, add_year) in [
    ("stat0_DM0", True),
    ("stat1_DM0", True),
    ("systuncorrdmeras_DM0", True),
    ("stat0_DM1", True),
    ("stat1_DM1", True),
    ("systuncorrdmeras_DM1", True),
    ("stat0_DM10", True),
    ("stat1_DM10", True),
    ("systuncorrdmeras_DM10", True),
    ("stat0_DM11", True),
    ("stat1_DM11", True),
    ("systuncorrdmeras_DM11", True),
    ("systcorrdmeras", False),
    ("systcorrdmuncorreras", True),
    ("systcorrerasgt140", False),
    ("stat0gt140", True),
    ("stat1gt140", True),
    ("extrapgt140", False),
]:
    ShapeNuisance.new(
        name=f"id_tauid_2d_{name}",
        combine_name=f"CMS_eff_t_{name}" + (r"_{year}" if add_year else ""),
        weights={"dauSFs": (f"dauSFs_tauid_2d_{name}_up", f"dauSFs_tauid_2d_{name}_down")},
    )
# TODO: are we certain all of the following uncertainties should be uncorrelated across years?
ShapeNuisance.new(
    name="id_etauFR_barrel",
    combine_name="CMS_bbtt_etauFR_barrel_{year}",
    weights={"dauSFs": ("dauSFs_etauFR_barrel_up", "dauSFs_etauFR_barrel_down")},
)
ShapeNuisance.new(
    name="id_etauFR_endcap",
    combine_name="CMS_bbtt_etauFR_endcap_{year}",
    weights={"dauSFs": ("dauSFs_etauFR_endcap_up", "dauSFs_etauFR_endcap_down")},
)
ShapeNuisance.new(
    name="id_mutauFR_etaLt0p4",
    combine_name="CMS_bbtt_mutauFR_etaLt0p4_{year}",
    weights={"dauSFs": ("dauSFs_mutauFR_etaLt0p4_up", "dauSFs_mutauFR_etaLt0p4_down")},
)
ShapeNuisance.new(
    name="id_mutauFR_eta0p4to0p8",
    combine_name="CMS_bbtt_mutauFR_eta0p4to0p8_{year}",
    weights={"dauSFs": ("dauSFs_mutauFR_eta0p4to0p8_up", "dauSFs_mutauFR_eta0p4to0p8_down")},
)
ShapeNuisance.new(
    name="id_mutauFR_eta0p8to1p2",
    combine_name="CMS_bbtt_mutauFR_eta0p8to1p2_{year}",
    weights={"dauSFs": ("dauSFs_mutauFR_eta0p8to1p2_up", "dauSFs_mutauFR_eta0p8to1p2_down")},
)
ShapeNuisance.new(
    name="id_mutauFR_etaGt1p2to1p7",
    combine_name="CMS_bbtt_mutauFR_eta1p2to1p7_{year}",
    weights={"dauSFs": ("dauSFs_mutauFR_eta1p2to1p7_up", "dauSFs_mutauFR_eta1p2to1p7_down")},
)
ShapeNuisance.new(
    name="id_mutauFR_etaGt1p7",
    combine_name="CMS_bbtt_mutauFR_etaGt1p7_{year}",
    weights={"dauSFs": ("dauSFs_mutauFR_etaGt1p7_up", "dauSFs_mutauFR_etaGt1p7_down")},
)
ShapeNuisance.new(
    name="id_muid",
    combine_name="CMS_eff_m_id_{year}",
    weights={"dauSFs": ("dauSFs_muID_up", "dauSFs_muID_down")},
)
ShapeNuisance.new(
    name="id_muiso",
    combine_name="CMS_eff_m_iso_{year}",
    weights={"dauSFs": ("dauSFs_muIso_up", "dauSFs_muIso_down")},
)
ShapeNuisance.new(
    name="id_elereco",
    combine_name="CMS_eff_e_reco_{year}",
    weights={"dauSFs": ("dauSFs_eleReco_up", "dauSFs_eleReco_down")},
)
ShapeNuisance.new(
    name="id_eleid",
    combine_name="CMS_eff_e_id_{year}",
    weights={"dauSFs": ("dauSFs_eleID_up", "dauSFs_eleID_down")},
)
ShapeNuisance.new(
    name="pu_jet_id",
    combine_name="CMS_eff_j_PUJET_id_{year}",
    weights={"PUjetID_SF": ("PUjetID_SF_up", "PUjetID_SF_down")},
)
ShapeNuisance.new(
    name="trigSF_tau_DM0",
    combine_name="CMS_bbtt_{year}_trigSFTauDM0",
    weights={"trigSF": ("trigSF_tau_DM0_up", "trigSF_tau_DM0_down")},
)
ShapeNuisance.new(
    name="trigSF_tau_DM1",
    combine_name="CMS_bbtt_{year}_trigSFTauDM1",
    weights={"trigSF": ("trigSF_tau_DM1_up", "trigSF_tau_DM1_down")},
)
ShapeNuisance.new(
    name="trigSF_tau_DM10",
    combine_name="CMS_bbtt_{year}_trigSFTauDM10",
    weights={"trigSF": ("trigSF_tau_DM10_up", "trigSF_tau_DM10_down")},
)
ShapeNuisance.new(
    name="trigSF_DM11",
    combine_name="CMS_bbtt_{year}_trigSFTauDM11",
    weights={"trigSF": ("trigSF_tau_DM11_up", "trigSF_tau_DM11_down")},
)
ShapeNuisance.new(
    name="trigSF_met",
    combine_name="CMS_bbtt_{year}_trigSFMET",
    weights={"trigSF": ("trigSF_met_up", "trigSF_met_down")},
)
ShapeNuisance.new(
    name="trigSF_stau",
    combine_name="CMS_bbtt_{year}_trigSFSingleTau",
    weights={"trigSF": ("trigSF_stau_up", "trigSF_stau_down")},
)
ShapeNuisance.new(
    name="trigSF_ele",
    combine_name="CMS_bbtt_{year}_trigSFEle",
    weights={"trigSF": ("trigSF_ele_up", "trigSF_ele_down")},
    channels={"etau"},
)
ShapeNuisance.new(
    name="trigSF_mu",
    combine_name="CMS_bbtt_{year}_trigSFEMu",
    weights={"trigSF": ("trigSF_mu_up", "trigSF_mu_down")},
    channels={"mutau"},
)
ShapeNuisance.new(
    name="ees",
    combine_name="CMS_scale_e_{year}",  # TODO: check
    discriminator_suffix=("ees_up", "ees_down"),
)
ShapeNuisance.new(
    name="eer",
    combine_name="CMS_sigma_e_{year}",  # TODO: check
    discriminator_suffix=("eer_sigma_up", "eer_sigma_down"),
)
ShapeNuisance.new(
    name="fes_DM0",
    combine_name="CMS_scale_t_eFake_DM0_{year}",
    discriminator_suffix=("fes_DM0_up", "fes_DM0_down"),
)
ShapeNuisance.new(
    name="fes_DM1",
    combine_name="CMS_scale_t_eFake_DM1_{year}",
    discriminator_suffix=("fes_DM1_up", "fes_DM1_down"),
)
ShapeNuisance.new(
    name="tes_DM0",
    combine_name="CMS_scale_t_DM0_{year}",
    discriminator_suffix=("tes_DM0_up", "tes_DM0_down"),
)
ShapeNuisance.new(
    name="tes_DM1",
    combine_name="CMS_scale_t_DM1_{year}",
    discriminator_suffix=("tes_DM1_up", "tes_DM1_down"),
)
ShapeNuisance.new(
    name="tes_DM10",
    combine_name="CMS_scale_t_DM10_{year}",
    discriminator_suffix=("tes_DM10_up", "tes_DM10_down"),
)
ShapeNuisance.new(
    name="tes_DM11",
    combine_name="CMS_scale_t_DM11_{year}",
    discriminator_suffix=("tes_DM11_up", "tes_DM11_down"),
)
ShapeNuisance.new(
    name="mes",
    combine_name="CMS_scale_t_muFake_{year}",
    discriminator_suffix=("mes_up", "mes_down"),
)
ShapeNuisance.new(
    name="PUReweight",
    combine_name="CMS_pileup_{year}",
    weights={"PUReweight": ("PUReweight_up", "PUReweight_down")},
)
ShapeNuisance.new(
    name="l1_prefiring",
    combine_name="CMS_l1_prefiring_{year}",
    weights={"L1pref_weight": ("L1pref_weight_up", "L1pref_weight_down")},
)

jes_names = {
    1: "CMS_j_Abs",
    2: "CMS_j_Abs_{year}",
    3: "CMS_j_BBEC1",
    4: "CMS_j_BBEC1_{year}",
    5: "CMS_j_EC2",
    6: "CMS_j_EC2_{year}",
    7: "CMS_j_FlavQCD",
    8: "CMS_j_HF",
    9: "CMS_j_HF_{year}",
    10: "CMS_j_RelBal",
    11: "CMS_j_RelSample_{year}",
}

for js in range(1, 12):
    ShapeNuisance.new(
        name=f"jes_{js}",
        combine_name=jes_names[js],
        discriminator_suffix=(f"jes_{js}_up", f"jes_{js}_down"),
        weights={"bTagweightReshape": (f"bTagweightReshape_jetup{js}", f"bTagweightReshape_jetdown{js}")},
    )

# TODO: JER

processes = OrderedDict({
    "TT": {
        "id": 1,
        "sample_patterns": ["TT_*"],
    },
    "ST": {
        "id": 2,
        "sample_patterns": ["ST_*"],
    },
    "DY": {
        "id": 3,
        "sample_patterns": ["DY_*"],
    },
    "W": {
        "id": 4,
        "sample_patterns": ["WJets_*"],
    },
    "EWK": {
        "id": 5,
        "sample_patterns": ["EWK*"],
    },
    "WW": {
        "id": 6,
        "sample_patterns": ["WW"],
    },
    "WZ": {
        "id": 7,
        "sample_patterns": ["WZ"],
    },
    "ZZ": {
        "id": 8,
        "sample_patterns": ["ZZ"],
    },
    "VVV": {
        "id": 9,
        "sample_patterns": ["WWW", "WWZ", "WZZ", "ZZZ"],
    },
    "TTV": {
        "id": 10,
        "sample_patterns": ["TTWJets*", "TTZTo*"],
    },
    "TTVV": {
        "id": 11,
        "sample_patterns": ["TTWW", "TTWZ", "TTZZ"],
    },
    "ggH_htt": {
        "id": 12,
        "sample_patterns": ["GluGluHToTauTau"],
    },
    "qqH_htt": {
        "id": 13,
        "sample_patterns": ["VBFHToTauTau"],
    },
    "ZH_htt": {
        "id": 14,
        "sample_patterns": ["ZHToTauTau"],
    },
    "WH_htt": {
        "id": 15,
        "sample_patterns": ["WminusHToTauTau", "WplusHToTauTau"],
    },
    "ttH_hbb": {
        "id": 16,
        "sample_patterns": ["ttHTobb"],
    },
    "ttH_htt": {
        "id": 17,
        "sample_patterns": ["ttHToTauTau"],
    },
    # "ggHH_hbbhtt": {
    #     "id": 18,
    #     "sample_patterns": ["GGHH_SM"],
    # },
    "QCD": {
        "id": 19,
        "sample_patterns": [],
    },
    **{
        f"ggf_spin_{spin}_mass_{mass}_hbbhtt": {
            "id": 0,
            "sample_patterns": [f"{resonance}{mass}"],
            "spin": spin,
            "mass": mass,
            "signal": True,
        }
        for mass in masses
        for spin, resonance in zip(spins, ["Rad", "Grav"])
    },
    "data_mu": {
        "sample_patterns": ["Muon*"],
        "data": True,
        "channels": ["mutau"],
    },
    "data_egamma": {
        "sample_patterns": ["EGamma*"],
        "data": True,
        "channels": ["etau"],
    },
    "data_tau": {
        "sample_patterns": ["Tau*"],
        "data": True,
        "channels": ["tautau"],
    },
    "data_met": {
        "sample_patterns": ["MET*"],
        "data": True,
        "channels": ["mutau", "etau", "tautau"],
    },
})
stat_model = {
    "BR_hbb": {
        "*_hbb": "0.9874/1.0124",
        "*_hbbhtt": "0.9874/1.0124",
    },
    "BR_htt": {
        "*_hbbhtt": "0.9837/1.0165",
        "*_htt": "0.9837/1.0165",
    },
    "pdf_gg": {
        "TT": "1.042",
    },
    "pdf_qqbar": {
        "ST": "1.028",  # conservatively from t-channel, also added to tW-channel
        "WZ": "1.044",
    },
    "pdf_Higgs_gg": {
        "ggH_*": "1.019",
    },
    "pdf_Higgs_qqbar": {
        "qqH_*": "1.021",
        "WH_*": "1.017",
        "ZH_*": "1.013",
    },
    "pdf_Higgs_ttH": {
        "ttH_*": "1.030",
    },
    "pdf_Higgs_ggHH": {
        "ggHH_*": "1.030",
    },
    "pdf_Higgs_qqHH": {
        "qqHH_*": "1.021",
    },
    "QCDscale_ttbar": {
        "TT": "0.965/1.024",
        "ST": "0.979/1.031",  # conservatively from t-channel
    },
    "QCDscale_VV": {
        "WZ": "1.036",
    },
    "QCDscale_ggH": {
        "ggH_*": "1.039",
    },
    "QCDscale_qqH": {
        "qqH_*": "0.997/1.004",
    },
    "QCDscale_VH": {
        "WH_*": "0.993/1.005",
        "ZH_*": "0.970/1.038",
    },
    "QCDscale_ttH": {
        "ttH_*": "0.908/1.058",
    },
    "QCDscale_ggHH": {
        "ggHH_*": "0.770/1.060",  # includes fully correlated mtop uncertainty
    },
    "QCDscale_qqHH": {
        "qqHH_*": "0.9996/1.0003",
    },
    "alpha_s": {
        "ggH_*": "1.026",
        "qqH_*": "1.005",
        "ZH_*": "1.009",
        "WH_*": "1.009",
        "ttH_*": "1.020",
    },
    "qqHH_pythiaDipoleOn": {
        "qqHH_*": "0.781/1.219",
    },
    # year dependent (both the selection of nuisances and their effect depend on the year)
    "lumi_13TeV_2016": {"!QCD": {"2016*": "1.010"}},
    "lumi_13TeV_2017": {"!QCD": {"2017": "1.020"}},
    "lumi_13TeV_2018": {"!QCD": {"2018": "1.015"}},
    "lumi_13TeV_1718": {"!QCD": {"2017": "1.006", "2018": "1.002"}},
    "lumi_13TeV_correlated": {"!QCD": {"2016*": "1.006", "2017": "1.009", "2018": "1.020"}},
}


def merge_dicts(*dicts):
    assert dicts
    merged = dicts[0].__class__()
    for d in dicts:
        merged.update(deepcopy(d))
    return merged


def sample_name_to_skim_dir(sample_name: str) -> str:
    # this used to be f"SKIM_{sample_name}"
    return sample_name


def dir_is_skim_dir(dir_name: str) -> bool:
    # without the gone SKIM_ prefix we can no longer check this
    return True


def make_list(x):
    return list(x) if isinstance(x, (list, tuple, set)) else [x]


def selector(
    needs: list | None = None,
    str_repr: str | None = None,
    **extra,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        # declare func to be a selector
        func.is_selector = True

        # store extra data
        func.extra = extra

        # store raw list of required columns
        func.raw_columns = list(needs or [])

        # store recursive flat list of actual column names
        func.flat_columns = []
        for obj in func.raw_columns:
            if isinstance(obj, str):
                func.flat_columns.append(obj)
            elif getattr(obj, "is_selector", False):
                func.flat_columns.extend(obj.flat_columns)
            else:
                raise TypeError(f"cannot interpret columns '{obj}'")
        func.flat_columns = sorted(set(func.flat_columns), key=func.flat_columns.index)

        # store the string representation
        func.str_repr = str_repr

        @wraps(func)
        def wrapper(*args, **kwargs) -> ak.Array:
            return ak.values_astype(func(*args, **kwargs), bool)

        return wrapper
    return decorator


@selector(
    needs=["pairType", "dau1_deepTauVsJet", "dau1_iso", "dau1_eleMVAiso"],
    str_repr="((pairType == 0) & (dau1_iso < 0.15)) | ((pairType == 1) & (dau1_eleMVAiso == 1)) | ((pairType == 2) & (dau1_deepTauVsJet >= 5))",  # noqa
)
def sel_iso_first_lep(array: ak.Array, **kwargs) -> ak.Array:
    return (
        ((array.pairType == 0) & (array.dau1_iso < 0.15)) |
        ((array.pairType == 1) & (array.dau1_eleMVAiso == 1)) |
        ((array.pairType == 2) & (array.dau1_deepTauVsJet >= 5))
    )


@selector(
    needs=["isLeptrigger", "isMETtrigger", "isSingleTautrigger"],
    str_repr="((isLeptrigger == 1) | (isMETtrigger == 1) | (isSingleTautrigger == 1))",
)
def sel_trigger(array: ak.Array, **kwargs) -> ak.Array:
    return (
        (array.isLeptrigger == 1) | (array.isMETtrigger == 1) | (array.isSingleTautrigger == 1)
    )


@selector(
    needs=[sel_trigger, sel_iso_first_lep, "nleps", "nbjetscand", "isBoosted"],
    str_repr=f"({sel_trigger.str_repr}) & ({sel_iso_first_lep.str_repr}) & (nleps == 0) & ((nbjetscand > 1) | (isBoosted == 1))",  # noqa
)
def sel_baseline(array: ak.Array, **kwargs) -> ak.Array:
    return (
        sel_trigger(array, **kwargs) &
        # including cut on first isolated lepton to reduce memory footprint
        # (note that this is not called "baseline" anymore by KLUB standards)
        sel_iso_first_lep(array, **kwargs) &
        (array.nleps == 0) &
        ((array.nbjetscand > 1) | (array.isBoosted == 1))
    )


@selector(
    needs=["isOS", "dau2_deepTauVsJet", sel_iso_first_lep],
)
def sel_region_os_iso(array: ak.Array, **kwargs) -> ak.Array:
    return (
        sel_iso_first_lep(array, **kwargs) &
        (array.isOS == 1) &
        (array.dau2_deepTauVsJet >= 5)
    )


@selector(
    needs=["isOS", "dau2_deepTauVsJet", sel_iso_first_lep],
)
def sel_region_ss_iso(array: ak.Array, **kwargs) -> ak.Array:
    return (
        sel_iso_first_lep(array, **kwargs) &
        (array.isOS == 0) &
        (array.dau2_deepTauVsJet >= 5)
    )


@selector(
    needs=["isOS", "dau2_deepTauVsJet", sel_iso_first_lep],
)
def sel_region_os_noniso(array: ak.Array, **kwargs) -> ak.Array:
    return (
        sel_iso_first_lep(array, **kwargs) &
        (array.isOS == 1) &
        (array.dau2_deepTauVsJet < 5) &
        (array.dau2_deepTauVsJet >= 1)
    )


@selector(
    needs=["isOS", "dau2_deepTauVsJet", sel_iso_first_lep],
)
def sel_region_ss_noniso(array: ak.Array, **kwargs) -> ak.Array:
    return (
        sel_iso_first_lep(array, **kwargs) &
        (array.isOS == 0) &
        (array.dau2_deepTauVsJet < 5) &
        (array.dau2_deepTauVsJet >= 1)
    )


region_sels = [
    sel_region_os_iso,
    sel_region_ss_iso,
    sel_region_os_noniso,
    sel_region_ss_noniso,
]


region_sel_names = ["os_iso", "ss_iso", "os_noniso", "ss_noniso"]


def category_factory(channel: str) -> dict[str, Callable]:
    pair_type = channels[channel]

    @selector(needs=["pairType"])
    def sel_channel(array: ak.Array, **kwargs) -> ak.Array:
        return array.pairType == pair_type

    @selector(needs=["isBoosted", "fatjet_particleNetMDJetTags_probXbb"])
    def sel_boosted(array: ak.Array, **kwargs) -> ak.Array:
        year = kwargs["year"]
        return (
            (array.isBoosted == 1) &
            (array.fatjet_particleNetMDJetTags_probXbb >= pnet_wps[year])
        )

    def sel_combinations(main_sel, sub_sels):
        def create(sub_sel):
            @selector(
                needs=[main_sel, sub_sel],
                channel=channel,
            )
            def func(array: ak.Array, **kwargs) -> ak.Array:
                return main_sel(array, **kwargs) & sub_sel(array, **kwargs)
            return func

        return [create(sub_sel) for sub_sel in sub_sels]

    @selector(needs=["bjet1_bID_deepFlavor", "bjet2_bID_deepFlavor"])
    def sel_btag_m(array: ak.Array, **kwargs) -> ak.Array:
        year = kwargs["year"]
        return (
            (array.bjet1_bID_deepFlavor > btag_wps[year]["medium"]) &
            (array.bjet2_bID_deepFlavor <= btag_wps[year]["medium"])
        ) | (
            (array.bjet1_bID_deepFlavor <= btag_wps[year]["medium"]) &
            (array.bjet2_bID_deepFlavor > btag_wps[year]["medium"])
        )

    @selector(needs=["bjet1_bID_deepFlavor", "bjet2_bID_deepFlavor"])
    def sel_btag_mm(array: ak.Array, **kwargs) -> ak.Array:
        year = kwargs["year"]
        return (
            (array.bjet1_bID_deepFlavor > btag_wps[year]["medium"]) &
            (array.bjet2_bID_deepFlavor > btag_wps[year]["medium"])
        )

    @selector(needs=["bjet1_bID_deepFlavor", "bjet2_bID_deepFlavor"])
    def sel_btag_ll(array: ak.Array, **kwargs) -> ak.Array:
        year = kwargs["year"]
        return (
            (array.bjet1_bID_deepFlavor > btag_wps[year]["loose"]) &
            (array.bjet2_bID_deepFlavor > btag_wps[year]["loose"])
        )

    @selector(needs=["bjet1_bID_deepFlavor", "bjet2_bID_deepFlavor"])
    def sel_btag_m_first(array: ak.Array, **kwargs) -> ak.Array:
        year = kwargs["year"]
        return (
            (array.bjet1_bID_deepFlavor > btag_wps[year]["medium"]) |
            (array.bjet2_bID_deepFlavor > btag_wps[year]["medium"])
        )

    @selector(needs=["tauH_mass", "bH_mass"])
    def sel_mass_window(array: ak.Array, **kwargs) -> ak.Array:
        return (
            (array.tauH_mass >= 20.0) &
            (array.tauH_mass <= 130.0) &
            (array.bH_mass >= 40.0) &
            (array.bH_mass <= 270.0)
        )

    @selector(
        needs=[sel_baseline],
        channel=channel,
    )
    def cat_baseline(array: ak.Array, **kwargs) -> ak.Array:
        return sel_baseline(array, **kwargs)

    @selector(
        needs=[sel_baseline, sel_channel, sel_boosted, sel_btag_m, sel_mass_window],
        channel=channel,
    )
    def cat_resolved_1b(array: ak.Array, **kwargs) -> ak.Array:
        return (
            sel_baseline(array, **kwargs) &
            sel_channel(array, **kwargs) &
            ~sel_boosted(array, **kwargs) &
            sel_btag_m(array, **kwargs) &
            sel_mass_window(array, **kwargs)
        )

    @selector(
        needs=[sel_baseline, sel_channel, sel_boosted, sel_btag_mm, sel_mass_window],
        channel=channel,
    )
    def cat_resolved_2b(array: ak.Array, **kwargs) -> ak.Array:
        return (
            sel_baseline(array, **kwargs) &
            sel_channel(array, **kwargs) &
            ~sel_boosted(array, **kwargs) &
            sel_btag_mm(array, **kwargs) &
            sel_mass_window(array, **kwargs)
        )

    @selector(
        needs=[sel_baseline, sel_channel, sel_boosted, sel_mass_window],
        channel=channel,
    )
    def cat_boosted(array: ak.Array, **kwargs) -> ak.Array:
        return (
            sel_baseline(array, **kwargs) &
            sel_channel(array, **kwargs) &
            sel_boosted(array, **kwargs) &
            sel_mass_window(array, **kwargs)
        )

    # create a dict of all selectors, but without subdivision into regions
    selectors = {
        "baseline": cat_baseline,
        "resolved1b": cat_resolved_1b,
        "resolved2b": cat_resolved_2b,
        "boosted": cat_boosted,
    }

    # add all region combinations
    for name, sel in list(selectors.items()):
        selectors.update({
            f"{name}_{region_name}": combined_sel
            for region_name, combined_sel in zip(
                region_sel_names,
                sel_combinations(sel, region_sels),
            )
        })

    return selectors


categories = {}
for channel in channels:
    for name, sel in category_factory(channel=channel).items():
        # categories per year
        for year in ["2016", "2016APV", "2017", "2018"]:
            categories[f"{year}_{channel}_{name}"] = {
                "selection": sel,
                "n_bins": 10,
                "year": year,
                **sel.extra,
            }

        # combined categories
        categories[f"run2_{channel}_{name}"] = {
            "selection": sel,
            "n_bins": 40,  # TODO: tune!
            "year": None,
            **sel.extra,
        }


#
# functions for loading inputs
#

def load_klub_file(
    skim_directory: str,
    sample_name: str,
    file_name: str,
    is_data: bool,
) -> tuple[ak.Array, float]:
    # all weight column patterns
    klub_weight_column_patterns = klub_weight_columns + [f"{c}*" for c in klub_weight_columns]

    # all columns that should be loaded and kept later on
    persistent_columns = klub_index_columns + klub_extra_columns + sel_baseline.flat_columns
    # add all columns potentially necessary for selections
    persistent_columns += sum([
        cat["selection"].flat_columns
        for cat in categories.values()
    ], [])

    # load the array
    f = uproot.open(os.path.join(skim_directory, sample_name_to_skim_dir(sample_name), file_name))
    array = f["HTauTauTree"].arrays(
        filter_name=list(set(persistent_columns + ([] if is_data else klub_weight_column_patterns))),
        cut=sel_baseline.str_repr.strip(),
    )

    # data / mc specifics
    if is_data:
        # fake weight for data
        array = ak.with_field(array, 1.0, "full_weight_nominal")
    else:
        # compute the full weight for each shape variation (includes nominal)
        # and complain when non-finite weights were found
        for nuisance in shape_nuisances.values():
            if not nuisance.is_nominal and not nuisance.weights:
                continue
            for direction in nuisance.get_directions():
                weight_name = f"full_weight_{nuisance.name + (direction and '_' + direction)}"
                array = ak.with_field(
                    array,
                    reduce(mul, (array[nuisance.get_varied_weight(c, direction)] for c in klub_weight_columns)),
                    weight_name,
                )
                mask = ~np.isfinite(array[weight_name])
                if np.any(mask):
                    print(
                        f"found {sum(mask)} ({100.0 * sum(mask) / len(mask):.2f}% of {len(mask)}) "
                        f"non-finite weight values in sample {sample_name}, file {file_name}, variation {direction}",
                    )
                    array = array[~mask]
                persistent_columns.append(weight_name)

    # drop weight columns
    for field in array.fields:
        if field not in persistent_columns:
            array = ak.without_field(array, field)

    # also get the sum of generated weights, for nominal and pu variations
    sum_gen_mc_weights = {
        key: len(array) if is_data else float(f["h_eff"].values()[hist_idx])
        for key, hist_idx in [
            ("nominal", 0),
            ("PUReweight_up", 4),
            ("PUReweight_down", 5),
        ]
    }

    return array, sum_gen_mc_weights


def load_dnn_file(
    eval_directory: str,
    sample_name: str,
    file_name: str,
    dnn_output_columns: list[str],
    is_data: bool,
) -> ak.Array:
    # prepare expressions
    expressions = klub_index_columns + dnn_output_columns
    # extended output columns for variations if not data
    if not is_data:
        expressions += [f"{c}*" for c in dnn_output_columns]
    expressions = list(set(expressions))

    # load the array
    f = uproot.open(os.path.join(eval_directory, sample_name_to_skim_dir(sample_name), file_name))
    try:
        array = f["evaluation"].arrays(filter_name=expressions)
    except uproot.exceptions.KeyInFileError:
        array = f["hbtres"].arrays(filter_name=expressions)

    return array


def load_file(
    skim_directory: str,
    eval_directory: str,
    sample_name: str,
    klub_file_name: str,
    eval_file_name: str,
    dnn_output_columns: list[str],
    is_data: bool,
) -> tuple[ak.Array, float]:
    # load the klub file
    klub_array, sum_gen_mc_weights = load_klub_file(skim_directory, sample_name, klub_file_name, is_data)

    # load the dnn output file
    if eval_directory:
        dnn_array = load_dnn_file(eval_directory, sample_name, eval_file_name, dnn_output_columns, is_data)

        # use klub array index to filter dnn array
        klub_mask = np.isin(klub_array[klub_index_columns], dnn_array[klub_index_columns])
        if ak.sum(klub_mask) != len(dnn_array):
            klub_path = os.path.join(skim_directory, sample_name_to_skim_dir(sample_name), klub_file_name)
            eval_path = os.path.join(eval_directory, sample_name_to_skim_dir(sample_name), eval_file_name)
            raise Exception(
                f"the number of matching klub array columns ({ak.sum(klub_mask)}) does not match the "
                f"number of elements in the dnn eval array ({len(dnn_array)}) for file {klub_file_name} "
                f"(klub: {klub_path}, dnn: {eval_path})",
            )
        klub_array = klub_array[klub_mask]

        # exact (event, run, lumi) index check to make sure the order is identical as well
        matches = (
            (dnn_array.EventNumber == klub_array.EventNumber) &
            (dnn_array.RunNumber == klub_array.RunNumber) &
            (dnn_array.lumi == klub_array.lumi)
        )
        if not ak.all(matches):
            raise Exception(
                f"found event mismatch between klub and dnn files in {int(ak.sum(~matches))} cases "
                f"in files {klub_file_name} / {eval_file_name}",
            )

    # drop index columns
    array = dnn_array
    for field in klub_index_columns:
        array = ak.without_field(array, field)

    # add klub columns
    for field in klub_array.fields:
        if field in klub_index_columns:
            continue
        array = ak.with_field(array, klub_array[field], field)

    return array, sum_gen_mc_weights


def load_file_mp(args: tuple[Any]) -> tuple[ak.Array, float]:
    return load_file(*args)


def get_cache_path(
    cache_directory: str,
    skim_directory: str,
    eval_directory: str,
    year: str,
    sample_name: str,
    dnn_output_columns: list[str],
) -> str | None:
    if not cache_directory:
        return None

    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory)

    # get a list of all columns potentially needed by all selectors
    klub_columns = sorted(set(sum([
        cat["selection"].flat_columns
        for cat in categories.values()
    ], [])))

    # create a hash
    h = [
        transform_data_dir_cache(skim_directory),
        transform_data_dir_cache(eval_directory),
        sel_baseline.str_repr.strip(),
        klub_columns,
        klub_extra_columns,
        sorted(dnn_output_columns),
    ]
    h = hashlib.sha256(str(h).encode("utf-8")).hexdigest()[:10]

    return os.path.join(cache_directory, f"{year}_{sample_name}_{h}.pkl")


def load_sample_data(
    skim_directory: str,
    eval_directory: str,
    year: str,
    sample_name: str,
    selection_columns: list[str] | None = None,
    dnn_output_columns: list[str] | None = None,
    n_parallel: int = 4,
    cache_directory: str = "",
) -> ak.Array:
    print(f"loading sample {sample_name} ({year}) ...")

    # load from cache?
    cache_path = get_cache_path(
        cache_directory,
        skim_directory,
        eval_directory,
        year,
        sample_name,
        dnn_output_columns or [],
    )
    if cache_path and os.path.exists(cache_path):
        print("reading from cache")
        with open(cache_path, "rb") as f:
            array = pickle.load(f)

    else:
        # check if this is data
        is_data = False
        for process_data in processes.values():
            if any(fnmatch(sample_name, pattern) for pattern in process_data["sample_patterns"]):
                is_data = process_data.get("data", False)
                break
        else:
            raise Exception(f"could not determine if sample {sample_name} is data")

        # determine file names and build arguments for the parallel load implementation
        load_args = [
            (
                skim_directory,
                eval_directory,
                sample_name,
                eval_file_name.replace("_nominal", "").replace("_systs", ""),
                eval_file_name,
                dnn_output_columns or [],
                is_data,
            )
            for eval_file_name in os.listdir(os.path.join(eval_directory, sample_name_to_skim_dir(sample_name)))
            if fnmatch(eval_file_name, "output_*_systs.root")
        ]

        # run in parallel
        if n_parallel > 1:
            # run in parallel
            with ProcessPool(n_parallel, maxtasksperchild=None) as pool:
                ret = list(tqdm(pool.imap(load_file_mp, load_args), total=len(load_args)))
        else:
            ret = list(tqdm(map(load_file_mp, load_args), total=len(load_args)))

        # combine values
        array = ak.concatenate([arr for arr, _ in ret], axis=0)
        sum_gen_mc_weights = defaultdict(float)
        for _, weight_dict in ret:
            for key, sum_weights in weight_dict.items():
                sum_gen_mc_weights[key] += sum_weights
        del ret
        gc.collect()

        # update the full weight
        for field in array.fields:
            if field.startswith("full_weight_"):
                for key, sum_weights in sum_gen_mc_weights.items():
                    if field.endswith(key):
                        break
                else:
                    sum_weights = sum_gen_mc_weights["nominal"]
                array = ak.with_field(array, array[field] / sum_weights, field)

        # add to cache?
        if cache_path:
            print("writing to cache")
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(array, f)
            except:
                try:
                    os.remove(cache_path)
                except:
                    pass
                raise

    print("done")

    return array


def expand_categories(category: str | Sequence[str]) -> list[str]:
    _categories = []
    for pattern in make_list(category):
        pattern_matched = False
        for category in categories:
            if fnmatch(category, pattern):
                pattern_matched = True
                if category not in _categories:
                    _categories.append(category)
        # still add the pattern to handle errors in input checks below
        if not pattern_matched:
            _categories.append(pattern)
    return _categories


#
# functions for writing datacards
#

def write_datacards(
    spin: int | Sequence[int],
    mass: int | Sequence[int],
    category: str | Sequence[str],
    skim_directories: dict[tuple[str, str], list[str] | None],
    eval_directories: dict[str, str],
    output_directory: str,
    output_pattern: str = "cat_{category}_spin_{spin}_mass_{mass}",
    variable_pattern: str = "dnn_spin{spin}_mass{mass}",
    binning: tuple[int, float, float, str] | tuple[float, float, str] = (0.0, 1.0, "flats"),
    qcd_estimation: bool = True,
    n_parallel_read: int = 4,
    n_parallel_write: int = 2,
    cache_directory: str = "",
    skip_existing: bool = False,
) -> list[tuple[str, str, list[float]]]:
    # cast arguments to lists
    _spins = make_list(spin)
    _masses = make_list(mass)
    _categories = expand_categories(category)

    # split skim directories and sample names to filter and actual directories, both mapped to years
    filter_sample_names = {
        year: sample_names or []
        for (year, _), sample_names in skim_directories.items()
    }
    skim_directories = {
        year: skim_dir
        for year, skim_dir in skim_directories
    }

    # input checks
    for spin in _spins:
        assert spin in spins
    for mass in _masses:
        assert mass in masses
    for category in _categories:
        assert category in categories
    for year in skim_directories:
        assert year in eval_directories

    # get a list of all sample names per skim directory
    all_sample_names = {
        year: [
            dir_name
            for dir_name in os.listdir(skim_dir)
            if (
                os.path.isdir(os.path.join(skim_dir, dir_name)) and
                dir_is_skim_dir(dir_name)
            )
        ]
        for year, skim_dir in skim_directories.items()
    }

    # fiter by given sample names
    all_sample_names = {
        year: [
            sample_name
            for sample_name in sample_names
            if any(fnmatch(sample_name, pattern) for pattern in filter_sample_names[year] or ["*"])
        ]
        for year, sample_names in all_sample_names.items()
    }

    # get a mapping of process name to sample names
    sample_map: dict[str, dict[str, list]] = defaultdict(dict)
    all_matched_sample_names: dict[str, list[str]] = defaultdict(list)
    for process_name, process_data in processes.items():
        # skip signals that do not match any spin or mass
        if (
            process_data.get("signal", False) and
            (process_data["spin"] not in _spins or process_data["mass"] not in _masses)
        ):
            continue

        # match sample names
        for year, _sample_names in all_sample_names.items():
            matched_sample_names = []
            for sample_name in _sample_names:
                if any(fnmatch(sample_name, pattern) for pattern in process_data["sample_patterns"]):
                    if sample_name in matched_sample_names:
                        raise Exception(f"sample '{sample_name}' already matched by a previous process")
                    all_matched_sample_names[year].append(sample_name)
                    matched_sample_names.append(sample_name)
                    continue
            if not matched_sample_names:
                print(f"process '{process_name}' has no matched samples, skipping")
                continue
            sample_map[year][process_name] = matched_sample_names

    # ensure that the output directory exists
    output_directory = os.path.expandvars(os.path.expanduser(output_directory))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # prepare columns to read from klub files for the selection
    selection_columns = list(set(sum((
        categories[category]["selection"].flat_columns for category in _categories
    ), [])))

    # prepare dnn output columns
    dnn_output_columns = [
        variable_pattern.format(spin=spin, mass=mass)
        for spin, mass in itertools.product(_spins, _masses)
    ]

    # loading data
    print(f"going to load {sum(map(len, all_matched_sample_names.values()))} samples")
    sample_data = {
        year: {
            sample_name: load_sample_data(
                skim_directories[year],
                eval_directories[year],
                year,
                sample_name,
                selection_columns,
                dnn_output_columns,
                n_parallel=n_parallel_read,
                cache_directory=cache_directory,
            )
            for sample_name in sample_names
        }
        for year, sample_names in all_matched_sample_names.items()
    }

    # write each spin, mass and category combination
    datacard_args = []
    for spin, mass, category in itertools.product(_spins, _masses, _categories):
        datacard_args.append((
            sample_map,
            sample_data,
            spin,
            mass,
            category,
            output_directory,
            output_pattern.format(spin=spin, mass=mass, category=category),
            variable_pattern.format(spin=spin, mass=mass),
            binning,
            qcd_estimation,
            skip_existing,
        ))

    print(f"\nwriting datacard{'s' if len(datacard_args) > 1 else ''} ...")
    if n_parallel_write > 1:
        # run in parallel
        with ThreadPool(n_parallel_write) as pool:
            datacard_results = list(tqdm(
                pool.imap(_write_datacard_mp, datacard_args),
                total=len(datacard_args),
            ))
    else:
        datacard_results = list(tqdm(
            map(_write_datacard_mp, datacard_args),
            total=len(datacard_args),
        ))
    print("done")

    # write bin edges into a file
    bin_edges_file = os.path.join(output_directory, "bin_edges.json")
    # load them first when the file is existing
    all_bin_edges = {}
    if os.path.exists(bin_edges_file):
        with open(bin_edges_file, "r") as f:
            all_bin_edges = json.load(f)
    # update with new bin edges
    for args, res in zip(datacard_args, datacard_results):
        spin, mass, category = args[2:5]
        edges = res[2]
        key = f"{category}__s{spin}__m{mass}"
        # do not overwrite when edges are None (in case the datacard was skipped)
        if key in all_bin_edges and not edges:
            continue
        all_bin_edges[key] = edges
    # write them
    with open(bin_edges_file, "w") as f:
        json.dump(all_bin_edges, f, indent=4)

    return datacard_results


def _write_datacard(
    sample_map: dict[str, dict[str, list[str]]],
    sample_data: dict[str, dict[str, ak.Array]],
    spin: int,
    mass: int,
    category: str,
    output_directory: str,
    output_name: str,
    variable_name: str,
    binning: tuple[int, float, float, str] | tuple[float, float, str],
    qcd_estimation: bool,
    skip_existing: bool,
) -> tuple[str | None, str | None, list[float] | None]:
    cat_data = categories[category]

    # input checks
    assert len(binning) in [3, 4]
    if len(binning) == 3:
        x_min, x_max, binning_algo = binning
        n_bins = cat_data["n_bins"]
    else:
        n_bins, x_min, x_max, binning_algo = binning
    assert x_max > x_min
    assert binning_algo in {"equal", "flats", "flatsguarded"}

    # check if there is data provided for this category if it is bound to a year
    assert cat_data["year"] in list(luminosities.keys()) + [None]
    if cat_data["year"] is not None and not any(cat_data["year"] == year for year in sample_data):
        print(f"category {category} is bound to a year but no data was provided for that year")
        return (None, None)

    # prepare the output paths
    datacard_path = f"datacard_{output_name}.txt"
    shapes_path = f"shapes_{output_name}.root"
    abs_datacard_path = os.path.join(output_directory, datacard_path)
    abs_shapes_path = os.path.join(output_directory, shapes_path)

    # mp-safe directory creation
    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
        except:
            time.sleep(0.5)
            if not os.path.exists(output_directory):
                raise

    if skip_existing and os.path.exists(abs_datacard_path) and os.path.exists(abs_shapes_path):
        return datacard_path, shapes_path, None

    # prepare qcd estimation if requested
    if qcd_estimation:
        # can only do qcd estimation in *_os_iso categories
        if not category.endswith("_os_iso"):
            raise Exception(f"cannot estimate QCD in non os-iso category {category}")
        # find corresponding qcd regions:
        # os_iso   : signal region
        # os_noniso: region from where the shape is taken
        # ss_iso   : normalization numerator
        # ss_noniso: normalization denominator
        qcd_categories = {
            region_name: f"{category[:-len('_os_iso')]}_{region_name}"
            for region_name in ["os_noniso", "ss_iso", "ss_noniso", "os_iso"]
        }

    # define shape patterns to use in the datacard and shape file
    shape_patterns = {
        "nom": "cat_{category}/{process}",
        "nom_comb": "$CHANNEL/$PROCESS",
        "syst": "cat_{category}/{process}__{parameter}{direction}",
        "syst_comb": "$CHANNEL/$PROCESS__$SYSTEMATIC",
    }

    # reduce the sample_map in three steps:
    # - when the category is bound to a year, drop other years
    # - remove signal processes from the sample map that do not correspond to spin or mass
    # - remove data processes that are not meant to be included for the channel
    reduced_sample_map = defaultdict(dict)
    for year, _map in sample_map.items():
        if cat_data["year"] not in (None, year):
            continue
        for process_name, sample_names in _map.items():
            # skip some signals
            if (
                processes[process_name].get("signal", False) and
                (processes[process_name]["spin"] != spin or processes[process_name]["mass"] != mass)
            ):
                continue
            # skip some data
            if (
                processes[process_name].get("data", False) and
                cat_data["channel"] not in processes[process_name]["channels"]
            ):
                continue
            reduced_sample_map[year][process_name] = sample_names
    sample_map = reduced_sample_map

    # drop years from sample_data if not needed
    sample_data = {
        year: data
        for year, data in sample_data.items()
        if year in sample_map
    }

    # reversed map to assign processes to samples
    sample_processes = defaultdict(dict)
    for year, _map in sample_map.items():
        for process_name, sample_names in _map.items():
            sample_processes[year].update({sample_name: process_name for sample_name in sample_names})

    # apply qcd estimation category selections
    if qcd_estimation:
        qcd_data = {
            region_name: {
                year: {
                    sample_name: data[sample_name][categories[qcd_category]["selection"](data[sample_name], year=year)]
                    for sample_name, process_name in sample_processes[year].items()
                    # skip signal
                    if not processes[process_name].get("signal", False)
                }
                for year, data in sample_data.items()
            }
            for region_name, qcd_category in qcd_categories.items()
        }

    # apply the category selection to sample data
    sample_data = {
        year: {
            sample_name: data[sample_name][cat_data["selection"](data[sample_name], year=year)]
            for sample_name, process_name in sample_processes[year].items()
        }
        for year, data in sample_data.items()
    }

    # complain when nan's were found
    for year, data in sample_data.items():
        for sample_name, _data in data.items():
            for field in _data.fields:
                # skip fields other than the shape variables
                if not field.startswith(variable_name):
                    continue
                n_nonfinite = np.sum(~np.isfinite(_data[field]))
                if n_nonfinite:
                    print(
                        f"{n_nonfinite} / {len(_data)} of events in {sample_name} ({year}) after {category} "
                        f"selection are non-finite in variable {field}",
                    )

    # derive bin edges
    if binning_algo == "equal":
        bin_edges = np.linspace(x_min, x_max, n_bins + 1).tolist()
    else:  # flats or flatsguarded
        # get the signal values and weights
        signal_process_names = {
            year: [
                process_name
                for process_name in _map
                if processes[process_name].get("signal", False)
            ]
            for year, _map in sample_map.items()
        }
        for year, names in signal_process_names.items():
            if len(names) != 1:
                raise Exception(
                    f"either none or too many signal processes found for year {year} to obtain flats binning: {names}",
                )
        signal_process_name = {year: names[0] for year, names in signal_process_names.items()}

        # helper to get values of weights of a process
        def get_values_and_weights(process_name: str | dict[str, str], weight_scale: float | int = 1.0):
            if isinstance(process_name, str):
                process_name = {year: process_name for year in sample_data}

            def extract(getter):
                return ak.concatenate(
                    sum(
                        (
                            [getter(year, data, sample_name) for sample_name in sample_map[year][process_name[year]]]
                            for year, data in sample_data.items()
                        ),
                        [],
                    ),
                    axis=0,
                )

            values = extract(lambda year, data, sample_name: data[sample_name][variable_name])
            weights = extract(lambda year, data, sample_name: data[sample_name].full_weight_nominal * luminosities[year] * weight_scale)  # noqa

            # complain when values are out of bounds or non-finite
            outlier_mask = (values < x_min) | (values > x_max) | ~np.isfinite(values)
            if ak.any(outlier_mask):
                print(
                    f"  found {ak.sum(outlier_mask)} outliers in ({category},{spin},{mass}) for process {process_name}",
                )
                values = values[~outlier_mask]
                weights = weights[~outlier_mask]

            return np.asarray(values), np.asarray(weights)

        # helper to sort values and weights by values
        def sort_values_and_weights(values, weights, inc=True):
            sort_indices = np.argsort(values)
            values, weights = values[sort_indices], weights[sort_indices]
            return (values if inc else np.flip(values, axis=0)), (weights if inc else np.flip(weights, axis=0))

        hh_values, hh_weights = get_values_and_weights(signal_process_name, weight_scale=br_hh_bbtt)

        # distinguish non-guarded and guarded flats binnings from here on
        if binning_algo == "flats":
            # the number of bins cannot be larger than the amount of unique signal values
            _n_bins_max = len(set(hh_values))
            if n_bins > _n_bins_max:
                print(
                    f"  reducing n_bins from {n_bins} to {_n_bins_max} in ({category},{spin},{mass}) "
                    f"due to limited signal statistics of process {signal_process_name}",
                )
                n_bins = _n_bins_max
            if n_bins < 1:
                print(f"  do not write datacard in ({category},{spin},{mass})")
                return (None, None, None)
            # sort by increasing value
            hh_values, hh_weights = sort_values_and_weights(hh_values, hh_weights)
            # compute quantiles
            weighted_quantiles = (
                (np.cumsum(hh_weights) - 0.5 * hh_weights) /
                np.sum(hh_weights)
            )
            # obtain edges
            thresholds = np.linspace(x_min, x_max, n_bins + 1)[1:-1]
            inner_edges = np.interp(thresholds, weighted_quantiles, hh_values)
            bin_edges = [x_min] + inner_edges.tolist() + [x_max]
            # floating point protection, round to 5 digits and sort
            bin_edges = sorted(set(round(edge, 5) for edge in bin_edges))
            _n_bins_actual = len(bin_edges) - 1
            if _n_bins_actual < n_bins:
                print(
                    f"  reducing n_bins from {n_bins} to {_n_bins_actual} in ({category},{spin},{mass}) "
                    f"due to edge value rounding in process {signal_process_name}",
                )
                n_bins = _n_bins_actual

        else:  # flatsguarded
            #
            # step 1: data preparation
            #

            # get tt and dy data
            tt_values, tt_weights = get_values_and_weights("TT")
            dy_values, dy_weights = get_values_and_weights("DY")
            # create a record array with eight entries:
            # - value
            # - process (0: hh, 1: tt, 2: dy)
            # - hh_count_cs, tt_count_cs, dy_count_cs (cumulative sums of raw event counts)
            # - hh_weight_cs, tt_weight_cs, dy_weight_cs (cumulative sums of weights)
            all_values_list = [hh_values, tt_values, dy_values]
            rec = np.core.records.fromarrays(
                [
                    # value
                    (all_values := np.concatenate(all_values_list, axis=0)),
                    # process
                    np.concatenate([i * np.ones(len(v), dtype=np.int8) for i, v in enumerate(all_values_list)], axis=0),
                    # counts and weights per process
                    (izeros := np.zeros(len(all_values), dtype=np.int32)),
                    (fzeros := np.zeros(len(all_values), dtype=np.float32)),
                    izeros,
                    fzeros,
                    izeros,
                    fzeros,
                ],
                names="value,process,hh_count_cs,hh_weight_cs,tt_count_cs,tt_weight_cs,dy_count_cs,dy_weight_cs",
            )
            # insert counts and weights into columns for correct processes
            # (faster than creating arrays above which then get copied anyway when the recarray is created)
            HH, TT, DY = range(3)
            rec.hh_count_cs[rec.process == HH] = 1
            rec.tt_count_cs[rec.process == TT] = 1
            rec.dy_count_cs[rec.process == DY] = 1
            rec.hh_weight_cs[rec.process == HH] = hh_weights
            rec.tt_weight_cs[rec.process == TT] = tt_weights
            rec.dy_weight_cs[rec.process == DY] = dy_weights
            # sort by decreasing value to start binning from "the right" later on
            rec.sort(order="value")
            rec = np.flip(rec, axis=0)
            # replace counts and weights with their cumulative sums
            rec.hh_count_cs[:] = np.cumsum(rec.hh_count_cs)
            rec.tt_count_cs[:] = np.cumsum(rec.tt_count_cs)
            rec.dy_count_cs[:] = np.cumsum(rec.dy_count_cs)
            rec.hh_weight_cs[:] = np.cumsum(rec.hh_weight_cs)
            rec.tt_weight_cs[:] = np.cumsum(rec.tt_weight_cs)
            rec.dy_weight_cs[:] = np.cumsum(rec.dy_weight_cs)
            # eager cleanup
            del all_values, izeros, fzeros
            del hh_values, hh_weights
            del tt_values, tt_weights
            del dy_values, dy_weights
            # now, between any two possible discriminator values, we can easily extract the hh, tt and dy integrals,
            # as well as raw event counts without the need for additional, costly accumulation ops (sum, count, etc.),
            # but rather through simple subtraction of values at the respective indices instead

            #
            # step 2: binning
            #

            # determine the approximate hh yield per bin
            hh_yield_per_bin = rec.hh_weight_cs[-1] / n_bins
            # keep track of bin edges and the hh yield accumulated so far
            bin_edges = [x_max]
            hh_yield_binned = 0.0
            min_hh_yield = 1.0e-5
            # during binning, do not remove leading entries, but remember the index that denotes the start of the bin
            offset = 0
            # helper to extract a cumulative sum between the start offset (included) and the stop index (not included)
            get_integral = lambda cs, stop: cs[stop - 1] - (0 if offset == 0 else cs[offset - 1])
            # bookkeep reasons for stopping binning
            stop_reason = ""
            # start binning
            while len(bin_edges) < n_bins:
                # stopping condition 1: reached end of events
                if offset >= len(rec):
                    stop_reason = "no more events left"
                    break
                # stopping condition 2: remaining hh yield too small, so cause a background bin to be created
                remaining_hh_yield = rec.hh_weight_cs[-1] - hh_yield_binned
                if remaining_hh_yield < min_hh_yield:
                    stop_reason = "remaining signal yield insufficient"
                    break
                # find the index of the event that would result in a hh yield increase of more than the expected
                # per-bin yield; this index would mark the start of the next bin given all constraints are met
                if remaining_hh_yield >= hh_yield_per_bin:
                    threshold = hh_yield_binned + hh_yield_per_bin
                    next_idx = offset + np.where(rec.hh_weight_cs[offset:] > threshold)[0][0]
                else:
                    # special case: remaining hh yield smaller than the expected per-bin yield, so find the last event
                    next_idx = offset + np.where(rec.process[offset:] == HH)[0][-1] + 1
                # advance the index until backgrounds constraints are met
                while next_idx < len(rec):
                    # get the number of tt events and their yield
                    n_tt = get_integral(rec.tt_count_cs, next_idx)
                    y_tt = get_integral(rec.tt_weight_cs, next_idx)
                    # get the number of dy events and their yield
                    n_dy = get_integral(rec.dy_count_cs, next_idx)
                    y_dy = get_integral(rec.dy_weight_cs, next_idx)
                    # evaluate constraints
                    # TODO: potentially relax constraints here, e.g when there are 3 (4?) tt events, drop the constraint
                    #       on dy, and vice-versa
                    constraints_met = (
                        # tt and dy events
                        n_tt >= 1 and
                        n_dy >= 1 and
                        n_tt + n_dy >= 3 and
                        # yields must be positive to avoid negative sums of weights per process
                        y_tt > 0 and
                        y_dy > 0
                    )
                    if constraints_met:
                        # TODO: maybe also check if the background conditions are just barely met and advance next_idx
                        # to the middle between the current value and the next one that would change anything about the
                        # background predictions; this might be more stable as the current implementation can highly
                        # depend on the exact value of a single event (the one that tips the constraints over the edge
                        # to fulfillment)

                        # bin found, stop
                        break
                    # constraints not met, advance index to include the next tt or dy event and try again
                    next_bkg_indices = np.where(rec.process[next_idx:] != HH)[0]
                    if len(next_bkg_indices) == 0:
                        # no more background events left, move to the last position and let the stopping condition 3
                        # below handle the rest
                        next_idx = len(rec)
                    else:
                        next_idx += next_bkg_indices[0] + 1
                else:
                    # stopping condition 3: no more events left, so the last bin (most left one) does not fullfill
                    # constraints; however, this should practically never happen
                    stop_reason = "no more events left while trying to fulfill constraints"
                    break
                # next_idx found, update values
                edge_value = x_min if next_idx == 0 else float(rec.value[next_idx - 1:next_idx + 1].mean())
                bin_edges.append(max(min(edge_value, x_max), x_min))
                hh_yield_binned += get_integral(rec.hh_weight_cs, next_idx)
                offset = next_idx

            # make sure the minimum is included
            if bin_edges[-1] != x_min:
                if len(bin_edges) > n_bins:
                    raise RuntimeError(f"number of bins reached and initial bin edge is not x_min (edges: {bin_edges})")
                bin_edges.append(x_min)

            # reverse edges and optionally re-set n_bins
            bin_edges = sorted(set(bin_edges))
            n_bins_actual = len(bin_edges) - 1
            if n_bins_actual > n_bins:
                raise Exception("number of actual bins ended up larger than requested (implementation bug)")
            if n_bins_actual < n_bins:
                print(
                    f"  reducing n_bins from {n_bins} to {n_bins_actual} in ({category},{spin},{mass})\n"
                    f"    -> reason: {stop_reason or 'NO REASON!?'}",
                )
                n_bins = n_bins_actual

    #
    # write shapes
    #

    # transpose the sample_map so that we have a "process -> year -> sample_names" mapping
    process_map = defaultdict(dict)
    for year, _map in sample_map.items():
        for process_name, sample_names in _map.items():
            process_map[process_name][year] = sample_names

    # histogram structures
    # mapping (year, process) -> (nuisance, direction) -> hist
    hists: dict[tuple[str, str], dict[tuple[str, str], hist.Hist]] = defaultdict(dict)

    # keep track per year if at least one variation lead to a valid qcd estimation
    any_qcd_valid = {year: False for year in sample_map.keys()}

    # outer loop over variations
    for nuisance in shape_nuisances.values():
        # skip the nuisance when configured to skip
        if nuisance.skip:
            continue
        # skip shape nuisances that do not apply to the channel of this category
        if not nuisance.is_nominal and not nuisance.applies_to_channel(cat_data["channel"]):
            continue

        # loop over up/down variations (or just "" for nominal)
        for direction in nuisance.get_directions():
            hist_name = (
                variable_name
                if nuisance.is_nominal
                else f"{variable_name}_{nuisance.name}{direction}"
            )
            varied_variable_name = nuisance.get_varied_discriminator(variable_name, direction)
            varied_weight_field = nuisance.get_varied_full_weight(direction)

            # define histograms
            for process_name, _map in process_map.items():
                if not nuisance.applies_to_process(process_name):
                    continue

                _hist_name, _process_name = hist_name, process_name
                if processes[process_name].get("data", False):
                    _hist_name = _process_name = "data_obs"
                for year in _map.keys():
                    datacard_year = datacard_years[year]
                    full_hist_name = ShapeNuisance.create_full_name(_hist_name, year=datacard_year)
                    try:
                        h = hist.Hist.new.Variable(bin_edges, name=full_hist_name).Weight()
                    except:
                        print(f"creating histogram in ({category},{spin},{mass}) with edges {bin_edges} failed")
                        raise
                    hists[(year, _process_name)][(nuisance.get_combine_name(year=datacard_year), direction)] = h

            # fill histograms
            for process_name, _map in process_map.items():
                if not nuisance.applies_to_process(process_name):
                    continue

                _hist_name, _process_name = hist_name, process_name

                # for real data, skip if the nuisance is not nominal, and change hist name in case it is
                is_data = processes[process_name].get("data", False)
                if is_data:
                    if not nuisance.is_nominal:
                        continue
                    _hist_name = _process_name = "data_obs"

                # fill the histogram
                for year, sample_names in _map.items():
                    datacard_year = datacard_years[year]
                    full_hist_name = ShapeNuisance.create_full_name(_hist_name, year=datacard_year)
                    h = hists[(year, _process_name)][(nuisance.get_combine_name(year=datacard_year), direction)]
                    scale = 1 if is_data else luminosities[year]
                    if processes[process_name].get("signal", False):
                        scale *= br_hh_bbtt
                    for sample_name in sample_names:
                        weight = 1
                        if not is_data:
                            weight = sample_data[year][sample_name][varied_weight_field] * scale
                        h.fill(**{
                            full_hist_name: sample_data[year][sample_name][varied_variable_name],
                            "weight": weight,
                        })

                    # add epsilon values at positions where bin contents are not positive
                    nom = h.view().value
                    mask = nom <= 0
                    nom[mask] = 1.0e-5
                    h.view().variance[mask] = 1.0e-5

            # actual qcd estimation
            if qcd_estimation:
                # mapping year -> region -> hist
                qcd_hists: dict[str, dict[str, hist.Hist]] = defaultdict(dict)

                # create data-minus-background histograms in the 4 regions
                for region_name, _qcd_data in qcd_data.items():
                    for year, data in _qcd_data.items():
                        datacard_year = datacard_years[year]
                        # create a histogram that is filled with both data and negative background
                        full_hist_name = ShapeNuisance.create_full_name(hist_name, year=datacard_year)
                        h = hist.Hist.new.Variable(bin_edges, name=full_hist_name).Weight()
                        for sample_name, _data in data.items():
                            process_name = sample_processes[year][sample_name]
                            if not nuisance.applies_to_process(process_name):
                                continue

                            # skip signals
                            if processes[process_name].get("signal", False):
                                continue

                            is_data = processes[process_name].get("data", False)
                            if is_data:
                                # for data, always will the nominal values
                                h.fill(**{
                                    full_hist_name: _data[variable_name],
                                    "weight": 1,
                                })
                            else:
                                # for mc, use varied values and subtract them from data by using a negative fill weight
                                scale = luminosities[year]
                                h.fill(**{
                                    full_hist_name: _data[varied_variable_name],
                                    "weight": -1 * _data[varied_weight_field] * scale,
                                })
                        qcd_hists[year][region_name] = h

                # ABCD method per year
                # TODO: consider using averaging between the two options where the shape is coming from
                for year, region_hists in qcd_hists.items():
                    datacard_year = datacard_years[year]
                    # take shape from region "C"
                    h_qcd = region_hists["os_noniso"]
                    # get the intgral and its uncertainty from region "B"
                    num_val = region_hists["ss_iso"].sum().value
                    num_var = region_hists["ss_iso"].sum().variance
                    # get the intgral and its uncertainty from region "D"
                    denom_val = region_hists["ss_noniso"].sum().value
                    denom_var = region_hists["ss_noniso"].sum().variance
                    # stop if any yield is negative (due to more MC than data)
                    qcd_invalid = h_qcd.sum().value <= 0 or num_val <= 0 or denom_val <= 0
                    if not qcd_invalid:
                        # create the normalization correction including uncorrelated uncertainty propagation
                        corr_val = num_val / denom_val
                        corr_var = corr_val**2 * (num_var / num_val**2 + denom_var / denom_val**2)
                        # scale the shape by updating values and variances in-place
                        val = h_qcd.view().value
                        _var = h_qcd.view().variance
                        new_val = val * corr_val
                        _var[:] = new_val**2 * (_var / val**2 + corr_var / corr_val**2)
                        val[:] = new_val
                    else:
                        # zero-fill the histogram
                        h_qcd *= 0.0
                        # print(
                        #     f"  skipping QCD estimation in ({category},{nuisance.name}{direction},{year},{spin},{mass}) "  # noqa
                        #     f"due to negative yields in normalization regions: ss_iso={num_val}, ss_noniso={denom_val}",  # noqa
                        # )
                    # zero-fill
                    hval = h_qcd.view().value
                    hval[hval <= 0] = 1.0e-5
                    # store it
                    hists[(year, "QCD")][(nuisance.get_combine_name(year=datacard_year), direction)] = h_qcd
                    any_qcd_valid[year] |= not qcd_invalid

    # drop qcd shapes in years where no valid estimation was found
    for year, qcd_valid in any_qcd_valid.items():
        if not qcd_valid:
            print(
                f"  completely dropping QCD shape in ({category},{year},{spin},{mass}) as no valid shape could be "
                "created for any nuisance",
            )
            del hists[(year, "QCD")]

    # TODO: temporary fix
    # flip_downs = [
    #     "CMS_eff_t_stat0_DM0_2017",
    #     "CMS_eff_t_stat1_DM0_2017",
    #     "CMS_eff_t_systuncorrdmeras_DM0_2017",
    #     "CMS_eff_t_stat0_DM1_2017",
    #     "CMS_eff_t_stat1_DM1_2017",
    #     "CMS_eff_t_systuncorrdmeras_DM1",
    #     "CMS_eff_t_stat0_DM10",
    #     "CMS_eff_t_stat1_DM10",
    #     "CMS_eff_t_systuncorrdmeras_DM10",
    #     "CMS_eff_t_stat0_DM11",
    #     "CMS_eff_t_stat1_DM11",
    #     "CMS_eff_t_systuncorrdmeras_DM11",
    #     "CMS_eff_t_stat0gt140_2017",
    #     "CMS_eff_t_stat1gt140_2017",
    # ]
    # for _hists in hists.values():
    #     for name, direction in _hists.keys():
    #         if name not in flip_downs or direction != "down":
    #             continue
    #         d = _hists[(name, "down")]
    #         n = _hists[("nominal", "")]
    #         d_value = d.view().value
    #         d_value[:] = np.maximum(2 * n.view().value - d_value, 1.0e-5)

    # gather rates from nominal histograms
    rates = {
        (year, process_name): _hists[("nominal", "")].sum().value
        for (year, process_name), _hists in hists.items()
    }

    # create process names joining raw names and years
    full_process_names = {
        (year, process_name): (
            "{1}_{0}{2}".format(year, *m.groups())
            if (m := re.match(r"^(.+)(_h[^_]+)$", process_name))
            else f"{process_name}_{year}"
        )
        for year, process_name in hists
        if process_name != "data_obs"
    }

    # save nominal shapes
    # note: since /eos does not like write streams, first write to a tmp file and then copy
    def write(path):
        # create a dictionary of all histograms
        content = {}
        for (year, process_name), _hists in hists.items():
            for (full_nuisance_name, direction), h in _hists.items():
                # determine the full process name and optionally skip data for nuisances
                if process_name == "data_obs":
                    if full_nuisance_name != "nominal":
                        continue
                    full_process_name = process_name
                else:
                    full_process_name = full_process_names[(year, process_name)]

                if full_nuisance_name == "nominal":
                    shape_name = shape_patterns["nom"].format(category=category, process=full_process_name)
                else:
                    shape_name = shape_patterns["syst"].format(
                        category=category,
                        process=full_process_name,
                        parameter=full_nuisance_name,
                        direction=direction.capitalize(),
                    )
                # the shape name be unique when it's not data
                if shape_name in content:
                    if process_name != "data_obs":
                        raise Exception(f"shape name {shape_name} already exists in histograms to write")
                    # add on top
                    content[shape_name] += h
                else:
                    content[shape_name] = h

        # write all histogarms to file
        root_file = uproot.recreate(path)
        for key, h in content.items():
            root_file[key] = h

    with tempfile.NamedTemporaryFile(suffix=".root") as tmp:
        write(tmp.name)
        shutil.copy2(tmp.name, abs_shapes_path)

    #
    # write the text file
    #

    # prepare blocks and lines to write
    blocks = OrderedDict()
    separators = set()
    empty_lines = set()

    # counts block
    blocks["counts"] = [
        ("imax", "*"),
        ("jmax", "*"),
        ("kmax", "*"),
    ]
    separators.add("counts")

    # shape lines
    blocks["shapes"] = [
        ("shapes", "*", "*", shapes_path, shape_patterns["nom_comb"], shape_patterns["syst_comb"]),
    ]
    separators.add("shapes")

    # observations
    blocks["observations"] = [
        ("bin", f"cat_{category}"),
        ("observation", int(round(sum(r for (year, process_name), r in rates.items() if process_name == "data_obs")))),
    ]
    separators.add("observations")

    # expected rates
    exp_processes: list[tuple[str, str, str]] = sorted(
        [
            (year, process_name, full_process_name)
            for (year, process_name), full_process_name in full_process_names.items()
            if not processes[process_name].get("data", False)
        ],
        key=lambda p: processes[p[1]]["id"],
    )
    process_ids = {}
    last_signal_id, last_background_id = 1, 0
    for year, process_name, _ in exp_processes:
        if processes[process_name].get("signal", False):
            last_signal_id -= 1
            process_id = last_signal_id
        else:
            last_background_id += 1
            process_id = last_background_id
        process_ids[(year, process_name)] = process_id
    blocks["rates"] = [
        ("bin", *([f"cat_{category}"] * len(exp_processes))),
        ("process", *(full_process_name for _, _, full_process_name in exp_processes)),
        ("process", *(process_ids[(year, process_name)] for year, process_name, _ in exp_processes)),
        ("rate", *[f"{rates[(year, process_name)]:.4f}" for year, process_name, _ in exp_processes]),
    ]
    separators.add("rates")

    # list of years used for naming nuisances in datacards
    # (years for which data was loaded, but mapped to datacard years, e.g. dropping 2016APV)
    nuisance_years = []
    for year in sample_data.keys():
        datacard_year = datacard_years[year]
        if datacard_year not in nuisance_years:
            nuisance_years.append(datacard_year)

    # tabular-style parameters
    blocks["tabular_parameters"] = []
    # rate nuisances from the statistical model
    added_rate_params = []
    for param_name, effects in stat_model.items():
        year_dependent = ShapeNuisance.create_full_name(param_name, year="X") != param_name
        for nuisance_year in (nuisance_years if year_dependent else [None]):
            full_param_name = ShapeNuisance.create_full_name(param_name, year=nuisance_year)
            effect_line = []
            for year, process_name, _ in exp_processes:
                effect = "-"
                if not year_dependent or datacard_years[year] == nuisance_year:
                    for process_pattern, _effect in effects.items():
                        # process pattern matched?
                        negated = False
                        if process_pattern.startswith("!"):
                            negated = True
                            process_pattern = process_pattern[1:]
                        if fnmatch(process_name, process_pattern) == negated:
                            # not matched
                            continue
                        # the nuisance name might be year-independent, but its effect might be (and then it's a dict)
                        if isinstance(_effect, dict):
                            # the effect is a dict year_pattern -> effect
                            for year_pattern, __effect in _effect.items():
                                if fnmatch(year, year_pattern):
                                    _effect = __effect
                                    break
                            else:
                                _effect = "-"
                        # one the first match, store the effect and stop
                        effect = _effect
                        break
                effect_line.append(effect)
            if set(effect_line) != {"-"}:
                blocks["tabular_parameters"].append((full_param_name, "lnN", *effect_line))
                added_rate_params.append(full_param_name)
    # shape nuisances
    added_shape_params = []
    for nuisance in shape_nuisances.values():
        if nuisance.skip or nuisance.is_nominal or not nuisance.applies_to_channel(cat_data["channel"]):
            continue
        year_dependent = nuisance.get_combine_name(year="X") != nuisance.combine_name
        for nuisance_year in (nuisance_years if year_dependent else [None]):
            full_nuisance_name = nuisance.get_combine_name(year=nuisance_year)
            effect_line = []
            for year, process_name, _ in exp_processes:
                effect = "-"
                if not year_dependent or datacard_years[year] == nuisance_year:
                    # count occurances of the nuisance in the hists
                    count = sum(
                        1
                        for (nuisance_name, _) in hists[(year, process_name)]
                        if full_nuisance_name == nuisance_name
                    )
                    if count == 2:
                        effect = "1"
                    elif count != 0:
                        raise Exception(f"nuisance {full_nuisance_name} has {count} occurances in {year} {process_name}")
                effect_line.append(effect)
            if set(effect_line) != {"-"}:
                blocks["tabular_parameters"].append((full_nuisance_name, "shape", *effect_line))
                added_shape_params.append(full_nuisance_name)

    if blocks["tabular_parameters"]:
        empty_lines.add("tabular_parameters")

    # line-style parameters
    blocks["line_parameters"] = [
        ("rate_nuisances", "group", "=", " ".join(added_rate_params)),
    ]
    if added_shape_params:
        blocks["line_parameters"].append(("shape_nuisances", "group", "=", " ".join(added_shape_params)))
    if blocks["line_parameters"]:
        empty_lines.add("line_parameters")

    # mc stats
    blocks["mc_stats"] = [("*", "autoMCStats", 8)]

    # prettify blocks
    blocks["observations"] = align_lines(list(blocks["observations"]))
    if blocks["tabular_parameters"]:
        blocks["rates"], blocks["tabular_parameters"] = align_rates_and_parameters(
            list(blocks["rates"]),
            list(blocks["tabular_parameters"]),
        )
    else:
        blocks["rates"] = align_lines(list(blocks["rates"]))
    if blocks["line_parameters"]:
        blocks["line_parameters"] = align_lines(list(blocks["line_parameters"]))
    if blocks["mc_stats"]:
        blocks["mc_stats"] = align_lines(list(blocks["mc_stats"]))

    # write the blocks
    with open(abs_datacard_path, "w") as f:
        for block_name, lines in blocks.items():
            if not lines:
                continue

            # block lines
            for line in lines:
                if isinstance(line, (list, tuple)):
                    line = "  ".join(map(str, line))
                f.write(f"{line.strip()}\n")

            # block separator
            if block_name in separators:
                f.write(100 * "-" + "\n")
            elif block_name in empty_lines:
                f.write("\n")

    # return output paths
    return abs_datacard_path, abs_shapes_path, bin_edges


def _write_datacard_mp(args: tuple[Any]) -> tuple[str, str]:
    return _write_datacard(*args)


def align_lines(
    lines: Sequence[Any],
) -> list[str]:
    lines = [
        (line.split() if isinstance(line, str) else list(map(str, line)))
        for line in lines
    ]

    lengths = {len(line) for line in lines}
    if len(lengths) > 1:
        raise Exception(
            f"line alignment cannot be performed with lines of varying lengths: {lengths}",
        )

    # convert to rows and get the maximum width per row
    n_rows = list(lengths)[0]
    rows = [
        [line[j] for line in lines]
        for j in range(n_rows)
    ]
    max_widths = [
        max(len(s) for s in row)
        for row in rows
    ]

    # stitch back
    return [
        "  ".join(f"{s: <{max_widths[j]}}" for j, s in enumerate(line))
        for line in lines
    ]


def align_rates_and_parameters(
    rates: Sequence[Any],
    parameters: Sequence[Any],
) -> tuple[list[str], list[str]]:
    rates, parameters = [
        [
            (line.split() if isinstance(line, str) else list(map(str, line)))
            for line in lines
        ]
        for lines in [rates, parameters]
    ]

    # first, align parameter names and types on their own
    param_starts = align_lines([line[:2] for line in parameters])

    # prepend to parameter lines
    parameters = [([start] + line[2:]) for start, line in zip(param_starts, parameters)]

    # align in conjunction with rates
    n_rate_lines = len(rates)
    lines = align_lines(rates + parameters)

    return lines[:n_rate_lines], lines[n_rate_lines:]
