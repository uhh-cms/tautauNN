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
import itertools
import hashlib
import pickle
import tempfile
import shutil
from functools import reduce, wraps
from operator import mul
from collections import OrderedDict, defaultdict
from fnmatch import fnmatch
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from copy import deepcopy
from typing import Sequence, Any, Callable

from tqdm import tqdm
import numpy as np
import awkward as ak
import uproot
import hist

from tautaunn.util import transform_data_dir_cache
from tautaunn.config import masses, spins, klub_index_columns, luminosities, btag_wps


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
    "IdFakeSF_deep_2d",
    "PUjetID_SF",
    "bTagweightReshape",
]
klub_extra_columns = [
    # "DNNoutSM_kl_1",
]
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
        "sample_patterns": ["DY_amc_*"],
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
            "sample_patterns": [f"ggF_{resonance}_m{mass}"],
            "spin": spin,
            "mass": mass,
            "signal": True,
        }
        for mass in masses
        for spin, resonance in zip(spins, ["Radion", "*Graviton"])
    },
    "data_mu": {
        "sample_patterns": ["SingleMuon_Run201*", "Mu_201*"],
        "data": True,
        "channels": ["mutau"],
    },
    "data_egamma": {
        "sample_patterns": ["EGamma_Run201*", "Ele_201*"],
        "data": True,
        "channels": ["etau"],
    },
    "data_tau": {
        "sample_patterns": ["Tau_Run201*", "Tau_201*"],
        "data": True,
        "channels": ["tautau"],
    },
    # "data_mumu": {
    #     "sample_patterns": ["DoubleMuon_Run2017*"],
    #     "data": True,
    #     "channels": ["mutau", "etau", "tautau"],
    # },
    # "data_met": {
    #     "sample_patterns": ["MET_Run2017*"],
    #     "data": True,
    #     "channels": ["mutau", "etau", "tautau"],
    # },
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
    "pu_reweight": {"*": "1.01"},
    # year dependent (both the selection of nuisances and their effect depend on the year)
    "lumi_13TeV_2016": {"*": {"2016*": "1.010"}},
    "lumi_13TeV_2017": {"*": {"2017": "1.020"}},
    "lumi_13TeV_2018": {"*": {"2018": "1.015"}},
    "lumi_13TeV_1718": {"*": {"2017": "1.006", "2018": "1.002"}},
    "lumi_13TeV_correlated": {"*": {"2016*": "1.006", "2017": "1.009", "2018": "1.020"}},
}


def merge_dicts(*dicts):
    merged = dicts[0].__class__()
    for d in dicts:
        merged.update(deepcopy(d))
    return merged


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


# the isLeptrigger was needed only for etau and mutau at some point
# @selector(
#     needs=["pairType", "isLeptrigger"],
#     str_repr="(((pairType == 0) | (pairType == 1)) & (isLeptrigger == 1)) | (pairType == 2)",
# )
# def sel_trigger(array: ak.Array, **kwargs) -> ak.Array:
#     return (
#         (((array.pairType == 0) | (array.pairType == 1)) & (array.isLeptrigger == 1)) |
#         (array.pairType == 2)
#     )


# now it is needed for all channels
@selector(
    needs=["isLeptrigger"],
    str_repr="isLeptrigger == 1",
)
def sel_trigger(array: ak.Array, **kwargs) -> ak.Array:
    return array.isLeptrigger == 1


@selector(
    needs=["isLeptrigger", "pairType", "nleps", "nbjetscand"],
    str_repr=f"({sel_trigger.str_repr}) & ({sel_iso_first_lep.str_repr}) & (nleps == 0) & (nbjetscand > 1)",  # noqa
)
def sel_baseline(array: ak.Array, **kwargs) -> ak.Array:
    return (
        sel_trigger(array, **kwargs) &
        # including cut on first isolated lepton to reduce memory footprint
        # (note that this is not called "baseline" anymore by KLUB standards)
        sel_iso_first_lep(array, **kwargs) &
        (array.nleps == 0) &
        (array.nbjetscand > 1)
    )


@selector(
    needs=["isOS", "dau2_deepTauVsJet", sel_iso_first_lep],
)
def sel_region_os_iso(array: ak.Array, **kwargs) -> ak.Array:
    return sel_iso_first_lep(array, **kwargs) & (array.isOS == 1) & (array.dau2_deepTauVsJet >= 5)


@selector(
    needs=["isOS", "dau2_deepTauVsJet", sel_iso_first_lep],
)
def sel_region_ss_iso(array: ak.Array, **kwargs) -> ak.Array:
    return sel_iso_first_lep(array, **kwargs) & (array.isOS == 0) & (array.dau2_deepTauVsJet >= 5)


@selector(
    needs=["isOS", "dau2_deepTauVsJet", sel_iso_first_lep],
)
def sel_region_os_noniso(array: ak.Array, **kwargs) -> ak.Array:
    return sel_iso_first_lep(array, **kwargs) & (array.isOS == 1) & (array.dau2_deepTauVsJet < 5) & (array.dau2_deepTauVsJet >= 1)


@selector(
    needs=["isOS", "dau2_deepTauVsJet", sel_iso_first_lep],
)
def sel_region_ss_noniso(array: ak.Array, **kwargs) -> ak.Array:
    return sel_iso_first_lep(array, **kwargs) & (array.isOS == 0) & (array.dau2_deepTauVsJet < 5) & (array.dau2_deepTauVsJet >= 1)


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

    @selector(needs=["isBoosted"])
    def sel_boosted(array: ak.Array, **kwargs) -> ak.Array:
        return array.isBoosted == 1

    @selector(needs=["isVBF", "VBFjj_mass", "VBFjj_deltaEta"])
    def sel_vbf(array: ak.Array, **kwargs) -> ak.Array:
        return (
            (array.isVBF == 1) &
            (array.VBFjj_mass > 500) &
            (array.VBFjj_deltaEta > 3)
        )

    @selector(needs=["HHKin_mass"])
    def sel_mhh1_resolved(array: ak.Array, **kwargs) -> ak.Array:
        return (array.HHKin_mass >= 250) & (array.HHKin_mass < 335)

    @selector(needs=["HHKin_mass"])
    def sel_mhh2_resolved(array: ak.Array, **kwargs) -> ak.Array:
        return (array.HHKin_mass >= 335) & (array.HHKin_mass < 475)

    @selector(needs=["HHKin_mass"])
    def sel_mhh3_resolved(array: ak.Array, **kwargs) -> ak.Array:
        return (array.HHKin_mass >= 475) & (array.HHKin_mass < 725)

    @selector(needs=["HHKin_mass"])
    def sel_mhh4_resolved(array: ak.Array, **kwargs) -> ak.Array:
        return (array.HHKin_mass >= 725) & (array.HHKin_mass < 1100)

    @selector(needs=["HHKin_mass"])
    def sel_mhh5_resolved(array: ak.Array, **kwargs) -> ak.Array:
        return array.HHKin_mass >= 1100

    @selector(needs=["HHKin_mass"])
    def sel_mhh1_boosted(array: ak.Array, **kwargs) -> ak.Array:
        return (array.HHKin_mass >= 250) & (array.HHKin_mass < 625)

    @selector(needs=["HHKin_mass"])
    def sel_mhh2_boosted(array: ak.Array, **kwargs) -> ak.Array:
        return (array.HHKin_mass >= 625) & (array.HHKin_mass < 775)

    @selector(needs=["HHKin_mass"])
    def sel_mhh3_boosted(array: ak.Array, **kwargs) -> ak.Array:
        return (array.HHKin_mass >= 755) & (array.HHKin_mass < 1100)

    @selector(needs=["HHKin_mass"])
    def sel_mhh4_boosted(array: ak.Array, **kwargs) -> ak.Array:
        return array.HHKin_mass >= 1100

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

    mhh_sels_resolved = [
        sel_mhh1_resolved,
        sel_mhh2_resolved,
        sel_mhh3_resolved,
        sel_mhh4_resolved,
        sel_mhh5_resolved,
    ]

    mhh_sels_boosted = [
        sel_mhh1_boosted,
        sel_mhh2_boosted,
        sel_mhh3_boosted,
        sel_mhh4_boosted,
    ]

    @selector(needs=["bjet1_bID_deepFlavor", "bjet2_bID_deepFlavor"])
    def sel_btag_m(array: ak.Array, **kwargs) -> ak.Array:
        year = kwargs["year"]
        return (
            (array.bjet1_bID_deepFlavor > btag_wps[year]["medium"]) &
            (array.bjet2_bID_deepFlavor < btag_wps[year]["medium"])
        ) | (
            (array.bjet1_bID_deepFlavor < btag_wps[year]["medium"]) &
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

    @selector(needs=["tauH_SVFIT_mass", "bH_mass_raw"])
    def sel_mass_window_resolved(array: ak.Array, **kwargs) -> ak.Array:
        return (
            ((array.tauH_SVFIT_mass - 129.0) / 53.0)**2.0 +
            ((array.bH_mass_raw - 169.0) / 145.0)**2.0
        ) < 1.0

    @selector(needs=["tauH_SVFIT_mass", "bH_mass_raw"])
    def sel_mass_window_boosted(array: ak.Array, **kwargs) -> ak.Array:
        return (
            ((array.tauH_SVFIT_mass - 128.0) / 60.0)**2.0 +
            ((array.bH_mass_raw - 159.0) / 94.0)**2.0
        ) < 1.0

    @selector(
        needs=[sel_baseline, sel_channel],
        channel=channel,
    )
    def cat_baseline(array: ak.Array, **kwargs) -> ak.Array:
        return sel_baseline(array, **kwargs) & sel_channel(array, **kwargs)

    @selector(
        needs=[sel_baseline, sel_channel, sel_btag_m, sel_boosted, sel_vbf, sel_btag_m_first],
        channel=channel,
    )
    def cat_resolved_1b(array: ak.Array, **kwargs) -> ak.Array:
        return (
            sel_baseline(array, **kwargs) &
            sel_channel(array, **kwargs) &
            sel_btag_m(array, **kwargs) &
            ~sel_boosted(array, **kwargs) &
            ~(sel_vbf(array, **kwargs) & sel_btag_m_first(array, **kwargs))
        )

    @selector(
        needs=[cat_resolved_1b, sel_mass_window_resolved],
        channel=channel,
    )
    def cat_resolved_1b_mwc(array: ak.Array, **kwargs) -> ak.Array:
        return cat_resolved_1b(array, **kwargs) & sel_mass_window_resolved(array, **kwargs)

    @selector(
        needs=[sel_baseline, sel_channel, sel_btag_mm, sel_boosted, sel_vbf, sel_btag_m_first],
        channel=channel,
    )
    def cat_resolved_2b(array: ak.Array, **kwargs) -> ak.Array:
        return (
            sel_baseline(array, **kwargs) &
            sel_channel(array, **kwargs) &
            sel_btag_mm(array, **kwargs) &
            ~sel_boosted(array, **kwargs) &
            ~(sel_vbf(array, **kwargs) & sel_btag_m_first(array, **kwargs))
        )

    @selector(
        needs=[cat_resolved_2b, sel_mass_window_resolved],
        channel=channel,
    )
    def cat_resolved_2b_mwc(array: ak.Array, **kwargs) -> ak.Array:
        return cat_resolved_2b(array, **kwargs) & sel_mass_window_resolved(array, **kwargs)

    @selector(
        needs=[sel_baseline, sel_channel, sel_btag_ll, sel_boosted, sel_vbf, sel_btag_m_first],
        channel=channel,
    )
    def cat_boosted(array: ak.Array, **kwargs) -> ak.Array:
        return (
            sel_baseline(array, **kwargs) &
            sel_channel(array, **kwargs) &
            sel_btag_ll(array, **kwargs) &
            sel_boosted(array, **kwargs) &
            ~(sel_vbf(array, **kwargs) & sel_btag_m_first(array, **kwargs))
        )

    @selector(
        needs=[cat_boosted, sel_mass_window_resolved],
        channel=channel,
    )
    def cat_boosted_mwc(array: ak.Array, **kwargs) -> ak.Array:
        return cat_boosted(array, **kwargs) & sel_mass_window_resolved(array, **kwargs)

    @selector(
        needs=[sel_baseline, sel_channel, sel_btag_ll, sel_boosted, sel_vbf, sel_btag_m_first],
        channel=channel,
    )
    def cat_vbf(array: ak.Array, **kwargs) -> ak.Array:
        return (
            sel_baseline(array, **kwargs) &
            sel_channel(array, **kwargs) &
            sel_vbf(array, **kwargs) &
            sel_btag_m_first(array, **kwargs)
        )

    # create a dict of all selectors, but without subdivision into regions
    selectors = {
        "baseline": cat_baseline,
        "resolved1b": cat_resolved_1b,
        "resolved2b": cat_resolved_2b,
        "boosted": cat_boosted,
        "vbf": cat_vbf,
        # mass window cuts
        "resolved1bmwc": cat_resolved_1b_mwc,
        "resolved2bmwc": cat_resolved_2b_mwc,
        "boostedmwc": cat_boosted_mwc,
        # mhh bins in resolved and boosted
        **{
            f"resolved1bmhh{i + 1}": sel
            for i, sel in enumerate(sel_combinations(cat_resolved_1b, mhh_sels_resolved))
        },
        **{
            f"resolved2bmhh{i + 1}": sel
            for i, sel in enumerate(sel_combinations(cat_resolved_2b, mhh_sels_resolved))
        },
        **{
            f"boostedmhh{i + 1}": sel
            for i, sel in enumerate(sel_combinations(cat_boosted, mhh_sels_boosted))
        },
        # mass window cuts and mhh bins in resolved and boosted
        **{
            f"resolved1bmwcmhh{i + 1}": sel
            for i, sel in enumerate(sel_combinations(cat_resolved_1b_mwc, mhh_sels_resolved))
        },
        **{
            f"resolved2bmwcmhh{i + 1}": sel
            for i, sel in enumerate(sel_combinations(cat_resolved_2b_mwc, mhh_sels_resolved))
        },
        **{
            f"boostedmwcmhh{i + 1}": sel
            for i, sel in enumerate(sel_combinations(cat_boosted_mwc, mhh_sels_boosted))
        },
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
            "n_bins": 10,
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
) -> tuple[ak.Array, float]:
    # prepare expressions
    expressions = klub_index_columns + klub_weight_columns + klub_extra_columns + sel_baseline.flat_columns

    # add all columns potentially necessary for selections
    expressions += sum([
        cat["selection"].flat_columns
        for cat in categories.values()
    ], [])

    # make them unique
    expressions = list(set(expressions))

    # load the array
    f = uproot.open(os.path.join(skim_directory, f"SKIM_{sample_name}", file_name))
    array = f["HTauTauTree"].arrays(expressions=expressions, cut=sel_baseline.str_repr.strip())

    # compute the weight and complain when non-finite weights were found
    array = ak.with_field(
        array,
        reduce(mul, (array[c] for c in klub_weight_columns)),
        "full_weight",
    )
    mask = ~np.isfinite(array.full_weight)
    if np.any(mask):
        print(
            f"found {sum(mask)} ({100.0 * sum(mask) / len(mask):.2f}% of {len(mask)}) "
            f"non-finite weight values in sample {sample_name}, file {file_name}",
        )
        array = array[~mask]

    # drop columns
    for c in klub_weight_columns:
        array = ak.without_field(array, c)

    # also get the sum of generated MC weights
    # TODO: don't use for data as the hist is not always available there
    sum_gen_mc_weights = float(f["h_eff"].values()[0])

    return array, sum_gen_mc_weights


def load_dnn_file(
    eval_directory: str,
    sample_name: str,
    file_name: str,
    dnn_output_columns: list[str],
) -> ak.Array:
    # prepare expressions
    expressions = klub_index_columns + dnn_output_columns
    expressions = list(set(expressions))

    # load the array
    f = uproot.open(os.path.join(eval_directory, f"SKIM_{sample_name}", file_name))
    try:
        array = f["evaluation"].arrays(filter_name=expressions)
    except uproot.exceptions.KeyInFileError:
        array = f["hbtres"].arrays(filter_name=expressions)

    return array


def load_file(
    skim_directory: str,
    eval_directory: str,
    sample_name: str,
    file_name: str,
    dnn_output_columns: list[str],
) -> tuple[ak.Array, float]:
    # load the klub file
    klub_array, sum_gen_mc_weights = load_klub_file(skim_directory, sample_name, file_name)

    # load the dnn output file
    if eval_directory:
        dnn_array = load_dnn_file(eval_directory, sample_name, file_name, dnn_output_columns)

        # use klub array index to filter dnn array
        dnn_mask = np.isin(dnn_array[klub_index_columns], klub_array[klub_index_columns])
        if ak.sum(dnn_mask) != len(klub_array):
            klub_path = os.path.join(skim_directory, f"SKIM_{sample_name}", file_name)
            eval_path = os.path.join(eval_directory, f"SKIM_{sample_name}", file_name)
            raise Exception(
                f"the number of matching dnn array columns ({ak.sum(dnn_mask)}) does not match the "
                f"number of elements in the klub array ({len(klub_array)}) for file {file_name} "
                f"(klub: {klub_path}, dnn: {eval_path})",
            )
        dnn_array = dnn_array[dnn_mask]

        # exact (event, run, lumi) index check to make sure the order is identical as well
        matches = (
            (klub_array.EventNumber == dnn_array.EventNumber) &
            (klub_array.RunNumber == dnn_array.RunNumber) &
            (klub_array.lumi == dnn_array.lumi)
        )
        if not ak.all(matches):
            raise Exception(
                f"found event mismatch between klub and dnn files in {int(ak.sum(~matches))} cases "
                f"in file {file_name}",
            )

    # drop index columns
    array = klub_array
    for field in klub_index_columns:
        array = ak.without_field(array, field)

    # add dnn columns
    if eval_directory:
        for field in dnn_array.fields:
            if field in klub_index_columns:
                continue
            array = ak.with_field(array, dnn_array[field], field)

    return array, sum_gen_mc_weights


def load_file_mp(args: tuple[Any]) -> tuple[ak.Array, float]:
    return load_file(*args)


def get_cache_path(
    cache_directory: str,
    skim_directory: str,
    eval_directory: str,
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

    return os.path.join(cache_directory, f"data_{sample_name}_{h}.pkl")


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
    cache_path = get_cache_path(cache_directory, skim_directory, eval_directory, sample_name, dnn_output_columns or [])
    if cache_path and os.path.exists(cache_path):
        print("reading from cache")
        with open(cache_path, "rb") as f:
            array = pickle.load(f)

    else:
        # determine file names and build arguments for the parallel load implementation
        load_args = [
            (skim_directory, eval_directory, sample_name, file_name, dnn_output_columns or [])
            for file_name in os.listdir(os.path.join(eval_directory, f"SKIM_{sample_name}"))
            if fnmatch(file_name, "output_*.root")
        ]

        # run in parallel
        if n_parallel > 1:
            # run in parallel
            with ProcessPool(n_parallel) as pool:
                ret = list(tqdm(pool.imap(load_file_mp, load_args), total=len(load_args)))
        else:
            ret = list(tqdm(map(load_file_mp, load_args), total=len(load_args)))

        # combine values
        array = ak.concatenate([arr for arr, _ in ret], axis=0)
        sum_gen_mc_weights = sum(f for _, f in ret)
        del ret
        gc.collect()

        # update the full weight
        array = ak.with_field(array, array.full_weight / sum_gen_mc_weights, "full_weight")

        # add to cache?
        if cache_path:
            print("writing to cache")
            with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
                with open(tmp.name, "wb") as f:
                    pickle.dump(array, f)
                shutil.copy2(tmp.name, cache_path)

    # remove unnecessary columns
    keep_columns = dnn_output_columns + klub_extra_columns + ["full_weight"] + (selection_columns or [])
    for c in array.fields:
        if c not in keep_columns:
            array = ak.without_field(array, c)
            gc.collect()

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
    binning: tuple[int, float, float, str] | tuple[float, float, str] = (0.0, 1.0, "flat_s"),
    qcd_estimation: bool = True,
    n_parallel_read: int = 4,
    n_parallel_write: int = 2,
    cache_directory: str = "",
    skip_existing: bool = False,
) -> list[tuple[str, str]]:
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
            dir_name[5:]
            for dir_name in os.listdir(skim_dir)
            if (
                os.path.isdir(os.path.join(skim_dir, dir_name)) and
                dir_name.startswith("SKIM_")
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
            datacard_paths = list(tqdm(
                pool.imap(_write_datacard_mp, datacard_args),
                total=len(datacard_args),
            ))
    else:
        datacard_paths = list(tqdm(
            map(_write_datacard_mp, datacard_args),
            total=len(datacard_args),
        ))
    print("done")

    return datacard_paths


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
) -> tuple[str | None, str | None]:
    cat_data = categories[category]

    # input checks
    assert len(binning) in [3, 4]
    if len(binning) == 3:
        x_min, x_max, binning_algo = binning
        n_bins = cat_data["n_bins"]
    else:
        n_bins, x_min, x_max, binning_algo = binning
    assert x_max > x_min
    assert binning_algo in ["equal_distance", "flat_s"]

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

    if skip_existing and os.path.exists(abs_datacard_path) and os.path.exists(abs_shapes_path):
        return datacard_path, shapes_path

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
            # skip data for now as were are using fake data from background-only below
            if not processes[process_name].get("data", False)
        }
        for year, data in sample_data.items()
    }

    # complain when nan's were found
    for year, data in sample_data.items():
        for sample_name, _data in data.items():
            n_nonfinite = np.sum(~np.isfinite(_data[variable_name]))
            if n_nonfinite:
                print(
                    f"{n_nonfinite} / {len(_data)} of events in {sample_name} ({year}) after {category} "
                    "selection are non-finite (nan or inf)",
                )

    # prepare the scaling values, signal is scaled to 1pb * br
    # scale = cat_data["scale"]
    # signal_scale = scale * br_hh_bbtt

    # derive bin edges
    if binning_algo == "equal_distance":
        bin_edges = np.linspace(x_min, x_max, n_bins + 1).tolist()
    else:  # flat_s
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
                    f"either none or too many signal processes found for year {year} to obtain flat_s binning: {names}",
                )
        signal_process_name = {year: names[0] for year, names in signal_process_names.items()}
        signal_values = ak.concatenate(
            sum(
                ([
                    data[sample_name][variable_name]
                    for sample_name in sample_map[year][signal_process_name[year]]
                ] for year, data in sample_data.items()),
                [],
            ),
            axis=0,
        )
        signal_weights = ak.concatenate(
            sum(
                ([
                    data[sample_name].full_weight * luminosities[year] * br_hh_bbtt
                    for sample_name in sample_map[year][signal_process_name[year]]
                ] for year, data in sample_data.items()),
                [],
            ),
            axis=0,
        )
        # apply axis limits and complain
        outlier_mask = (signal_values < x_min) | (signal_values > x_max)
        if ak.any(outlier_mask):
            print(f"  found {ak.sum(outlier_mask)} outliers in ({category},{spin},{mass})")
        signal_values = signal_values[~outlier_mask]
        signal_weights = signal_weights[~outlier_mask]
        # the number of bins cannot be larger than the amount of unique signal values
        _n_bins_max = len(set(signal_values))
        if n_bins > _n_bins_max:
            print(
                f"  reducing n_bins from {n_bins} to {_n_bins_max} in ({category},{spin},{mass}) "
                f"due to limited signal statistics of process {signal_process_name}",
            )
            n_bins = _n_bins_max
        if n_bins < 1:
            print(f"  do not write datacard in ({category},{spin},{mass})")
            return (None, None)
        # sort by increasing value
        sort_indices = ak.argsort(signal_values)
        signal_values = signal_values[sort_indices]
        signal_weights = signal_weights[sort_indices]
        # compute quantiles
        weighted_quantiles = (
            (np.cumsum(signal_weights) - 0.5 * signal_weights) /
            np.sum(signal_weights)
        )
        # obtain edges
        thresholds = np.linspace(x_min, x_max, n_bins + 1)[1:-1]
        inner_edges = np.interp(thresholds, weighted_quantiles, signal_values)
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

    #
    # write shapes
    #

    # transpose the sample_map so that we have a "process -> year -> sample_names" mapping
    process_map = defaultdict(dict)
    for year, _map in sample_map.items():
        for process_name, sample_names in _map.items():
            process_map[process_name][year] = sample_names

    # fill histograms
    # (add zero bin offsets with 1e-5 and 100% error)
    hists: dict[tuple[name, name], hist.Hist] = {}
    for process_name, _map in process_map.items():
        # skip data
        if processes[process_name].get("data", False):
            continue

        for year, sample_names in _map.items():
            # create and fill the histogram
            h = hist.Hist.new.Variable(bin_edges, name=variable_name).Weight()
            scale = luminosities[year]
            if processes[process_name].get("signal", False):
                scale *= br_hh_bbtt
            for sample_name in sample_names:
                h.fill(**{
                    variable_name: sample_data[year][sample_name][variable_name],
                    "weight": sample_data[year][sample_name].full_weight * scale,
                })

            # add epsilon values at positions where bin contents are not positive
            nom = h.view().value
            mask = nom <= 0
            nom[mask] = 1.0e-5
            h.view().variance[mask] = 1.0e-5

            # store it
            hists[(year, process_name)] = h

    # actual qcd estimation
    if qcd_estimation:
        qcd_hists: dict[str, dict[str, hist.Hist]] = defaultdict(dict)
        for region_name, _qcd_data in qcd_data.items():
            for year, data in _qcd_data.items():
                # create a histogram that is filled with both data and negative background
                h = hist.Hist.new.Variable(bin_edges, name=variable_name).Weight()
                for sample_name, _data in data.items():
                    weight = 1
                    if not processes[sample_processes[year][sample_name]].get("data", False):
                        weight = -1 * _data.full_weight * scale
                    h.fill(**{variable_name: _data[variable_name], "weight": weight})
                qcd_hists[year][region_name] = h

        # ABCD method per year
        for year, _qcd_hists in qcd_hists.items():
            # take shape from region "C"
            h_qcd = _qcd_hists["os_noniso"]
            # get the intgral and its uncertainty from region "B"
            num_val = _qcd_hists["ss_iso"].sum().value
            num_var = _qcd_hists["ss_iso"].sum().variance
            # get the intgral and its uncertainty from region "D"
            denom_val = _qcd_hists["ss_noniso"].sum().value
            denom_var = _qcd_hists["ss_noniso"].sum().variance
            # stop if any yield is negative (due to more MC than data)
            if num_val <= 0 or denom_val <= 0:
                print(
                    f"  skipping QCD estimation in ({category},{year},{spin},{mass}) due to negative yields "
                    f"in normalization regions: ss_iso={num_val}, ss_noniso={denom_val}",
                )
                qcd_estimation = False
                break
            # create the normalization correction including uncorrelated uncertainty propagation
            corr_val = num_val / denom_val
            corr_var = corr_val**2 * (num_var / num_val**2 + denom_var / denom_val**2)
            # scale the shape by updating values and variances in-place
            val = h_qcd.view().value
            _var = h_qcd.view().variance
            new_val = val * corr_val
            _var[:] = new_val**2 * (_var / val**2 + corr_var / corr_val**2)
            val[:] = new_val
            # set negative values to epsilon values but keep potentially large uncertainties
            val[val <= 0] = 1.0e-5
            # store it
            hists[(year, "QCD")] = h_qcd

    # fake data using the sum of all backgrounds
    hists[(None, "data_obs")] = (
        hist.Hist.new
        .Variable(bin_edges, name=variable_name)
        .Double()
    )
    data_values = hists[(None, "data_obs")].view()
    for (year, process_name), h in hists.items():
        # add backgrounds
        if process_name != "data_obs" and not processes[process_name].get("signal", False):
            data_values += h.view().value
    data_values[...] = np.round(data_values)

    # gather rates
    rates = {
        (year, process_name): (h.sum() if process_name == "data_obs" else h.sum().value)
        for (year, process_name), h in hists.items()
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
        root_file = uproot.recreate(path)
        for (year, process_name), h in hists.items():
            full_name = process_name if year is None else full_process_names[(year, process_name)]
            shape_name = shape_patterns["nom"].format(category=category, process=full_name)
            root_file[shape_name] = h

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
        ("observation", int(round(rates[(None, "data_obs")]))),
    ]
    separators.add("observations")

    # expected rates
    exp_processes: list[tuple[str, str, str]] = sorted(
        [
            (year, process_name, full_name)
            for (year, process_name), full_name in full_process_names.items()
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
        ("process", *(full_name for _, _, full_name in exp_processes)),
        ("process", *(process_ids[(year, process_name)] for year, process_name, _ in exp_processes)),
        ("rate", *[f"{rates[(year, process_name)]:.4f}" for year, process_name, _ in exp_processes]),
    ]
    separators.add("rates")

    # tabular-style parameters
    blocks["tabular_parameters"] = []
    added_param_names = []
    for param_name, effects in stat_model.items():
        effect_line = []
        for year, process_name, full_name in exp_processes:
            for process_pattern, effect in effects.items():
                if isinstance(effect, dict):
                    # the effect is a dict year_pattern -> effect
                    for year_pattern, _effect in effect.items():
                        if fnmatch(year, year_pattern):
                            effect = _effect
                            break
                    else:
                        effect = "-"
                if fnmatch(process_name, process_pattern):
                    break
            else:
                effect = "-"
            effect_line.append(effect)
        if set(effect_line) != {"-"}:
            blocks["tabular_parameters"].append((param_name, "lnN", *effect_line))
            added_param_names.append(param_name)
    if blocks["tabular_parameters"]:
        empty_lines.add("tabular_parameters")

    # line-style parameters
    blocks["line_parameters"] = [
        ("model_nuisances", "group", "=", " ".join(added_param_names)),
    ]
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
                f.write(f"{line}\n")

            # block separator
            if block_name in separators:
                f.write(100 * "-" + "\n")
            elif block_name in empty_lines:
                f.write("\n")

    # return output paths
    return abs_datacard_path, abs_shapes_path


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
