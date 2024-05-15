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
import itertools
import hashlib
import pickle
import tempfile
import shutil
import pprint
from functools import reduce, wraps
from operator import mul
from collections import OrderedDict
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
from tautaunn.config import masses, spins, klub_index_columns, luminosities, btag_wps, pnet_wps
from tautaunn.binning_algorithms import uncertainty_driven, tt_dy_driven, flat_signal_ud


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

klub_weight_variation_map = {
    "trigSF" : [*[
            f"trigSF_{s}_{ud}"
            for s in ["ele", "met", "mu", "stau",
                    *[
                        f"DM{i}"
                        for i in [0, 1, 10, 11]
                    ]]
            for ud in ["up", "down"]
        ]
    ],
    "IdFakeSF_deep_2d" :[
        *[
            f"idFakeSF_etauFR_{be}_{ud}"
            for be in ["barrel", "endcap"]
            for ud in ["up", "down"]
        ],
        *[
            f"idFakeSF_mutauFR_eta{rng}_{ud}"
            for rng in ["0p4to0p8", "0p8to1p2", "1p2to1p7", "Gt1p7", "Lt0p4"]
            for ud in ["up", "down"]
        ],
        *[
            f"idFakeSF_tauid_2d_stat{i}_{ud}"
            for i in ['0', '1', 'gt140']
            for ud in ["up", "down"]
        ],
        *[
            f"idFakeSF_tauid_2d_systcorrdm{s}_{ud}"
            for s in ["eras", "uncorreras"]
            for ud in ["up", "down"]
        ],
        *[
            f"idFakeSF_tauid_2d_syst{s}_{ud}"
            for s in ["correrasgt140", "systuncorrdmeras"]
            for ud in ["up", "down"]
        ],
    ],
    "PUjetID_SF" : [*[
            f"PUjetID_SF_{s}{ud}"
            for s in ["",
                    *[f"{em}{s}"
                        for em in ["eff_", "mistag_"] 
                        for s in ["", 
                                *[f"eta_{ls}2p5_"
                                    for ls in ["l", "s"]]
                                ]
                        ]
                    ]
            for ud in ["up", "down"]
        ],
    ],
    "bTagweightReshape": [*[
            f"bTagweightReshape_jet{ud}{i}"
            for ud in ["up", "down"]
            for i in range(1, 12)
        ],
        *[
            f"bTagweightReshape_{lh}f{s}_{ud}"
            for lh in ["l", "h"]
            for s in ["", "stats1", "stats2"]
            for ud in ["up", "down"]
        ],
        *[
            f"bTagweightReshape_cferr{i}_{ud}"
            for i in range(1,3)
            for ud in ["up", "down"]
        ]
    ]
}

klub_weight_variation_columns = list(itertools.chain(*klub_weight_variation_map.values()))

klub_extra_columns = [
    # "DNNoutSM_kl_1",
]

dnn_shape_columns = [
    *[
        f"pdnn_m{mass}_s{spin}_hh_mes_{ud}"
        for mass in masses
        for spin in spins
        for ud in ["up", "down"]
    ],
    *[
        f"pdnn_m{mass}_s{spin}_hh_ees_{dm}_{ud}"
        for mass in masses
        for spin in spins
        for dm in ["DM0", "DM1"]
        for ud in ["up", "down"]
    ],
    *[
        f"pdnn_m{mass}_s{spin}_hh_tes_{dm}_{ud}"
        for mass in masses
        for spin in spins
        for dm in ["DM0", "DM1", "DM10", "DM11"]
        for ud in ["up", "down"]
    ],
    *[
        f"pdnn_m{mass}_s{spin}_hh_jes_{i}_{ud}"
        for mass in masses
        for spin in spins
        for i in range(1,12)
        for ud in ["up", "down"]
    ],
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
    # "data_mumu": {
    #     "sample_patterns": ["DoubleMuon_Run2017*"],
    #     "data": True,
    #     "channels": ["mutau", "etau", "tautau"],
    # },
    # TODO: enable MET? contains all channels?
    # "data_met": {
    #     "sample_patterns": ["MET*"],
    #     "data": True,
    #     "channels": ["mutau", "etau", "tautau"],
    # },
})
stat_model_common = {
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
}
stat_model_2016 = {
    "lumi_13TeV_2016": {"*": "1.010"},
    "lumi_13TeV_correlated": {"*": "1.006"},
}
stat_model_2017 = {
    "lumi_13TeV_2017": {"*": "1.020"},
    "lumi_13TeV_correlated": {"*": "1.009"},
    "lumi_13TeV_1718": {"*": "1.006"},
}
stat_model_2018 = {
    "lumi_13TeV_2018": {"*": "1.015"},
    "lumi_13TeV_correlated": {"*": "1.020"},
    "lumi_13TeV_1718": {"*": "1.002"},
}
categories = {}


def apply_outlier_mask(values, name, x_min, x_max, category, spin, mass, return_mask=False):
    outlier_mask = (values < x_min) | (values > x_max)
    if ak.any(outlier_mask):
        print(f"  found {ak.sum(outlier_mask)} {name} outliers in ({category},{spin},{mass})")
    return outlier_mask if return_mask else values[~outlier_mask]


def merge_dicts(*dicts):
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
def sel_iso_first_lep(array: ak.Array) -> ak.Array:
    return (
        ((array.pairType == 0) & (array.dau1_iso < 0.15)) |
        ((array.pairType == 1) & (array.dau1_eleMVAiso == 1)) |
        ((array.pairType == 2) & (array.dau1_deepTauVsJet >= 5))
    )


@selector(
    needs=["isLeptrigger", "isMETtrigger", "isSingleTautrigger"],
    str_repr="((isLeptrigger == 1) | (isMETtrigger == 1) | (isSingleTautrigger == 1))",
)
def sel_trigger(array: ak.Array) -> ak.Array:
    return (
        (array.isLeptrigger == 1) | (array.isMETtrigger == 1) | (array.isSingleTautrigger == 1)
    )


@selector(
    needs=[sel_trigger, sel_iso_first_lep, "nleps", "nbjetscand", "isBoosted"],
    str_repr=f"({sel_trigger.str_repr}) & ({sel_iso_first_lep.str_repr}) & (nleps == 0) & ((nbjetscand > 1) | (isBoosted == 1))",  # noqa
)
def sel_baseline(array: ak.Array) -> ak.Array:
    return (
        sel_trigger(array) &
        # including cut on first isolated lepton to reduce memory footprint
        # (note that this is not called "baseline" anymore by KLUB standards)
        sel_iso_first_lep(array) &
        (array.nleps == 0) &
        ((array.nbjetscand > 1) | (array.isBoosted == 1))
    )


@selector(
    needs=["isOS", "dau2_deepTauVsJet", sel_iso_first_lep],
)
def sel_region_os_iso(array: ak.Array) -> ak.Array:
    return sel_iso_first_lep(array) & (array.isOS != 0) & (array.dau2_deepTauVsJet >= 5)


@selector(
    needs=["isOS", "dau2_deepTauVsJet", sel_iso_first_lep],
)
def sel_region_ss_iso(array: ak.Array) -> ak.Array:
    return sel_iso_first_lep(array) & (array.isOS == 0) & (array.dau2_deepTauVsJet >= 5)


@selector(
    needs=["isOS", "dau2_deepTauVsJet", sel_iso_first_lep],
)
def sel_region_os_noniso(array: ak.Array) -> ak.Array:
    return sel_iso_first_lep(array) & (array.isOS != 0) & (array.dau2_deepTauVsJet < 5) & (array.dau2_deepTauVsJet >= 1)


@selector(
    needs=["isOS", "dau2_deepTauVsJet", sel_iso_first_lep],
)
def sel_region_ss_noniso(array: ak.Array) -> ak.Array:
    return sel_iso_first_lep(array) & (array.isOS == 0) & (array.dau2_deepTauVsJet < 5) & (array.dau2_deepTauVsJet >= 1)


region_sels = [
    sel_region_os_iso,
    sel_region_ss_iso,
    sel_region_os_noniso,
    sel_region_ss_noniso,
]


region_sel_names = ["os_iso", "ss_iso", "os_noniso", "ss_noniso"]


def category_factory(
    year: str,
    channel: str,
) -> dict[str, Callable]:
    pair_type = channels[channel]

    @selector(needs=["pairType"])
    def sel_channel(array: ak.Array) -> ak.Array:
        return array.pairType == pair_type

    @selector(needs=["isBoosted", "fatjet_particleNetMDJetTags_probXbb"])
    def sel_boosted(array: ak.Array) -> ak.Array:
        return (
            (array.isBoosted == 1) &
            (array.fatjet_particleNetMDJetTags_probXbb >= pnet_wps[year])
        )


    def sel_combinations(main_sel, sub_sels):
        def create(sub_sel):
            @selector(
                needs=[main_sel, sub_sel],
                year=year,
                channel=channel,
            )
            def func(array: ak.Array) -> ak.Array:
                return main_sel(array) & sub_sel(array)
            return func

        return [create(sub_sel) for sub_sel in sub_sels]


    @selector(needs=["bjet1_bID_deepFlavor", "bjet2_bID_deepFlavor"])
    def sel_btag_m(array: ak.Array) -> ak.Array:
        return (
            (array.bjet1_bID_deepFlavor > btag_wps[year]["medium"]) &
            (array.bjet2_bID_deepFlavor <= btag_wps[year]["medium"])
        ) | (
            (array.bjet1_bID_deepFlavor <= btag_wps[year]["medium"]) &
            (array.bjet2_bID_deepFlavor > btag_wps[year]["medium"])
        )

    @selector(needs=["bjet1_bID_deepFlavor", "bjet2_bID_deepFlavor"])
    def sel_btag_mm(array: ak.Array) -> ak.Array:
        return (
            (array.bjet1_bID_deepFlavor > btag_wps[year]["medium"]) &
            (array.bjet2_bID_deepFlavor > btag_wps[year]["medium"])
        )

    @selector(
        needs=[sel_baseline],
        year=year,
        channel=channel,
    )
    def cat_baseline(array: ak.Array) -> ak.Array:
        return sel_baseline(array)

    @selector(
        needs=[sel_baseline, sel_boosted, sel_btag_m],
        year=year,
        channel=channel,
    )
    def cat_resolved_1b(array: ak.Array) -> ak.Array:
        return (
            sel_baseline(array) &
            ~sel_boosted(array) &
            sel_btag_m(array)
        )

    @selector(
        needs=[sel_baseline, sel_boosted, sel_btag_mm],
        year=year,
        channel=channel,
    )
    def cat_resolved_2b(array: ak.Array) -> ak.Array:
        return (
            sel_baseline(array) &
            ~sel_boosted(array) &
            sel_btag_mm(array)
        )

    @selector(
        needs=[sel_baseline, sel_boosted],
        year=year,
        channel=channel,
    )
    def cat_boosted(array: ak.Array) -> ak.Array:
        return (
            sel_baseline(array) &
            sel_boosted(array)
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


for channel in channels:
    cats_2016APV = category_factory(year="2016APV", channel=channel)
    for name, sel in cats_2016APV.items():
        categories[f"2016APV_{channel}_{name}"] = {
            "selection": sel,
            "n_bins": 10,
            "scale": luminosities[sel.extra["year"]],
            "stat_model": merge_dicts(stat_model_common, stat_model_2016),
            **sel.extra,
        }
    cats_2016 = category_factory(year="2016", channel=channel)
    for name, sel in cats_2016.items():
        categories[f"2016_{channel}_{name}"] = {
            "selection": sel,
            "n_bins": 10,
            "scale": luminosities[sel.extra["year"]],
            "stat_model": merge_dicts(stat_model_common, stat_model_2016),
            **sel.extra,
        }
    cats_2017 = category_factory(year="2017", channel=channel)
    for name, sel in cats_2017.items():
        categories[f"2017_{channel}_{name}"] = {
            "selection": sel,
            "n_bins": 10,
            "scale": luminosities[sel.extra["year"]],
            "stat_model": merge_dicts(stat_model_common, stat_model_2017),
            **sel.extra,
        }
    cats_2018 = category_factory(year="2018", channel=channel)
    for name, sel in cats_2018.items():
        categories[f"2018_{channel}_{name}"] = {
            "selection": sel,
            "n_bins": 10,
            "scale": luminosities[sel.extra["year"]],
            "stat_model": merge_dicts(stat_model_common, stat_model_2018),
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
    expressions = klub_index_columns + klub_weight_columns + klub_weight_variation_colums + klub_extra_columns + sel_baseline.flat_columns

    # add all columns potentially necessary for selections
    expressions += sum([
        cat["selection"].flat_columns
        for cat in categories.values()
    ], [])

    # make them unique
    expressions = list(set(expressions))

    # load the array
    f = uproot.open(os.path.join(skim_directory, sample_name_to_skim_dir(sample_name), file_name))
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


def load_nominal_dnn(
    eval_directory: str,
    sample_name: str,
    file_name: str,
    dnn_output_columns: list[str],
) -> ak.Array:
    # prepare expressions
    expressions = klub_index_columns + dnn_output_columns
    expressions = list(set(expressions))

    # load the array
    f = uproot.open(os.path.join(eval_directory, sample_name_to_skim_dir(sample_name), file_name))
    array = f["hbtres"].arrays(filter_name=expressions)
    return array


def load_shapes_dnn(
    eval_directory: str,
    sample_name: str,
    file_name: str,
    dnn_output_columns: list[str],
) -> ak.Array:
    # prepare expressions
    expressions = klub_index_columns + dnn_output_columns + dnn_shape_columns
    expressions = list(set(expressions))

    # load the array
    f = uproot.open(os.path.join(eval_directory, sample_name_to_skim_dir(sample_name), file_name))
    array = f["hbtres"].arrays(filter_name=expressions)
    return array


def load_file(
    skim_directory: str,
    eval_directory: str,
    sample_name: str,
    file_name: str,
    dnn_output_columns: list[str],
) -> tuple[ak.Array, float, ak.Array]:
    # load the klub file
    klub_array, sum_gen_mc_weights = load_klub_file(skim_directory, sample_name, file_name)

    # load the dnn output file
    if eval_directory:
        dnn_array = load_nominal_dnn(eval_directory,
                                     sample_name,
                                     file_name.replace(".root", "_nominal.root"),
                                     dnn_output_columns)

        # use klub array index to filter dnn array
        dnn_mask = np.isin(dnn_array[klub_index_columns], klub_array[klub_index_columns])
        if ak.sum(dnn_mask) != len(klub_array):
            klub_path = os.path.join(skim_directory, sample_name_to_skim_dir(sample_name), file_name)
            eval_path = os.path.join(eval_directory, sample_name_to_skim_dir(sample_name), file_name)
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
        
        dnn_shapes_array = load_shapes_dnn(eval_directory,
                                           sample_name,
                                           file_name.replace(".root", "_systs.root"),
                                           dnn_output_columns)
        

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
        # shapes cannot be added to array because they have a different length
        #for field in dnn_shapes_array.fields:
            #if field in klub_index_columns:
                #continue
            #array = ak.with_field(array, dnn_shapes_array[field], field)
    return array, sum_gen_mc_weights, dnn_shapes_array


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
    sample_name: str,
    selection_columns: list[str] | None = None,
    dnn_output_columns: list[str] | None = None,
    n_parallel: int = 4,
    cache_directory: str = "",
) -> ak.Array:
    print(f"loading sample {sample_name} ...")

    # load from cache?
    cache_path = get_cache_path(cache_directory, skim_directory, eval_directory, sample_name, dnn_output_columns or [])
    if cache_path and os.path.exists(cache_path):
        print("reading from cache")
        print(cache_path)
        with open(cache_path, "rb") as f:
            array = pickle.load(f)

    else:
        # determine file names and build arguments for the parallel load implementation
        load_args = [
            (skim_directory, eval_directory, sample_name, file_name, dnn_output_columns or [])
            for file_name in os.listdir(os.path.join(skim_directory, sample_name_to_skim_dir(sample_name)))
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
        array = ak.concatenate([arr for arr, _, _ in ret], axis=0)
        sum_gen_mc_weights = sum(f for _, f, _ in ret)
        dnn_shapes_array = ak.concatenate([arr for _, _, arr in ret], axis=0)
        from IPython import embed; embed()
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
    make_list = lambda x: list(x) if isinstance(x, (list, tuple, set)) else [x]
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
    skim_directory: str,
    eval_directory: str,
    output_directory: str,
    output_pattern: str = "cat_{category}_spin_{spin}_mass_{mass}",
    variable_pattern: str = "dnn_spin{spin}_mass{mass}",
    sample_names: list[str] | None = None,
    binning: tuple[int, float, float, str] | tuple[float, float, str] = (0.0, 1.0, "flats"),
    uncertainty: float = 0.1,
    signal_uncertainty: float = 0.5,
    qcd_estimation: bool = True,
    n_parallel_read: int = 4,
    n_parallel_write: int = 2,
    cache_directory: str = "",
    skip_existing: bool = False,
) -> list[tuple[str, str]]:
    # cast arguments to lists
    make_list = lambda x: list(x) if isinstance(x, (list, tuple, set)) else [x]
    _spins = make_list(spin)
    _masses = make_list(mass)
    _categories = expand_categories(category)

    # input checks
    for spin in _spins:
        assert spin in spins
    for mass in _masses:
        assert mass in masses
    for category in _categories:
        assert category in categories

    # get the year
    year = categories[category]["year"]

    # get a list of all sample names in the skim directory
    all_sample_names = [
        dir_name.replace("SKIM_", "")
        for dir_name in os.listdir(skim_directory)
        if (
            os.path.isdir(os.path.join(skim_directory, dir_name)) and
            dir_is_skim_dir(dir_name)
        )
    ]

    # fiter by given sample names
    if sample_names:
        all_sample_names = [
            sample_name
            for sample_name in all_sample_names
            if any(fnmatch(sample_name, pattern) for pattern in sample_names)
        ]

    # get a mapping of process name to sample names
    sample_map = {}
    matched_sample_names = []
    for process_name, process_data in processes.items():
        # skip signals that do not match any spin or mass
        if (
            process_data.get("signal", False) and
            (process_data["spin"] not in _spins or process_data["mass"] not in _masses)
        ):
            continue

        # match sample names
        _sample_names = []
        for sample_name in all_sample_names:
            if any(fnmatch(sample_name, pattern) for pattern in process_data["sample_patterns"]):
                if sample_name in matched_sample_names:
                    raise Exception(f"sample '{sample_name}' already matched by a previous process")
                matched_sample_names.append(sample_name)
                _sample_names.append(sample_name)
                continue
        if not _sample_names:
            print(f"process '{process_name}' has no matched samples, skipping")
            continue
        sample_map[process_name] = _sample_names

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
        "DNNoutSM_kl_1",
        "dnn_output",
    ]
    for spin, mass in itertools.product(_spins, _masses):
        dnn_output_columns.append(variable_pattern.format(spin=spin, mass=mass))

    # prepare loading data
    print(f"going to load {len(matched_sample_names)} samples: {', '.join(matched_sample_names)}")
    #data_gen = (
        #load_sample_data(
            #skim_directory,
            #eval_directory,
            #sample_name,
            #selection_columns,
            #dnn_output_columns,
            #n_parallel=n_parallel_read,
            #cache_directory=cache_directory,
        #)
        #for sample_name in matched_sample_names
    #)

    # for debugging: just load data and free memory
    #print("just load data and exit")
    #for _ in data_gen:
        #gc.collect()
    #print("done loading data, exit")
    for sample_name in matched_sample_names:
        if sample_name == 'ZZZ':
            load_sample_data(
                skim_directory,
                eval_directory,
                sample_name,
                selection_columns,
                dnn_output_columns,
                n_parallel=n_parallel_read,
                cache_directory=cache_directory,
            )
            gc.collect()
            print("done")
    return

    # actually load
    sample_data = dict(zip(matched_sample_names, data_gen))

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
            output_pattern.format(year=year, spin=spin, mass=mass, category=category),
            variable_pattern.format(spin=spin, mass=mass),
            binning,
            uncertainty,
            signal_uncertainty,
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
    sample_map: dict[str, list[str]],
    sample_data: dict[str, ak.Array],
    spin: int,
    mass: int,
    category: str,
    output_directory: str,
    output_name: str,
    variable_name: str,
    binning: tuple[int, float, float, str] | tuple[float, float, str],
    uncertainty: float,
    signal_uncertainty: float,
    qcd_estimation: bool,
    skip_existing: bool,
) -> tuple[str | None, str | None]:
    # input checks
    assert len(binning) in [3, 4]
    if len(binning) == 3:
        x_min, x_max, binning_algo = binning
        n_bins = categories[category]["n_bins"]
    else:
        n_bins, x_min, x_max, binning_algo = binning
    assert x_max > x_min
    assert binning_algo in ["equal_distance", "flat_s", "ud", "ud_flats", "tt_dy_driven"]

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

    # remove signal processes from the sample map that do not correspond to spin or mass
    sample_map = {
        process_name: sample_names
        for process_name, sample_names in sample_map.items()
        if (
            # background or data
            not processes[process_name].get("signal", False) or
            # matching signal
            (processes[process_name]["spin"] == spin and processes[process_name]["mass"] == mass)
        )
    }

    # remove data processes from the sample map that are not meant to be included for the channel
    sample_map = {
        process_name: sample_names
        for process_name, sample_names in sample_map.items()
        if (
            # signal or background
            not processes[process_name].get("data", False) or
            # data with matching channel
            categories[category]["channel"] in processes[process_name]["channels"]
        )
    }

    # reversed map to assign processes to samples
    sample_processes = {}
    for process_name, sample_names in sample_map.items():
        sample_processes.update({sample_name: process_name for sample_name in sample_names})

    # apply qcd estimation category selections
    if qcd_estimation:
        qcd_data = {
            region_name: {
                sample_name: sample_data[sample_name][categories[qcd_category]["selection"](sample_data[sample_name])]
                for sample_name, process_name in sample_processes.items()
                # skip signal
                if not processes[process_name].get("signal", False)
            }
            for region_name, qcd_category in qcd_categories.items()
        }

    # apply the category selection to sample data
    sample_data = {
        sample_name: sample_data[sample_name][categories[category]["selection"](sample_data[sample_name])]
        for sample_name, process_name in sample_processes.items()
        # skip data for now as were are using fake data from background-only below
        if not processes[process_name].get("data", False)
    }

    # complain when nan's were found
    for sample_name, data in sample_data.items():
        n_nonfinite = np.sum(~np.isfinite(data[variable_name]))
        if n_nonfinite:
            print(
                f"{n_nonfinite} / {len(data)} of events in {sample_name} after {category} "
                "selection are non-finite (nan or inf)",
            )

    # prepare the scaling values, signal is scaled to 1pb * br
    scale = categories[category]["scale"]
    signal_scale = scale * br_hh_bbtt

    # derive bin edges
    if binning_algo == "equal_distance":
        bin_edges = np.linspace(x_min, x_max, n_bins + 1).tolist()
    elif binning_algo == "ud_flats":  # uncertainty-driven flat_s
        # if uncertainty is None:
        #     raise Exception("uncertainty must be specified for uncertainty-driven binning")
        # get the signal values
        signal_process_names = [
            process_name
            for process_name in sample_map
            # dict.get() returns the key if it exits, otherwise the default value (False here)
            if processes[process_name].get("signal", False)
        ]
        if len(signal_process_names) != 1:
            raise Exception(
                "either none or too many signal processes found to obtain uncertainty_driven binning: "
                f"{signal_process_names}",
            )
        signal_process_name = signal_process_names[0]
        signal_values = ak.concatenate([
            sample_data[sample_name][variable_name]
            for sample_name in sample_map[signal_process_name]
        ], axis=0)
        signal_weights = ak.concatenate([
            sample_data[sample_name].full_weight * signal_scale
            for sample_name in sample_map[signal_process_name]
        ], axis=0)
        # get the background values
        bkgd_process_names = [
            process_name
            for process_name in sample_map
            # dict.get() returns the key if it exits, otherwise the default value (False here)
            if (not processes[process_name].get("signal", False)) and (not processes[process_name].get("data", False))
        ]
        bkgd_sample_names = set()
        for process in bkgd_process_names:
            bkgd_sample_names |= set(sample_map[process])
        bkgd_values = ak.concatenate([
            sample_data[sample_name][variable_name]
            for sample_name in bkgd_sample_names
        ], axis=0)
        # apply axis limits and complain
        outlier_mask_sig = apply_outlier_mask(
            signal_values, "signal", x_min, x_max, category, spin, mass, return_mask=True,
        )
        signal_values = signal_values[~outlier_mask_sig]
        signal_weights = signal_weights[~outlier_mask_sig]
        bkgd_values = apply_outlier_mask(bkgd_values, "bkgd", x_min, x_max, category, spin, mass)
        # get tt and dy values
        tt_values = ak.concatenate([
            sample_data[sample_name][variable_name]
            for sample_name in sample_map["TT"]
        ], axis=0)
        dy_values = ak.concatenate([
            sample_data[sample_name][variable_name]
            for sample_name in sample_map["DY"]
        ], axis=0)
        # apply axis limits and complain
        tt_values = apply_outlier_mask(tt_values, "tt", x_min, x_max, category, spin, mass)
        dy_values = apply_outlier_mask(dy_values, "dy", x_min, x_max, category, spin, mass)
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
        bin_edges = flat_signal_ud(
            signal_values=signal_values,
            signal_weights=signal_weights,
            bkgd_values=bkgd_values,
            tt_values=tt_values,
            dy_values=dy_values,
            uncertainty=uncertainty,
            x_min=x_min,
            x_max=x_max,
            n_bins=n_bins,
        )
    elif binning_algo == "tt_dy_driven":
        if uncertainty is None:
            raise Exception("uncertainty must be specified for uncertainty-driven binning")
        # get the signal values
        signal_process_names = [
            process_name
            for process_name in sample_map
            # dict.get() returns the key if it exits, otherwise the default value (False here)
            if processes[process_name].get("signal", False)
        ]
        if len(signal_process_names) != 1:
            raise Exception(
                "either none or too many signal processes found to obtain tt_dy_driven binning: "
                f"{signal_process_names}",
            )
        signal_process_name = signal_process_names[0]
        signal_values = ak.concatenate([
            sample_data[sample_name][variable_name]
            for sample_name in sample_map[signal_process_name]
        ], axis=0)
        # get tt and dy values
        tt_values = ak.concatenate([
            sample_data[sample_name][variable_name]
            for sample_name in sample_map["TT"]
        ], axis=0)
        dy_values = ak.concatenate([
            sample_data[sample_name][variable_name]
            for sample_name in sample_map["DY"]
        ], axis=0)
        # apply axis limits and complain
        tt_values = apply_outlier_mask(tt_values, "tt", x_min, x_max, category, spin, mass)
        dy_values = apply_outlier_mask(dy_values, "dy", x_min, x_max, category, spin, mass)
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
        bin_edges = tt_dy_driven(
            signal_values=signal_values,
            dy_values=dy_values,
            tt_values=tt_values,
            uncertainty=uncertainty,
            signal_uncertainty=signal_uncertainty,
            mode="min",
            n_bins=n_bins,
            x_min=x_min,
            x_max=x_max,
        )
    elif binning_algo == "ud":  # uncertainty-driven
        if uncertainty is None:
            raise Exception("uncertainty must be specified for uncertainty-driven binning")
        # get the signal values
        signal_process_names = [
            process_name
            for process_name in sample_map
            # dict.get() returns the key if it exits, otherwise the default value (False here)
            if processes[process_name].get("signal", False)
        ]
        if len(signal_process_names) != 1:
            raise Exception(
                "either none or too many signal processes found to obtain uncertainty_driven binning: "
                f"{signal_process_names}",
            )
        signal_process_name = signal_process_names[0]
        signal_values = ak.concatenate([
            sample_data[sample_name][variable_name]
            for sample_name in sample_map[signal_process_name]
        ], axis=0)
        # get the background values
        bkgd_process_names = [
            process_name
            for process_name in sample_map
            # dict.get() returns the key if it exits, otherwise the default value (False here)
            if (not processes[process_name].get("signal", False)) and (not processes[process_name].get("data", False))
        ]
        bkgd_sample_names = set()
        for process in bkgd_process_names:
            bkgd_sample_names |= set(sample_map[process])
        bkgd_values = ak.concatenate([
            sample_data[sample_name][variable_name]
            for sample_name in bkgd_sample_names
        ], axis=0)
        # get tt and dy values
        tt_values = ak.concatenate([
            sample_data[sample_name][variable_name]
            for sample_name in sample_map["TT"]
        ], axis=0)
        dy_values = ak.concatenate([
            sample_data[sample_name][variable_name]
            for sample_name in sample_map["DY"]
        ], axis=0)
        # apply axis limits and complain
        signal_values = apply_outlier_mask(signal_values, "signal", x_min, x_max, category, spin, mass)
        bkgd_values = apply_outlier_mask(bkgd_values, "bkgd", x_min, x_max, category, spin, mass)
        tt_values = apply_outlier_mask(tt_values, "tt", x_min, x_max, category, spin, mass)
        dy_values = apply_outlier_mask(dy_values, "dy", x_min, x_max, category, spin, mass)
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
        bin_edges = uncertainty_driven(
            signal_values=signal_values,
            bkgd_values=bkgd_values,
            tt_values=tt_values,
            dy_values=dy_values,
            bkgd_uncertainty=uncertainty,
            signal_uncertainty=signal_uncertainty,
            n_bins=n_bins,
            x_min=x_min,
            x_max=x_max,
        )
    else:  # flat_s
        # get the signal values and weights
        signal_process_names = [
            process_name
            for process_name in sample_map
            if processes[process_name].get("signal", False)
        ]
        if len(signal_process_names) != 1:
            raise Exception(
                "either none or too many signal processes found to obtain flat_s binning: "
                f"{signal_process_names}",
            )
        signal_process_name = signal_process_names[0]
        signal_values = ak.concatenate([
            sample_data[sample_name][variable_name]
            for sample_name in sample_map[signal_process_name]
        ], axis=0)
        signal_weights = ak.concatenate([
            sample_data[sample_name].full_weight * signal_scale
            for sample_name in sample_map[signal_process_name]
        ], axis=0)
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

    # fill histograms
    # (add zero bin offsets with 1e-5 and 100% error)
    hists = {}
    for process_name, sample_names in sample_map.items():
        # skip data
        if processes[process_name].get("data", False):
            continue

        # create and fill the histogram
        h = hist.Hist.new.Variable(bin_edges, name=variable_name).Weight()
        _scale = signal_scale if processes[process_name].get("signal", False) else scale
        for sample_name in sample_names:
            h.fill(**{
                variable_name: sample_data[sample_name][variable_name],
                "weight": sample_data[sample_name].full_weight * _scale,
            })

        # add epsilon values at positions where bin contents are not positive
        nom = h.view().value
        mask = nom <= 0
        nom[mask] = 1.0e-5
        h.view().variance[mask] = 1.0e-5

        # store it
        hists[process_name] = h

    # actual qcd estimation
    if qcd_estimation:
        qcd_hists = {}
        for region_name, _qcd_data in qcd_data.items():
            # create a histogram that is filled with both data and negative background
            h = hist.Hist.new.Variable(bin_edges, name=variable_name).Weight()
            for sample_name, data in _qcd_data.items():
                weight = 1
                if not processes[sample_processes[sample_name]].get("data", False):
                    weight = -1 * data.full_weight * scale
                h.fill(**{variable_name: data[variable_name], "weight": weight})
            qcd_hists[region_name] = h

        # ABCD method
        # take shape from region "C"
        h_qcd = qcd_hists["os_noniso"]
        # get the intgral and its uncertainty from region "B"
        num_val = qcd_hists["ss_iso"].sum().value
        num_var = qcd_hists["ss_iso"].sum().variance
        # get the intgral and its uncertainty from region "D"
        denom_val = qcd_hists["ss_noniso"].sum().value
        denom_var = qcd_hists["ss_noniso"].sum().variance
        # stop if any yield is negative (due to more MC than data)
        if num_val <= 0 or denom_val <= 0:
            print(
                f"  skipping QCD estimation in ({category},{spin},{mass}) due to negative yields "
                f"in normalization regions: ss_iso={num_val}, ss_noniso={denom_val}",
            )
        else:
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
            hists["QCD"] = h_qcd

    # fake data using the sum of all backgrounds
    hists["data_obs"] = (
        hist.Hist.new
        .Variable(bin_edges, name=variable_name)
        .Double()
    )
    data_values = hists["data_obs"].view()
    for process_name, h in hists.items():
        # add backgrounds
        if process_name != "data_obs" and not processes[process_name].get("signal", False):
            data_values += h.view().value
    data_values[...] = np.round(data_values)

    # gather rates
    rates = {
        process_name: (h.sum() if process_name == "data_obs" else h.sum().value)
        for process_name, h in hists.items()
    }

    # save nominal shapes
    # note: since /eos does not like write streams, first write to a tmp file and then copy
    def write(path):
        root_file = uproot.recreate(path)
        for process_name, h in hists.items():
            shape_name = shape_patterns["nom"].format(category=category, process=process_name)
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
        ("observation", int(round(rates["data_obs"]))),
    ]
    separators.add("observations")

    # expected rates
    exp_processes = sorted(
        [
            process_name
            for process_name in sample_map
            if not processes[process_name].get("data", False)
        ],
        key=lambda p: processes[p]["id"],
    )
    if "QCD" in hists and "QCD" not in exp_processes:
        exp_processes.append("QCD")
    blocks["rates"] = [
        ("bin", *([f"cat_{category}"] * len(exp_processes))),
        ("process", *exp_processes),
        ("process", *[processes[process_name]["id"] for process_name in exp_processes]),
        ("rate", *[f"{rates[process_name]:.4f}" for process_name in exp_processes]),
    ]
    separators.add("rates")

    # tabular-style parameters
    stat_model = categories[category]["stat_model"]
    blocks["tabular_parameters"] = []
    added_param_names = []
    for param_name, effects in stat_model.items():
        effect_line = []
        for process_name in exp_processes:
            for process_pattern, effect in effects.items():
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


def main():
    from argparse import ArgumentParser

    csv = lambda s: [_s.strip() for _s in s.strip().split(",")]
    csv_int = lambda s: list(map(int, csv(s)))

    parser = ArgumentParser(
        description="write datacards for the hh->bbtt analysis",
    )
    parser.add_argument(
        "--spin",
        "-s",
        type=csv_int,
        default="0",
        help="comma-separated list of spins to use; default: 0",
    )
    parser.add_argument(
        "--mass",
        "-m",
        type=csv_int,
        default=",".join(map(str, masses)),
        help="comma-separated list of masses to use; default: all masses",
    )
    parser.add_argument(
        "--categories",
        "-c",
        type=csv,
        default=(default_cats := "2017_*tau_resolved?b_os_iso,2017_*tau_boosted_os_iso"),
        help=f"comma separated list of categories or patterns; default: {default_cats}",
    )
    parser.add_argument(
        "--no-qcd",
        action="store_true",
        help="disable the QCD estimation; default: False",
    )
    parser.add_argument(
        "--binning",
        "-b",
        choices=("equal", "flats", "ud_flats", "ud", "tt_dy_driven"),
        default="equal",
        help="binning strategy to use; default: equal",
    )
    parser.add_argument(
        "--uncertainty",
        "-u",
        type=float,
        default=None,
        help="uncertainty to use for uncertainty-driven binning; default: None",
    )
    parser.add_argument(
        "--signal-uncertainty",
        type=float,
        default=0.5,
        help="signal uncertainty to use for uncertainty-driven binning; default: 0.5",
    )
    parser.add_argument(
        "--n-bins",
        "-n",
        type=int,
        default=10,
        help="number of bins to use; default: 10",
    )
    parser.add_argument(
        "--skim-directory",
        default=os.environ["TN_SKIMS_2017"],
        help="dnn evaluation directory; default: $TN_SKIMS_2017",
    )
    parser.add_argument(
        "--eval-directory",
        default="",
        help="dnn evaluation directory; default: empty",
    )
    parser.add_argument(
        "--variable",
        "-v",
        default="DNNoutSM_kl_1",
        help="variable to use; can also contain '{spin}' and '{mass}'; default: DNNoutSM_kl_1",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        default=(default_path := os.getenv("DC_OUTPUT_PATH", "/eos/user/m/mrieger/hhres_dnn_datacards/cards")),
        help=f"output directory; default: {default_path}",
    )
    parser.add_argument(
        "--output-label",
        "-l",
        default=(default_label := "{binning}{n_bins}_{qcd_str}"),
        help="output label (name of last directory); can also contain '{binning}', '{n_bins}', '{qcd_str}'; "
        f"default: {default_label}",
    )
    parser.add_argument(
        "--cache-directory",
        default=(default_cache := os.getenv("DC_CACHE_PATH", "/eos/user/m/mrieger/hhres_dnn_datacards/cache")),
        help=f"cache directory; default: {default_cache}",
    )
    parser.add_argument(
        "--parallel-read",
        "-r",
        type=int,
        default=4,
        help="number of parallel processes to use for reading input files; default: 4",
    )
    parser.add_argument(
        "--parallel-write",
        "-w",
        type=int,
        default=4,
        help="number of parallel processes to use for writing cards; default: 4",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip writing datacards that already exist; default: False",
    )

    args = parser.parse_args()

    # buid the full output directory
    output_directory = os.path.join(
        args.output_path,
        args.output_label.format(
            binning=args.binning,
            n_bins=args.n_bins,
            qcd_str="noqcd" if args.no_qcd else "qcd",
        ),
    )

    # prepare kwargs
    kwargs = dict(
        spin=args.spin,
        mass=args.mass,
        category=args.categories,
        variable_pattern=args.variable,
        skim_directory=args.skim_directory,
        eval_directory=args.eval_directory,
        output_directory=output_directory,
        binning=(args.n_bins, 0.0, 1.0, args.binning),
        qcd_estimation=not args.no_qcd,
        n_parallel_read=args.parallel_read,
        n_parallel_write=args.parallel_write,
        cache_directory=args.cache_directory,
        skip_existing=args.skip_existing,
        uncertainty=args.uncertainty,
        signal_uncertainty=args.signal_uncertainty,
    )
    print("writing datacards with arguments")
    pprint.pprint(kwargs)
    print("\n")

    # write the datacards
    write_datacards(**kwargs)


# entry hook
if __name__ == "__main__":
    main()

    # old instructions for manual adjustments
    # for name, binning, variable_pattern, qcd_estimation in [
    #     # ("equal10_dnn", (10, 0.0, 1.0, "equal_distance"), "DNNoutSM_kl_1", False),
    #     # ("equal10_taunn", (10, 0.0, 1.0, "equal_distance"), "htt_class_hh", False),
    #     # ("flats10_dnn", (10, 0.0, 1.0, "flat_s"), "DNNoutSM_kl_1", False),
    #     # ("flats10_taunn", (10, 0.0, 1.0, "flat_s"), "htt_class_hh", False),
    #     # ("equal20_qcd_dnn", (20, 0.0, 1.0, "equal_distance"), "DNNoutSM_kl_1", True),
    #     # ("equal10_qcd_taunn", (10, 0.0, 1.0, "equal_distance"), "htt_class_hh", True),
    #     # ("flats10_qcd_dnn", (10, 0.0, 1.0, "flat_s"), "DNNoutSM_kl_1", True),
    #     # ("flats10_qcd_taunn", (10, 0.0, 1.0, "flat_s"), "htt_class_hh", True),
    #     # ("equal10_qcd_taunn_param", (10, 0.0, 1.0, "equal_distance"), "htt_para_s{spin}_m{mass}_class_hh", True),
    #     # ("equal20_qcd_taunn_param", (20, 0.0, 1.0, "equal_distance"), "htt_para_s{spin}_m{mass}_class_hh", True),
    #     # ("flats10_qcd_taunn_param", (10, 0.0, 1.0, "flat_s"), "htt_para_s{spin}_m{mass}_class_hh", True),
    #     # ("flats20_qcd_taunn_param", (20, 0.0, 1.0, "flat_s"), "htt_para_s{spin}_m{mass}_class_hh", True),
    #     ("flats10_qcd_taunn_param_v2", (10, 0.0, 1.0, "flat_s"), "htt_para_v2_s{spin}_m{mass}_class_hh", True),
    # ]:
    #     write_datacards(
    #         # all cards for spin 0
    #         spin=0,
    #         mass=masses,
    #         # mass=[1000],
    #         category=[
    #             "2017_*tau_resolved?b_os_iso",
    #             "2017_*tau_boosted_os_iso",
    #             "2017_*tau_vbf_os_iso",
    #             # "2017_*tau_resolved?bmwc_os_iso",
    #             # "2017_*tau_boostedmwc_os_iso",
    #         ],
    #         variable_pattern=variable_pattern,
    #         output_directory=f"/eos/user/m/mrieger/hhres_dnn_datacards/cards/{name}",
    #         binning=binning,
    #         qcd_estimation=qcd_estimation,

    #         # common settings
    #         n_parallel_read=6,
    #         n_parallel_write=8,
    #     )
