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


klub_weight_columns = [
    "MC_weight",
    "PUReweight",
    "L1pref_weight",
    "trigSF",
    "IdFakeSF_deep_2d",
    "PUjetID_SF",
    "bTagweightReshape",
]

dnn_shape_columns = [
    *[
        f"pdnn_m{mass}_s{spin}_{c}_ees_{dm}_{ud}"
        for mass in masses
        for spin in spins
        for c in ["hh", "tt"]
        for dm in ["DM0", "DM1"]
        for ud in ["up", "down"]],
    *[
        f"pdnn_m{mass}_s{spin}_{c}_jes_{i}_{ud}"
        for mass in masses
        for spin in spins
        for c in ["hh", "tt"]
        for i in range(1,12)
        for ud in ["up", "down"]
    ],
    *[
        f"pdnn_m{mass}_s{spin}_{c}_tes_{dm}_{ud}"
        for mass in masses
        for spin in spins
        for c in ["hh", "tt"]
        for dm in ["DM0", "DM1", "DM10", "DM11"]
        for ud in ["up", "down"]
    ],
    *[
        f"pdnn_m{mass}_s{spin}_{c}_mes_{ud}"
        for mass in masses
        for spin in spins
        for c in ["hh", "tt"]
        for ud in ["up", "down"]
    ]
]

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
    needs=["isLeptrigger", "isMETtrigger", "isSingleTauTrigger"],
    str_repr="((isLeptrigger == 1) | (isMETtrigger == 1) | (isSingleTauTrigger == 1))",
)
def sel_trigger(array: ak.Array) -> ak.Array:
    return (
        (array.isLeptrigger == 1) | (array.isMETtrigger == 1) | (array.isSingleTauTrigger == 1)
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

def merge_dicts(*dicts):
    merged = dicts[0].__class__()
    for d in dicts:
        merged.update(deepcopy(d))
    return merged

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


def load_klub_file(
    skim_directory: str,
    sample_name: str,
    file_name: str,
) -> tuple[ak.Array, float]:
    # prepare expressions
    expressions = klub_index_columns + klub_weight_columns + sel_baseline.flat_columns

    # add all columns potentially necessary for selections
    expressions += sum([
        cat["selection"].flat_columns
        for cat in categories.values()
    ], [])

    # make them unique
    expressions = list(set(expressions))

    # load the array
    f = uproot.open(os.path.join(skim_directory, sample_name, file_name))
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
    f = uproot.open(os.path.join(eval_directory, sample_name, file_name))
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
    f = uproot.open(os.path.join(eval_directory, sample_name, file_name))
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
        dnn_array = load_nominal_dnn(eval_directory, sample_name, file_name, dnn_output_columns)

        # use klub array index to filter dnn array
        dnn_mask = np.isin(dnn_array[klub_index_columns], klub_array[klub_index_columns])
        if ak.sum(dnn_mask) != len(klub_array):
            klub_path = os.path.join(skim_directory, sample_name, file_name)
            eval_path = os.path.join(eval_directory, sample_name, file_name)
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
                                           file_name.replace("_nominal", "_systs"),
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
        for field in dnn_shapes_array.fields:
            if field in klub_index_columns:
                continue
            array = ak.with_field(array, dnn_shapes_array[field], field)
    return array, sum_gen_mc_weights


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

def load_sample_data(
    skim_directory: str,
    eval_directory: str,
    sample_name: str,
    selection_columns: list[str] | None = None,
    dnn_output_columns: list[str] | None = None,
    cache_directory: str = "",
) -> ak.Array:
    print(f"loading sample {sample_name} ...")

    # determine file names and build arguments for the parallel load implementation
    load_args = [
        (skim_directory, eval_directory, sample_name, file_name, dnn_output_columns or [])
        for file_name in os.listdir(os.path.join(skim_directory, sample_name))
        if fnmatch(file_name, "output_*_nominal.root")
    ]

    ret = load_file(*load_args)

    # combine values
    array = ak.concatenate([arr for arr, _ in ret], axis=0)
    sum_gen_mc_weights = sum(f for _, f in ret)
    del ret
    gc.collect()

    # update the full weight
    array = ak.with_field(array, array.full_weight / sum_gen_mc_weights, "full_weight")

    # remove unnecessary columns
    keep_columns = dnn_output_columns + ["full_weight"] + (selection_columns or [])
    for c in array.fields:
        if c not in keep_columns:
            array = ak.without_field(array, c)
            gc.collect()

    print("done")

    return array


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
        dir_name[5:]
        for dir_name in os.listdir(skim_directory)
        if (
            os.path.isdir(os.path.join(skim_directory, dir_name)) and
            dir_name
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
    data_gen = (
        load_sample_data(
            skim_directory,
            eval_directory,
            sample_name,
            selection_columns,
            dnn_output_columns,
            n_parallel=n_parallel_read,
            cache_directory=cache_directory,
        )
        for sample_name in matched_sample_names
    )
    from IPython import embed; embed()

    
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
        default=(default_cats := "2017_*tau_resolved?b_os_iso,2017_*tau_boosted_os_iso,2017_*tau_vbf_os_iso"),
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
        help="uncertainty to use for uncertainty-driven binning (0.1 for 10%).; default: None",
    )
    parser.add_argument(
        "--signal-uncertainty",
        type=float,
        default=0.5,
        help="signal uncertainty to use for uncertainty-driven binning (0.1 for 10%).; default: 0.5",
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

if __name__ == "__main__":
    main()
