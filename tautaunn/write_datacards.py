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

import tautaunn.config as cfg

from tautaunn.util import transform_data_dir_cache
from tautaunn.config import masses, spins, klub_index_columns, luminosities, btag_wps, pnet_wps, klub_weight_columns
from tautaunn.nuisances import ShapeNuisance, RateNuisance, shape_nuisances, rate_nuisances
from tautaunn.cat_selectors import category_factory, sel_baseline 

from tautaunn.binning_algorithms import flats_systs, flatsguarded


# configurations

br_hh_bbtt = 0.073056256
channels = {
    "mutau": 0,
    "etau": 1,
    "tautau": 2,
}

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
    # reduce for debugging
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
    sample_data: dict[str, tuple[ak.Array, ak.Array]],
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
    assert binning_algo in ["equal_distance", "flat_s", "flatsguarded", "ud", "ud_flats", "tt_dy_driven"]

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
    dnn_data = {sample_name: data[1] for sample_name, data in sample_data.items()}
    sample_data = {sample_name: data[0] for sample_name, data in sample_data.items()}

    #for sample_name, data in sample_data.items():
        ##slice down sample_data to match the length of dnn_data
        #dnn_mask = np.isin(dnn_data[sample_name][klub_index_columns], data[klub_index_columns])
        #sample_data[sample_name] = data[dnn_mask] 

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

        qcd_data_shapes = {
            region_name: {
                sample_name: dnn_data[sample_name][categories[qcd_category]["selection"](dnn_data[sample_name])]
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

    dnn_data = {
        sample_name: dnn_data[sample_name][categories[category]["selection"](dnn_data[sample_name])]
        for sample_name, process_name in sample_processes.items()
        # skip data for now as were are using fake data from background-only below
        if not processes[process_name].get("data", False)
    }

    # complain when nan's were found
    for sample_name, data in sample_data.items():
        n_nonfinite_dnn = np.sum(~np.isfinite(dnn_data[sample_name][variable_name]))
        if n_nonfinite_dnn:
            print(
                f"{n_nonfinite_dnn} / {len(dnn_data[sample_name])} of events in {sample_name} after {category} "
                "selection are non-finite (nan or inf) in DNN data",
            )

    # prepare the scaling values, signal is scaled to 1pb * br
    scale = categories[category]["scale"]
    signal_scale = scale * br_hh_bbtt
    # retrieve the shapes model
    shapes_model = categories[category]["shapes_model"]

    # derive bin edges
    if binning_algo == "equal_distance":
        bin_edges = np.linspace(x_min, x_max, n_bins + 1).tolist()
    elif binning_algo == "flatsguarded":

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
        # helper to get values of weights of a process
        hh_values = ak.concatenate([
            dnn_data[sample_name][variable_name]
            for sample_name in sample_map[signal_process_name]
        ], axis=0)
        hh_weights = ak.concatenate([
            sample_data[sample_name].full_weight * signal_scale
            for sample_name in sample_map[signal_process_name]
        ], axis=0)
        #
        # step 1: data preparation
        #

        # get tt and dy data
        tt_values = ak.concatenate([
            dnn_data[sample_name][variable_name]
            for sample_name in sample_map["TT"]
        ], axis=0)
        tt_weights = ak.concatenate([
            sample_data[sample_name].full_weight
            for sample_name in sample_map["TT"]
        ], axis=0)
        dy_values = ak.concatenate([
            dnn_data[sample_name][variable_name]
            for sample_name in sample_map["DY"]
        ], axis=0)
        dy_weights = ak.concatenate([
            sample_data[sample_name].full_weight
            for sample_name in sample_map["DY"]
        ], axis=0)
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
            dnn_data[sample_name][variable_name]
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
                variable_name: dnn_data[sample_name][variable_name],
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
        for region_name, _qcd_data in qcd_data_shapes.items():
            # create a histogram that is filled with both data and negative background
            h = hist.Hist.new.Variable(bin_edges, name=variable_name).Weight()
            for sample_name, data in _qcd_data.items():
                weight = 1
                if not processes[sample_processes[sample_name]].get("data", False):
                    weight = -1 * qcd_data[region_name][sample_name].full_weight * _scale
                h.fill(**{variable_name: data[variable_name], "weight": weight})
            qcd_hists[region_name] = h
        h_qcd = estimate_qcd(qcd_hists)
        if not h_qcd is None:
            hists["QCD"] = h_qcd
                             
    shape_hists = {}
    for process_name, sample_names in sample_map.items():
        # skip data
        if processes[process_name].get("data", False):
            continue

        for shape_name, shape_data in shapes_model.items():
            if not fnmatch(process_name, shape_data["processes"]):
                continue
            for direction in ["up", "down"]:
                hist_name = f"{process_name}_{shape_name}_{direction}"
                h = hist.Hist.new.Variable(bin_edges, name=hist_name).Weight()
                for sample_name in sample_names:
                    weight = sample_data[sample_name].full_weight * _scale # nominal weight
                    fill_arr = dnn_data[sample_name][variable_name] # nominal dnn
                    if "klub_name" in shape_data:
                        weight = -1 * sample_data[sample_name][shape_data["klub_name"].format(direction=direction)] * _scale
                    if "dnn_shape_pattern" in shape_data:
                        fill_arr = dnn_data[sample_name][shape_data['dnn_shape_pattern'].format(variable_name=variable_name, direction=direction)]
                    h.fill(**{
                        hist_name: fill_arr, 
                        "weight": weight, 
                    })

                # add epsilon values at positions where bin contents are not positive
                nom = h.view().value
                mask = nom <= 0
                nom[mask] = 1.0e-5
                h.view().variance[mask] = 1.0e-5

                # store it with the corresponding pattern from line 1265
                shape_hists[hist_name] = {"hist":h,
                                          "parameter": shape_name,
                                          "process": process_name,
                                          "direction": direction}

                if qcd_estimation:
                    if fnmatch("QCD", shape_data["processes"]):
                        if not h_qcd is None:
                            qcd_hists_shape = {}
                            for region_name, _qcd_data in qcd_data_shapes.items():
                                # create a histogram that is filled with both data and negative background
                                hist_name = f"QCD_{shape_name}_{direction}"
                                h = hist.Hist.new.Variable(bin_edges, name=hist_name).Weight()
                                for sample_name, data in _qcd_data.items():
                                    weight = 1
                                    fill_arr = data[variable_name]
                                    if not processes[sample_processes[sample_name]].get("data", False):
                                        if "klub_name" in shape_data:
                                            weight = -1 * qcd_data[region_name][sample_name][shape_data["klub_name"].format(direction=direction)] * _scale
                                        if "dnn_shape_pattern" in shape_data:
                                            fill_arr = data[shape_data['dnn_shape_pattern'].format(variable_name=variable_name, direction=direction)]
                                    h.fill(**{hist_name: fill_arr, "weight": weight})
                                qcd_hists_shape[region_name] = h
                            h_qcd_shape = estimate_qcd(qcd_hists_shape)
                            if not h_qcd_shape is None:
                                shape_hists[hist_name] = {"hist":h_qcd_shape,
                                                            "parameter": shape_name,
                                                            "process": "QCD",
                                                            "direction": direction}
                

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
        for shape_name, shapes_hist_data in shape_hists.items():
            hist_name = shape_patterns["syst"].format(category=category,
                                                       process=shapes_hist_data["process"],
                                                       parameter=shapes_hist_data["parameter"],
                                                       direction=shapes_hist_data["direction"].capitalize())
            try:   
                root_file[hist_name] = shapes_hist_data["hist"] 
            except TypeError:
                from IPython import embed; embed()

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

    for shape_name, shape_data in shapes_model.items():
        effect_line = []
        for process_name in exp_processes:
            process_pattern = shape_data["processes"]
            if fnmatch(process_name, process_pattern):
                if process_name != "QCD":
                    effect = "1.0"
                else:
                    qcd_valid = all([h in shape_hists.keys() for h in [f"QCD_{shape_name}_up", f"QCD_{shape_name}_down"]])
                    if qcd_valid:
                        effect = "1.0"
            else:
                effect = "-"
            effect_line.append(effect)
        if set(effect_line) != {"-"}:
            blocks["tabular_parameters"].append((shape_name, "shape", *effect_line))
            added_param_names.append(shape_name)
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
        choices=("equal_distance", "flat_s", "flatsguarded", "ud_flats", "ud", "tt_dy_driven"),
        default="equal_distance",
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
