# coding: utf-8

"""
This script is used to determine the binning for the DNN output. The idea is to
just load the HH, TT and DY samples into memory (plus potentially all of their shifts)
and then derive a binning that asserts that none of the bins has an empty TT or DY contribution.
"""

import os
import gc
import pickle
import json
import hashlib
import itertools
import pprint

from typing import Any, Sequence
from functools import reduce
from operator import mul
from fnmatch import fnmatch

import uproot
import numpy as np
import awkward as ak
from tqdm import tqdm

from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool

from tautaunn.config import masses, spins, klub_index_columns, luminosities, btag_wps, pnet_wps
from tautaunn.util import transform_data_dir_cache
from tautaunn.cat_selectors import category_factory
from tautaunn.shape_nuisances import shape_nuisances
from tautaunn.write_datacards_stack import processes
from tautaunn.cat_selectors import selector, sel_baseline


skip_files = ["/gpfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_UL16/MuonH/output_45.root"]

klub_weight_columns = [
    "MC_weight",
    "PUReweight",
    "L1pref_weight",
    "trigSF",
    "idFakeSF",  # originally named "IdFakeSF_deep_2d" in KLUB for the central value
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

br_hh_bbtt = 0.073056256
channels = {
    "mutau": 0,
    "etau": 1,
    "tautau": 2,
}

categories = {}
for channel in channels:
    for name, sel in category_factory(channel=channel).items():
        # categories per year
        for year in ["2016", "2016APV", "2017", "2018"]:
            categories[f"{year}_{channel}_{name}"] = {
                "selection": sel,
                "n_bins": 10,
                "year": year,
                "scale": luminosities[year],
                **sel.extra,
            }

def load_klub_file(
    skim_directory: str,
    sample_name: str,
    file_name: str,
) -> tuple[ak.Array, float]:
    # all weight column patterns
    klub_weight_column_patterns = klub_weight_columns + [f"{c}*" for c in klub_weight_columns] + ["IdFakeSF_deep_2d"]

    # all columns that should be loaded and kept later on
    persistent_columns = klub_index_columns + klub_extra_columns + sel_baseline.flat_columns
    # add all columns potentially necessary for selections
    persistent_columns += sum([
        cat["selection"].flat_columns
        for cat in categories.values()
    ], [])

    # load the array
    f = uproot.open(os.path.join(skim_directory, sample_name, file_name))
    array = f["HTauTauTree"].arrays(
        filter_name=list(set(persistent_columns + (klub_weight_column_patterns))),
        cut=sel_baseline.str_repr.strip(),
    )

    # aliases do not work with filter_name for some reason, so swap names manually
    array = ak.with_field(array, array["IdFakeSF_deep_2d"], "idFakeSF")
    array = ak.without_field(array, "IdFakeSF_deep_2d")

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

    # also get the sum of generated weights
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
    # extended output columns for variations if not data
    expressions += [f"{c}*" for c in dnn_output_columns]
    expressions = list(set(expressions))

    # load the array
    f = uproot.open(os.path.join(eval_directory, sample_name, file_name))
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
    dnn_output_columns: list[str],
) -> tuple[ak.Array, float]:
    eval_file_name = klub_file_name.replace(".root", "_systs.root")
    if not os.path.exists(os.path.join(eval_directory, sample_name, eval_file_name)):
        raise Exception(f"evaluation file {eval_file_name} does not exist")
    # load the klub file
    klub_array, sum_gen_mc_weights = load_klub_file(skim_directory, sample_name, klub_file_name)

    # load the dnn output file
    if eval_directory:
        dnn_array = load_dnn_file(eval_directory, sample_name, eval_file_name, dnn_output_columns)

        # use klub array index to filter dnn array
        klub_mask = np.isin(klub_array[klub_index_columns], dnn_array[klub_index_columns])
        if ak.sum(klub_mask) != len(dnn_array):
            klub_path = os.path.join(skim_directory, sample_name, klub_file_name)
            eval_path = os.path.join(eval_directory, sample_name, eval_file_name)
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
            for file_name in os.listdir(os.path.join(skim_directory, sample_name))
            if fnmatch(file_name, "output_*.root") and not os.path.join(skim_directory, sample_name,file_name) in skip_files
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
        sum_gen_mc_weights = sum([f for _, f in ret])
        del ret
        gc.collect()

        # update the full weight
        for field in array.fields:
            if field.startswith("full_weight_"):
                array = ak.with_field(array, array[field] / sum_gen_mc_weights, field)

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

def get_binnings(
    spin: int | Sequence[int],
    mass: int | Sequence[int],
    category: str | Sequence[str],
    skim_directory: str,
    eval_directory: str,
    output_directory: str,
    variable_pattern: str = "dnn_spin{spin}_mass{mass}",
    sample_names: list[str] | None = None,
    binning: tuple[int, float, float, str] | tuple[float, float, str] = (0.0, 1.0, "flatsguarded"),
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

    # get a list of all sample names in the skim directory
    all_sample_names = [
        dir_name.replace("SKIM_", "")
        for dir_name in os.listdir(skim_directory)
        if (os.path.isdir(os.path.join(skim_directory, dir_name)))
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
        # also skip any bkg processes that are not needed for binning (everything except TT, DY)
        if not process_data.get("signal", False) and not process_name in ["TT", "DY"]:
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
    print(f"using cache directory: {cache_directory}")
    data_gen = (
        load_sample_data(
            skim_directory,
            eval_directory,
            sample_name,
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
            variable_pattern.format(spin=spin, mass=mass),
            binning,
        ))

    print(f"\nwriting datacard{'s' if len(datacard_args) > 1 else ''} ...")
    if n_parallel_write > 1:
        # run in parallel
        with ThreadPool(n_parallel_write) as pool:
            binning_results = list(tqdm(
                pool.imap(_get_binning_mp, datacard_args),
                total=len(datacard_args),
            ))
    else:
        binning_results = list(tqdm(
            map(_get_binning_mp, datacard_args),
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
    for args, edges in zip(datacard_args, binning_results):
        spin, mass, category = args[2:5]
        key = f"{category}__s{spin}__m{mass}"
        # do not overwrite when edges are None (in case the datacard was skipped)
        if key in all_bin_edges and not edges:
            continue
        all_bin_edges[key] = edges
    # write them
    with open(bin_edges_file, "w") as f:
        json.dump(all_bin_edges, f, indent=4)


def _get_binning(
    sample_map: dict[str, list[str]],
    sample_data: dict[str, tuple[ak.Array, ak.Array]],
    spin: int,
    mass: int,
    category: str,
    variable_name: str,
    binning: tuple[int, float, float, str] | tuple[float, float, str],
) -> tuple[str | None, str | None]:
    # input checks
    assert len(binning) in [3, 4]
    if len(binning) == 3:
        x_min, x_max, binning_algo = binning
        n_bins = categories[category]["n_bins"]
    else:
        n_bins, x_min, x_max, binning_algo = binning
    assert x_max > x_min
    assert binning_algo in ["flatsguarded",]


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


    # reversed map to assign processes to samples
    sample_processes = {}
    for process_name, sample_names in sample_map.items():
        sample_processes.update({sample_name: process_name for sample_name in sample_names})

    # apply the category selection to sample data
    sample_data = {
        sample_name: sample_data[sample_name][categories[category]["selection"](sample_data[sample_name], year=categories[category]["year"])]
        for sample_name, process_name in sample_processes.items()
        # skip data for now as were are using fake data from background-only below
        if not processes[process_name].get("data", False)
    }

    # complain when nan's were found
    for sample_name, data in sample_data.items():
        n_nonfinite = np.sum(~np.isfinite(data[variable_name]))
        if n_nonfinite:
            print(
                f"{n_nonfinite} / {len(data[variable_name])} of events in {sample_name} after {category} "
                "selection are non-finite (nan or inf) in DNN data",
            )

    # prepare the scaling values, signal is scaled to 1pb * br
    scale = categories[category]["scale"]
    signal_scale = scale * br_hh_bbtt


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

    #
    # step 1: data preparation
    #
    hh_values = ak.concatenate([
        sample_data[sample_name][variable_name]
        for sample_name in sample_map[signal_process_name]
    ], axis=0)
    hh_weights = ak.concatenate([
        sample_data[sample_name].full_weight_nominal * signal_scale
        for sample_name in sample_map[signal_process_name]
    ], axis=0)

    # get tt and dy data
    tt_values = ak.concatenate([
        sample_data[sample_name][variable_name]
        for sample_name in sample_map["TT"]
    ], axis=0)
    tt_weights = ak.concatenate([
        sample_data[sample_name].full_weight_nominal
        for sample_name in sample_map["TT"]
    ], axis=0)
    dy_values = ak.concatenate([
        sample_data[sample_name][variable_name]
        for sample_name in sample_map["DY"]
    ], axis=0)
    dy_weights = ak.concatenate([
        sample_data[sample_name].full_weight_nominal
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

    return bin_edges

def _get_binning_mp(args: tuple[Any]) -> tuple[str, str]:
    return _get_binning(*args)

def main():
    from argparse import ArgumentParser
    default_cats = ["{year}_*tau_resolved?b_os_iso" ,"{year}_*tau_boosted_os_iso"]
    default_skim_dir = "TN_SKIMS_{year}"

    csv = lambda s: [_s.strip() for _s in s.strip().split(",")]
    csv_int = lambda s: list(map(int, csv(s)))

    parser = ArgumentParser(
        description="write datacards for the hh->bbtt analysis",
    )
    parser.add_argument(
        "--year",
        "-y",
        type=str,
        default="2017",
        help="year to use; default: 2017",
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
        "--binning",
        "-b",
        choices=("flatsguarded",),
        default="flatsguarded",
        help="binning strategy to use; default: flatsguarded",
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
        default="$TN_SKIMS_{year}",
        help="dnn evaluation directory; default: $TN_SKIMS_{year}",
    )
    parser.add_argument(
        "--eval-directory",
        default="",
        help="dnn evaluation directory; default: empty",
    )
    parser.add_argument(
        "--variable",
        "-v",
        default="pdnn_m{mass}_s{spin}_hh",
        help="variable to use; can also contain '{spin}' and '{mass}'; default: pdnn_m{mass}_s{spin}_hh",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        default=(default_path :=  "/nfs/dust/cms/user/jwulff/taunn_data/store/GetBinning"),
        help=f"output directory; default: {default_path}",
    )
    parser.add_argument(
        "--output-label",
        "-l",
        default=(default_label := "{binning}{n_bins}"),
        help="output label (name of last directory); can also contain '{binning}', '{n_bins}'; "
        f"default: {default_label}",
    )
    parser.add_argument(
        "--cache-directory",
        default=(default_cache := "/nfs/dust/cms/user/jwulff/taunn_data/store/GetBinningCache"),
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
        ),
    )

    # prepare kwargs
    kwargs = dict(
        spin=args.spin,
        mass=args.mass,
        category=[c.format(year=args.year) for c in default_cats],
        skim_directory=os.environ[default_skim_dir.format(year=args.year)],
        eval_directory=args.eval_directory,
        output_directory=output_directory,
        variable_pattern=args.variable,
        binning=(args.n_bins, 0.0, 1.0, args.binning),
        n_parallel_read=args.parallel_read,
        n_parallel_write=args.parallel_write,
        cache_directory=args.cache_directory,
        skip_existing=args.skip_existing,
    )
    print("writing datacards with arguments")
    pprint.pprint(kwargs)
    print("\n")

    # write the datacards
    get_binnings(**kwargs)

if __name__ == "__main__":
    main()