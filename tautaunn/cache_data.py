# coding: utf-8
import time
import os
import gc
import itertools
import hashlib
import pickle
from functools import reduce
from operator import mul
from collections import defaultdict
from fnmatch import fnmatch
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from copy import deepcopy
from typing import Sequence, Any, Callable

from tqdm import tqdm
import numpy as np
import awkward as ak
import uproot


from pathlib import Path
from tautaunn.util import transform_data_dir_cache
from tautaunn.config import klub_index_columns, klub_weight_columns, processes
from tautaunn.nuisances import shape_nuisances
from tautaunn.cat_selectors import sel_baseline, category_factory
import tautaunn.config as cfg

radgrav = {0: "Rad", 2: "Grav"}

categories = {}
for channel in ["mutau", "etau", "tautau"]:
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
    persistent_columns = klub_index_columns + sel_baseline.flat_columns
    # add all columns potentially necessary for selections
    persistent_columns += sum([
        cat["selection"].flat_columns
        for cat in categories.values()
    ], [])

    # load the array
    with uproot.open(os.path.join(skim_directory, sample_name, file_name)) as f:
         
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
    with uproot.open(os.path.join(eval_directory, sample_name, file_name)) as f:
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
    # TODO: remove trailing path sep and resolve links to abs location
    h = [
        transform_data_dir_cache(skim_directory),  # .rstrip(os.sep)
        transform_data_dir_cache(eval_directory),  # .rstrip(os.sep)
        sel_baseline.str_repr.strip(),
        klub_columns,
        sorted(dnn_output_columns),
    ]
    h = hashlib.sha256(str(h).encode("utf-8")).hexdigest()[:10]

    return os.path.join(cache_directory, f"{year}_{sample_name}_{h}.pkl")


def load_sample_data(
    skim_directory: str,
    eval_directory: str,
    year: str,
    sample_name: str,
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
            for eval_file_name in os.listdir(os.path.join(eval_directory, sample_name))
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


def load_data(
    year: str, 
    sample_names: list[str],
    skim_directory: list[str],
    eval_directory: list[str],
    output_directory: str,
    cache_directory: str = "",
    variable_pattern: str = "pdnn_m{mass}_s{spin}_hh",
    n_parallel_read: int = 4,
    skip_existing: bool = False,
) -> list[tuple[str, str, list[float]]]:

    
    _spins = cfg.spins
    _masses = cfg.masses
    # prepare dnn output columns
    dnn_output_columns = [
        variable_pattern.format(spin=spin, mass=mass)
        for spin, mass in itertools.product(_spins, _masses)
    ]
    # get the hash
    cashe_path = get_cache_path(
        os.environ["TN_DATACARD_CACHE_DIR"],
        os.environ[f"TN_SKIMS_{year}"],
        os.path.join(eval_directory, year),
        year,
        "TT_SemiLep", 
        [variable_pattern.format(mass=mass, spin=spin)
            for mass in _masses for spin in _spins], 
    )
    h = Path(cashe_path).stem.split("_")[-1]

    # get a mapping of process name to sample names
    sample_map: dict[str, list] = {} 
    all_matched_sample_names: list[str] = [] 
    for process_name, process_data in processes.items():
        # skip signals that do not match any spin or mass
        if (
            process_data.get("signal", False) and
            (process_data["spin"] not in _spins or process_data["mass"] not in _masses)
        ):
            continue
        # match sample names
        matched_sample_names = []
        for sample_name in sample_names:
            if any(fnmatch(sample_name, pattern) for pattern in process_data["sample_patterns"]):
                if sample_name in matched_sample_names:
                    raise Exception(f"sample '{sample_name}' already matched by a previous process")
                all_matched_sample_names.append(sample_name)
                matched_sample_names.append(sample_name)
                continue
        if not matched_sample_names:
            print(f"process '{process_name}' has no matched samples, skipping")
            continue
        sample_map[process_name] = matched_sample_names

    # reverse the mapping to get a mapping of sample names to process names
    sample_processes = {str(val): key for key, values in sample_map.items() for val in values}

    # ensure that the output directory exists
    output_directory = os.path.expandvars(os.path.expanduser(output_directory))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    else:
        if skip_existing:
            existing_files = os.listdir(output_directory)
            existing_files = [f for f in existing_files if fnmatch(f, f"*_{h}.pkl")]
            # check for each hypothesis if the file already exists
            for hypothesis_name in [f"m{mass}_s{spin}" for spin, mass in itertools.product(_spins, _masses)]: 
                if any(fnmatch(f, f"{hypothesis_name}_{h}.pkl") for f in existing_files):
                    print(f"skipping hypothesis {hypothesis_name}, file already exists")
                    del hypothesis_cols[hypothesis_name]

    # loading data
    print(f"going to load {len(all_matched_sample_names)} samples")
    tic = time.time()
    sample_data = {
            sample_name: load_sample_data(
                skim_directory,
                eval_directory,
                year,
                sample_name,
                dnn_output_columns,
                n_parallel=n_parallel_read,
                cache_directory=cache_directory,
            )
            for i, sample_name in enumerate(all_matched_sample_names)
        }

    toc = time.time()
    print(f"loading took {toc - tic:.2f} s")
    # split this data into the the individual hypotheses
    all_cols = sample_data[all_matched_sample_names[0]].fields
    # get the columns that are not the dnn output columns
    non_dnn_cols = [c for c in all_cols if "dnn" not in c and "full_weight" not in c]
    # get the weight columns
    weight_cols = [c for c in all_cols if "full_weight" in c]
    # get the dnn columns
    hypothesis_cols = {f"m{mass}_s{spin}": [c for c in all_cols if f"pdnn_m{mass}_s{spin}" in c]
                       for spin, mass in itertools.product(_spins, _masses)}
    bkgd_and_data_samples = [s for s in all_matched_sample_names if 
                             processes[sample_processes[s]].get("data", False) or
                             not processes[sample_processes[s]].get("signal", False)]
    tic = time.time()
    for hypothesis_name, cols in tqdm(hypothesis_cols.items(), desc="Storing hypotheses"):
        tqdm.write(f"Storing {hypothesis_name}")
        spin, mass = hypothesis_name.split("_")
        spin, mass = spin[1:], mass[1:]
        hypothesis_data = {
            sample_name: sample_data[sample_name][non_dnn_cols + weight_cols + cols]
            for sample_name in bkgd_and_data_samples + [f"{radgrav[spin]}{mass}"]
        }
        with open(os.path.join(output_directory, f"{hypothesis_name}_{h}.pkl"), "wb") as f:
            pickle.dump(hypothesis_data, f)

    toc = time.time()
    print(f"storing took {toc - tic:.2f} s")
    return [(hypothesis_name, f"{hypothesis_name}_{h}.pkl", [mass, spin]) for hypothesis_name in hypothesis_cols.keys()]
    
    
