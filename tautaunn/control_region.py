# coding: utf-8

"""

plot the data / mc agreement in the control region

"""

import awkward as ak
import numpy as np
import pickle
import hist
import os
from fnmatch import fnmatch
import tqdm
from collections import defaultdict
import time
import itertools
import re
import uproot
import tempfile
import shutil
from tqdm import tqdm

from typing import Sequence, Any

from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool

from scinum import Number

# data loading same as for write_datacards.py
from tautaunn.write_datacards_stack import load_dnn_file, load_klub_file, load_file, load_file_mp
from tautaunn.write_datacards_stack import get_cache_path, sample_name_to_skim_dir, processes, load_sample_data
from tautaunn.write_datacards_stack import categories, spins, masses, datacard_years, luminosities, categories, shape_nuisances
from tautaunn.write_datacards_stack import expand_categories, make_list, dir_is_skim_dir, ShapeNuisance
from tautaunn.write_datacards_stack import br_hh_bbtt
import tautaunn.config as cfg


def write_datacards(
    spin: int | Sequence[int],
    mass: int | Sequence[int],
    category: str | Sequence[str],
    skim_directories: dict[tuple[str, str], list[str] | None],
    eval_directories: dict[str, str],
    output_directory: str,
    n_bins: int = 5,
    output_pattern: str = "cat_{category}_spin_{spin}_mass_{mass}",
    variable_pattern: str = "dnn_spin{spin}_mass{mass}",
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
            n_bins,
            output_directory,
            output_pattern.format(spin=spin, mass=mass, category=category),
            variable_pattern.format(spin=spin, mass=mass),
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

    return datacard_results


def _write_datacard(
    sample_map: dict[str, dict[str, list[str]]],
    sample_data: dict[str, dict[str, ak.Array]],
    spin: int,
    mass: int,
    category: str,
    n_bins: int,
    output_directory: str,
    output_name: str,
    variable_name: str,
    qcd_estimation: bool,
    skip_existing: bool,
) -> tuple[str | None, str | None, list[float] | None]:
    cat_data = categories[category]

    # check if there is data provided for this category if it is bound to a year
    assert cat_data["year"] in list(luminosities.keys()) + [None]
    if cat_data["year"] is not None and not any(cat_data["year"] == year for year in sample_data):
        print(f"category {category} is bound to a year but no data was provided for that year")
        return (None, None)

    # check if the requested category is indeed a control region
    if not "_cr" in category:
        raise ValueError(f"category {category} is not a control region")

    # prepare the output paths
    shapes_path = f"shapes_{output_name}.root"
    abs_shapes_path = os.path.join(output_directory, shapes_path)

    # mp-safe directory creation
    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
        except:
            time.sleep(0.5)
            if not os.path.exists(output_directory):
                raise

    if skip_existing and os.path.exists(abs_shapes_path):
        return shapes_path, None

    shape_patterns = {
        "nom": "cat_{category}/{process}",
        "nom_comb": "$CHANNEL/$PROCESS",
        "syst": "cat_{category}/{process}__{parameter}{direction}",
        "syst_comb": "$CHANNEL/$PROCESS__$SYSTEMATIC",
    }

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

    # reduce the sample_map in three steps:
    # - when the category is bound to a year, drop other years
    # - remove signal processes from the sample map that do not correspond to spin or mass
    # - remove data processes that are not meant to be included for the channel
    reduced_sample_map = defaultdict(dict)
    for year, _map in sample_map.items():
        if cat_data["year"] not in (None, year):
            continue
        for process_name, sample_names in _map.items():
            # skip all signals
            if (
                processes[process_name].get("signal", False)
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

    #
    # write shapes
    #

    # just 5 equal-width bins for now
    bin_edges = np.linspace(0, 1, n_bins + 1) 

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
                    h = hist.Hist.new.Variable(bin_edges, name=full_hist_name).Weight()
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
                    h.view().variance[mask] = 0.0

            # actual qcd estimation
            if qcd_estimation:
                # mapping year -> region -> hist
                qcd_hists: dict[str, dict[str, tuple[hist.Hist, hist.Hist]]] = defaultdict(dict)

                # create data-minus-background histograms in the 4 regions
                for region_name, _qcd_data in qcd_data.items():
                    for year, data in _qcd_data.items():
                        datacard_year = datacard_years[year]
                        # create a histogram that is filled with both data and negative background
                        full_hist_name = ShapeNuisance.create_full_name(hist_name, year=datacard_year)
                        h_data = hist.Hist.new.Variable(bin_edges, name=f"{full_hist_name}_data").Weight()
                        h_mc = hist.Hist.new.Variable(bin_edges, name=f"{full_hist_name}_mc").Weight()
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
                                h_data.fill(**{
                                    f"{full_hist_name}_data": _data[variable_name],
                                    "weight": 1,
                                })
                            else:
                                scale = luminosities[year]
                                h_mc.fill(**{
                                    f"{full_hist_name}_mc": _data[varied_variable_name],
                                    "weight": _data[varied_weight_field] * scale,
                                })
                        # subtract the mc from the data
                        # h_qcd = hist.Hist.new.Variable(bin_edges, name=f"{full_hist_name}").Weight()
                        # h_qcd.view().value[...] = h_data.view().value - h_mc.view().value
                        # h_qcd.view().variance[...] = h_mc.view().variance
                        # qcd_hists[year][region_name] = h_qcd
                        qcd_hists[year][region_name] = (h_data, h_mc)

                # ABCD method per year
                # TODO: consider using averaging between the two options where the shape is coming from
                for year, region_hists in qcd_hists.items():
                    datacard_year = datacard_years[year]
                    full_hist_name = ShapeNuisance.create_full_name(hist_name, year=datacard_year)
                    h_qcd = hist.Hist.new.Variable(bin_edges, name=f"{full_hist_name}").Weight()
                    # shape placeholders
                    B, C, D = "ss_iso", "os_noniso", "ss_noniso"
                    # test
                    # B, C, D = "os_noniso", "ss_iso", "ss_noniso"
                    h_data_b, h_mc_b = region_hists[B]
                    h_data_c, h_mc_c = region_hists[C]
                    h_data_d, h_mc_d = region_hists[D]
                    # compute transfer factor and separate mc and data uncertainties
                    int_data_c = Number(h_data_c.sum().value, {"data": h_data_c.sum().variance**0.5})
                    int_data_d = Number(h_data_d.sum().value, {"data": h_data_d.sum().variance**0.5})
                    int_mc_c = Number(h_mc_c.sum().value, {"mc": h_mc_c.sum().variance**0.5})
                    int_mc_d = Number(h_mc_d.sum().value, {"mc": h_mc_d.sum().variance**0.5})
                    # deem the qcd estimation invalid if either difference is negative
                    qcd_invalid = (int_mc_c > int_data_c) or (int_mc_d > int_data_d)
                    if not qcd_invalid:
                        # compute the QCD shape with error propagation
                        values_data_b = Number(h_data_b.view().value, {"data": h_data_b.view().variance**0.5})
                        values_mc_b = Number(h_mc_b.view().value, {"mc": h_mc_b.view().variance**0.5})
                        tf = (int_data_c - int_mc_c) / (int_data_d - int_mc_d)
                        qcd = (values_data_b - values_mc_b) * tf
                        # inject values
                        h_qcd.view().value[...] = qcd.n
                        # inject variances, combining data and mc uncertainties, assuming symmetric errors
                        h_qcd.view().variance[...] = qcd.get("up", unc=True)**2
                    # zero-fill
                    hval = h_qcd.view().value
                    hvar = h_qcd.view().variance
                    zero_mask = hval <= 0
                    # keep the variance proportion that reaches into positive values
                    hvar[zero_mask] = (np.maximum(0, hvar[zero_mask]**0.5 + hval[zero_mask]))**2
                    hval[zero_mask] = 1.0e-5
                    # store it
                    hists[(year, "QCD")][(nuisance.get_combine_name(year=datacard_year), direction)] = h_qcd

    # drop qcd shapes in years where no valid estimation was found
    if qcd_estimation:
        for year, qcd_valid in any_qcd_valid.items():
            if not qcd_valid:
                print(
                    f"  completely dropping QCD shape in ({category},{year},{spin},{mass}) as no valid shape could be "
                    "created for any nuisance",
                )
                del hists[(year, "QCD")]

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
        os.chmod(abs_shapes_path, 0o0664)

    # return output paths
    return abs_shapes_path


def _write_datacard_mp(args: tuple[Any]) -> tuple[str, str]:
    return _write_datacard(*args)


default_categories = ["2017_*tau_resolved1b_noak8_cr_os_iso", "2017_*tau_resolved2b_first_cr_os_iso"]

def main(output_dir: str,
         spins: list[int] = cfg.spins,
         masses: list[int] = cfg.masses,
         categories: list[str] = default_categories,
         ):

    eval_dir = ("/nfs/dust/cms/user/riegerma/taunn_data/store/EvaluateSkims/"
                "hbtres_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_"
                "ED10_LU8x128_CTdense_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_RSv6_"
                "fi80_lbn_ft_lt20_lr1_LBdefault_daurot_fatjet_composite_FIx5_SDx5/prod7")
    if "max-" in os.environ["HOSTNAME"]:
        eval_dir = eval_dir.replace("nfs", "data") 

    def split_skim_name(skim_name: str) -> tuple[str, str, str, cfg.Sample]:
        m = re.match(rf"^({'|'.join(cfg.skim_dirs.keys())})_(.+)$", skim_name)
        if not m:
            raise ValueError(f"invalid skim name format '{skim_name}'")
        skim_year = m.group(1)
        sample_name = m.group(2)
        return sample_name, skim_year

    # get the year from the categories
    years = set([cat.split("_")[0] for cat in categories])
    if len(years) != 1:
        raise ValueError("categories must all be from the same year")
    year = years.pop()
    
    skim_directories = defaultdict(list)
    eval_directories = {}
    for skim_name in os.listdir(f"{eval_dir}/{year}"): 
        sample = cfg.get_sample(f"{year}_{skim_name}", silent=True)
        if sample is None:
            sample_name, skim_year = split_skim_name(f"{year}_{skim_name}")
            sample = cfg.Sample(sample_name, year=skim_year)
        skim_directories[(sample.year, cfg.skim_dirs[sample.year])].append(sample.name)
        if sample.year not in eval_directories:
            #eval_directories[sample.year] = inp[skim_name].collection.dir.parent.path
            eval_directories[sample.year] = os.path.join(eval_dir, sample.year)

    datacard_kwargs = dict(
        spin=list(cfg.spins),
        mass=list(cfg.masses),
        category=default_categories,
        skim_directories=skim_directories,
        eval_directories=eval_directories,
        output_directory=output_dir,
        n_bins=5,
        variable_pattern="pdnn_m{mass}_s{spin}_hh",
        qcd_estimation=True,
        n_parallel_read=10,
        n_parallel_write=10,
        cache_directory=os.environ["TN_DATACARD_CACHE_DIR"],
        skip_existing=True,
    )

    # create the cards
    write_datacards(**datacard_kwargs)


if __name__ == "__main__":

    import argparse

    def make_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("output_dir", type=str)
        parser.add_argument("--categories", type=str, nargs="+", default=default_categories)
        return parser

    parser = make_parser()
    args = parser.parse_args() 
    main(args.output_dir,
         categories=args.categories,
         )