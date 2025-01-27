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
import re
import tempfile
import shutil
import hashlib
from collections import OrderedDict, defaultdict
import itertools
from fnmatch import fnmatch
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from copy import deepcopy
from typing import Sequence, Any, Callable

from tqdm import tqdm
import time
import numpy as np
import awkward as ak
import uproot
import hist
from scinum import Number
from pathlib import Path
import matplotlib.pyplot as plt
import mplhep as hep
from hist import Hist, Stack

import tautaunn.config as cfg

from tautaunn.util import transform_data_dir_cache
from tautaunn.config import channels, processes, masses, spins, luminosities, datacard_years, br_hh_bbtt
from tautaunn.nuisances import ShapeNuisance, RateNuisance, shape_nuisances, rate_nuisances
from tautaunn.cat_selectors import category_factory, sel_baseline 
from tautaunn.binning_algorithms import flats, flats_systs, flatsguarded
hep.style.use(hep.style.CMS)


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




def make_list(x):
    return list(x) if isinstance(x, (list, tuple, set)) else [x]


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


def plot_mc_data_sig(data_hist: Hist,
                     bkgd_stack: Stack,
                     year: str,
                     channel: str,
                     cat: str,
                     savename: str | Path | None = None
                     ) -> None:

    color_map = {
        "DY": "#7a21dd",
        "TT": "#9c9ca1",
        "ST": "#e42536",
        "W": "#964a8b",
        "QCD": "#f89c20",
        "Others":"#5790fc",
    }
    lumi = {"2016APV": "19.5", "2016": "16.8", "2017": "41.5", "2018": "59.7"}[year]

    fig, ax1 = plt.subplots(1, 1,
                            figsize=(10, 12))
    hep.cms.text(" Preliminary", fontsize=20, ax=ax1)
    mu, tau = '\u03BC','\u03C4'
    chn_map = {"etau": r"$bbe$"+tau, "tautau":r"$bb$"+tau+tau, "mutau": r"$bb$"+mu+tau}
    hep.cms.lumitext(r"{} $fb^{{-1}}$ (13 TeV)".format(lumi), fontsize=20, ax = ax1)
    ax1.text(0.05, .91, f"{chn_map[channel]}\n{cat}", fontsize=15,transform=ax1.transAxes)
    
    bkgd_stack.plot(stack=True, ax=ax1, color=[color_map[i.name] for i in bkgd_stack], histtype='fill')
    data_hist.plot(ax=ax1, color='black', label="Data", histtype='errorbar')
        
    lgd = ax1.legend( fontsize = 12,bbox_to_anchor = (0.99, 0.99), loc="upper right", ncols=2,
                    frameon=True, facecolor='white', edgecolor='black')
    lgd.get_frame().set_boxstyle("Square", pad=0.0)
    ax1.set_yscale("log")
    max_y = sum(bkgd_stack).values().max()
    ax1.set_xlabel("")
    ax1.set_ylabel("Events")
    if not savename is None:
        if not Path(savename).parent.exists():
            os.makedirs(Path(savename).parent)
        plt.savefig(savename, bbox_inches='tight', pad_inches=0.05)
        plt.close()


def write_datacard(
    sample_data: dict[str, ak.Array],
    spin: int,
    mass: int,
    category: str,
    binning: tuple[int, float, float, str] | tuple[float, float, str] | list[float],
    output_directory: str,
    output_name: str,
    variable_name: str,
    qcd_estimation: bool,
    skip_existing: bool,
) -> tuple[str | None, str | None, list[float] | None]:
    cat_data = categories[category]

    if isinstance(binning, list):
        binning_algo = "custom"
        print(f"using custom binning for category {category}")
    else:
        # input checks
        assert len(binning) in [3, 4]
        if len(binning) == 3:
            x_min, x_max, binning_algo = binning
            n_bins = cat_data["n_bins"]
        else:
            n_bins, x_min, x_max, binning_algo = binning
        assert x_max > x_min
        assert binning_algo in {"equal", "flats", "flatsguarded", "flats_systs", "non_res_like"}

    # check if there is data provided for this category if it is bound to a year
    assert cat_data["year"] in list(luminosities.keys()) + [None]

    year = cat_data["year"]
    datacard_year = datacard_years[year]

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

    shapes_path = f"shapes_{output_name}.root"
    abs_shapes_path = os.path.join(output_directory, shapes_path)


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

    # reversed map to assign processes to samples
    sample_processes = {str(val): key for key, values in sample_map.items() for val in values}

    # reduce the sample_map in three steps:
    # - when the category is bound to a year, drop other years
    # - remove signal processes from the sample map that do not correspond to spin or mass
    # - remove data processes that are not meant to be included for the channel
    reduced_sample_map = defaultdict(dict)
    for process_name, sample_names in sample_map.items():
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
        reduced_sample_map[process_name] = sample_names

    sample_map = reduced_sample_map

    # apply qcd estimation category selections
    if qcd_estimation:
        qcd_data = {
            region_name: {
                    sample_name: sample_data[sample_name][categories[qcd_category]["selection"](sample_data[sample_name], year=year)]
                    for sample_name, process_name in sample_processes.items()
                    # skip signal
                    if not processes[process_name].get("signal", False)
            }
            for region_name, qcd_category in qcd_categories.items()
        }

    # apply the category selection to sample data
    sample_data = {
            sample_name: sample_data[sample_name][cat_data["selection"](sample_data[sample_name], year=year)]
            for sample_name, process_name in sample_processes.items()
    }

    # complain when nan's were found
    for sample_name, _data in sample_data.items():
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

    # write shapes
    #

    if binning_algo == "equal":
        bin_edges = np.linspace(x_min, x_max, n_bins + 1).tolist()
    elif binning_algo in ('flats', 'flatsguarded', 'flats_systs', 'non_res_like',):
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
        def get_values_and_weights(
            process_name: str | dict[str, str],
            nuisance: ShapeNuisance = shape_nuisances["nominal"],
            direction: str = "",
            weight_scale: float | int = 1.0,
        ):
            if isinstance(process_name, str):
                process_name = {year: process_name for year in sample_data}

            def extract(getter):
                return ak.concatenate(
                    list(itertools.chain.from_iterable(
                            [getter(year, data, sample_name) for sample_name in sample_map[year][process_name[year]]]
                            for year, data in sample_data.items()
                        ))
                )

            values = extract(lambda year, data, sample_name: data[sample_name][nuisance.get_varied_discriminator(variable_name, direction)])  # noqa
            weights = extract(lambda year, data, sample_name: data[sample_name][nuisance.get_varied_full_weight(direction)] * luminosities[year] * weight_scale)  # noqa

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

        hh_values, hh_weights = get_values_and_weights(process_name=signal_process_name, weight_scale=br_hh_bbtt)

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
            tt_values, tt_weights = get_values_and_weights("TT")
            dy_values, dy_weights = get_values_and_weights("DY")
            all_bkgds = {}
            for proc in processes:
                if ((processes[proc].get("data", False)) or (processes[proc].get("signal", False)) or (proc == "QCD")):
                    continue
                elif proc in ["TT", "DY"]:
                    continue
                all_bkgds[proc] = get_values_and_weights(proc)
            all_bkgds_values = np.concatenate([v[0] for v in all_bkgds.values()])
            all_bkgds_weights = np.concatenate([v[1] for v in all_bkgds.values()])
            if len(hh_values) == 0:
                print(f"no signal events found in ({category},{spin},{mass})")
                bin_edges, stop_reason, bin_counts = [0., 1.], "no signal events found", None
            else:
                bin_edges, stop_reason = flats(
                    hh=(hh_values, hh_weights),
                    tt=(tt_values, tt_weights),
                    dy=(dy_values, dy_weights),
                    all_bkgds=(all_bkgds_values, all_bkgds_weights),
                    n_bins=n_bins,
                    x_min=x_min,
                    x_max=x_max,
                )
        elif binning_algo == "flatsguarded":  # flatsguarded
            #
            # step 1: data preparation
            #

            # get tt and dy data
            tt_values, tt_weights = get_values_and_weights("TT")
            dy_values, dy_weights = get_values_and_weights("DY")
            bin_edges, stop_reason = flatsguarded(
                hh_values=hh_values,
                hh_weights=hh_weights,
                tt_values=tt_values,
                tt_weights=tt_weights,
                dy_values=dy_values,
                dy_weights=dy_weights,
                n_bins=n_bins,
                x_min=x_min,
                x_max=x_max,
            )
        elif binning_algo == "flats_systs":
            hh_shifts = OrderedDict()
            tt_shifts = OrderedDict()
            dy_shifts = OrderedDict()
            all_bkgds = {}
            for nuisance in shape_nuisances.values():
                for direction in nuisance.get_directions():
                    key = f"{nuisance.name}_{direction}" if not nuisance.is_nominal else "nominal"
                    hh_values, hh_weights = get_values_and_weights(signal_process_name, nuisance, direction, br_hh_bbtt)
                    tt_values, tt_weights = get_values_and_weights("TT", nuisance, direction)
                    dy_values, dy_weights = get_values_and_weights("DY", nuisance, direction)
                    if key == "nominal":
                        all_bkgds["TT"] = tt_values, tt_weights
                        all_bkgds["DY"] = dy_values, dy_weights
                    tt_shifts[key] = (tt_values, tt_weights)
                    dy_shifts[key] = (dy_values, dy_weights)
                    hh_shifts[key] = (hh_values, hh_weights)
            # add all other bkgd processes to all_bkgds (just nominal)
            for proc in processes:
                if ((processes[proc].get("data", False)) or (processes[proc].get("signal", False)) or (proc == "QCD")):
                    continue
                elif proc in ["TT", "DY"]:
                    continue
                all_bkgds[proc] = get_values_and_weights(proc)
            all_bkgds_values = np.concatenate([v[0] for v in all_bkgds.values()])
            all_bkgds_weights = np.concatenate([v[1] for v in all_bkgds.values()])

            if len(hh_values) == 0:
                print(f"no signal events found in ({category},{spin},{mass})")
                bin_edges, stop_reason, bin_counts = [0., 1.], "no signal events found", None
            else:
                bin_edges, stop_reason, bin_counts = flats_systs(
                    hh_shifts=hh_shifts,
                    tt_shifts=tt_shifts,
                    dy_shifts=dy_shifts,
                    all_bkgds=(all_bkgds_values, all_bkgds_weights),
                    error_target=1,
                    n_bins=n_bins,
                    x_min=x_min,
                    x_max=x_max,
                )
    elif binning_algo == "custom":
        bin_edges = binning


    # histogram structures
    # mapping process -> (nuisance, direction) -> hist
    hists: dict[str, dict[tuple[str, str], hist.Hist]] = defaultdict(dict)

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
            for process_name, _map in sample_map.items():
                if not nuisance.applies_to_process(process_name):
                    continue

                _hist_name, _process_name = hist_name, process_name
                if processes[process_name].get("data", False):
                    _hist_name = _process_name = "data_obs"
                full_hist_name = ShapeNuisance.create_full_name(_hist_name, year=datacard_year)
                h = hist.Hist.new.Variable(bin_edges, name=full_hist_name).Weight()
                hists[_process_name][(nuisance.get_combine_name(year=datacard_year), direction)] = h

            # fill histograms
            for process_name, sample_names in sample_map.items():
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
                full_hist_name = ShapeNuisance.create_full_name(_hist_name, year=datacard_year)
                h = hists[_process_name][(nuisance.get_combine_name(year=datacard_year), direction)]
                scale = 1 if is_data else luminosities[year]
                if processes[process_name].get("signal", False):
                    scale *= br_hh_bbtt
                for sample_name in sample_names:
                    weight = 1
                    if not is_data:
                        weight = sample_data[sample_name][varied_weight_field] * scale
                    h.fill(**{
                        full_hist_name: sample_data[sample_name][varied_variable_name],
                        "weight": weight,
                    })

                    # add epsilon values at positions where bin contents are not positive
                    nom = h.view().value
                    mask = nom <= 0
                    nom[mask] = 1.0e-5
                    h.view().variance[mask] = 0.0

            # actual qcd estimation
            if qcd_estimation:
                # mapping region -> hist
                qcd_hists: dict[str, tuple[hist.Hist, hist.Hist]] = defaultdict(dict)
                # create data-minus-background histograms in the 4 regions
                for region_name, _qcd_data in qcd_data.items():
                    datacard_year = datacard_years[year]
                    # create a histogram that is filled with both data and negative background
                    full_hist_name = ShapeNuisance.create_full_name(hist_name, year=datacard_year)
                    h_data = hist.Hist.new.Variable(bin_edges, name=f"{full_hist_name}_data").Weight()
                    # h_mc = hist.Hist.new.Variable(bin_edges, name=f"{full_hist_name}_mc").Weight()
                    mc_hists = {"TT": hist.Hist.new.Variable(bin_edges, name=f"{full_hist_name}").Weight(),
                                "ST": hist.Hist.new.Variable(bin_edges, name=f"{full_hist_name}").Weight(),
                                "DY": hist.Hist.new.Variable(bin_edges, name=f"{full_hist_name}").Weight(),
                                "W": hist.Hist.new.Variable(bin_edges, name=f"{full_hist_name}").Weight(),
                                "Others": hist.Hist.new.Variable(bin_edges, name=f"{full_hist_name}").Weight()}
                    for sample_name, _data in _qcd_data.items():
                        process_name = sample_processes[sample_name]
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
                        elif process_name in [key for key in mc_hists.keys() if key != "Others"]: 
                            scale = luminosities[year]
                            mc_hists[process_name].fill(**{
                                f"{full_hist_name}": _data[varied_variable_name],
                                "weight": _data[varied_weight_field] * scale,
                            })
                        else:
                            scale = luminosities[year]
                            mc_hists["Others"].fill(**{
                                f"{full_hist_name}": _data[varied_variable_name],
                                "weight": _data[varied_weight_field] * scale,
                            })

                    mc_stack = hist.Stack.from_dict(mc_hists)
                    # subtract the mc from the data
                    # h_qcd = hist.Hist.new.Variable(bin_edges, name=f"{full_hist_name}").Weight()
                    # h_qcd.view().value[...] = h_data.view().value - h_mc.view().value
                    # h_qcd.view().variance[...] = h_mc.view().variance
                    # qcd_hists[year][region_name] = h_qcd
                    qcd_hists[region_name] = (h_data, mc_stack)
                # ABCD method per year
                # TODO: consider using averaging between the two options where the shape is coming from
                datacard_year = datacard_years[year]
                full_hist_name = ShapeNuisance.create_full_name(hist_name, year=datacard_year)
                h_qcd = hist.Hist.new.Variable(bin_edges, name=f"{full_hist_name}").Weight()
                # shape placeholders
                B, C, D = "ss_iso", "os_noniso", "ss_noniso"
                # test
                # B, C, D = "os_noniso", "ss_iso", "ss_noniso"
                h_data_b, h_mc_b = qcd_hists[B]
                h_data_c, h_mc_c = qcd_hists[C]
                h_data_d, h_mc_d = qcd_hists[D]
                # compute transfer factor and separate mc and data uncertainties
                int_data_c = Number(h_data_c.sum().value, {"data": h_data_c.sum().variance**0.5})
                int_data_d = Number(h_data_d.sum().value, {"data": h_data_d.sum().variance**0.5})
                int_mc_c = Number(sum(h_mc_c).sum().value, {"mc": sum(h_mc_c).sum().variance**0.5})
                int_mc_d = Number(sum(h_mc_d).sum().value, {"mc": sum(h_mc_d).sum().variance**0.5})
                # deem the qcd estimation invalid if either difference is negative
                qcd_invalid = (int_mc_c > int_data_c) or (int_mc_d > int_data_d) or (int_data_d == int_mc_d)
                if not qcd_invalid:
                    # compute the QCD shape with error propagation
                    values_data_b = Number(h_data_b.view().value, {"data": h_data_b.view().variance**0.5})
                    values_mc_b = Number(sum(h_mc_b).view().value, {"mc": sum(h_mc_b).view().variance**0.5})
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
                hists["QCD"][(nuisance.get_combine_name(year=datacard_year), direction)] = h_qcd
                if nuisance.is_nominal and qcd_invalid:
                    print(f"QCD estimation invalid for {year} in category {category}")
                    # plot the data and mc in the qcd regions
                    plot_mc_data_sig(h_data_b,
                                     h_mc_b,
                                     year,
                                     cat_data["channel"],
                                     category,
                                     f"./qcd_plots/SS_iso_{year}_{category}_s{spin}_m{mass}.png")
                    plot_mc_data_sig(h_data_c,
                                        h_mc_c,
                                        year,
                                        cat_data["channel"],
                                        category,
                                        f"./qcd_plots/OS_noniso_{year}_{category}_s{spin}_m{mass}.png")
                    plot_mc_data_sig(h_data_d,
                                        h_mc_d,
                                        year,
                                        cat_data["channel"],
                                        category,
                                        f"./qcd_plots/SS_noniso_{year}_{category}_s{spin}_m{mass}.png")


    # gather rates from nominal histograms
    rates = {
        process_name: _hists[("nominal", "")].sum().value
        for process_name, _hists in hists.items()
    }

    # create process names joining raw names and years
    full_process_names = {
        process_name: (
            "{1}_{0}{2}".format(year, *m.groups())
            if (m := re.match(r"^(.+)(_h[^_]+)$", process_name))
            else f"{process_name}_{year}"
        )
        for process_name in hists
        if process_name != "data_obs"
    }

    # save nominal shapes
    # note: since /eos does not like write streams, first write to a tmp file and then copy
    def write(path):
        # create a dictionary of all histograms
        content = {}
        for  process_name, _hists in hists.items():
            for (full_nuisance_name, direction), h in _hists.items():
                # determine the full process name and optionally skip data for nuisances
                if process_name == "data_obs":
                    if full_nuisance_name != "nominal":
                        continue
                    full_process_name = process_name
                else:
                    full_process_name = full_process_names[process_name]

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
    rate_category = category.split("_", 2)[2]
    for rate_nuisance in rate_nuisances.values():
        # determine the effects per expected process
        effect_line = []
        for year, process_name, _ in exp_processes:
            effect = "-"
            # check of the nuisance has any rate effect that applies here
            # if so, add it and stop, otherwise skip the nuisance alltogether
            for rate_effect in rate_nuisance.rate_effects:
                # if the nuisances does not apply to either the channel or the category, skip it
                if (
                    rate_effect.applies_to_channel(cat_data["channel"]) and
                    rate_effect.applies_to_category(rate_category) and
                    rate_effect.applies_to_year(year) and
                    rate_effect.applies_to_process(process_name)
                ):
                    assert effect == "-"
                    effect = rate_effect.effect
            effect_line.append(effect)
        if set(effect_line) != {"-"}:
            blocks["tabular_parameters"].append((rate_nuisance.name, "lnN", *effect_line))
            added_rate_params.append(rate_nuisance.name)

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
    blocks["line_parameters"] = []
    # blocks["line_parameters"] = [
    #     ("rate_nuisances", "group", "=", " ".join(added_rate_params)),
    # ]
    # if added_shape_params:
    #     blocks["line_parameters"].append(("shape_nuisances", "group", "=", " ".join(added_shape_params)))
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
    os.chmod(abs_datacard_path, 0o664)

    # return output paths
    return abs_datacard_path, abs_shapes_path, bin_edges, stop_reason


def write_datacards(
    sample_data: dict[str, ak.Array],
    spin: int | Sequence[int],
    mass: int | Sequence[int],
    category: str | Sequence[str],
    output_directory: str,
    variable_pattern: str = "pdnn_s{spin}_m{mass}_hh",
    output_pattern: str = "cat_{category}_spin_{spin}_mass_{mass}",
    binning: tuple[int, float, float, str] | tuple[float, float, str] = (0.0, 1.0, "flats"),
    binning_file: str = "",
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

    # input checks
    for spin in _spins:
        assert spin in spins
    for mass in _masses:
        assert mass in masses
    for category in _categories:
        assert category in categories

    if binning_file != "":
        import json
        print(f"reading binning from file {binning_file}")
        with open(binning_file, "r") as f:
            binnings = json.load(f)
        cats_in_file = list(set([key.split("__")[0] for key in binnings.keys()]))
        print(f"found binning for categories: {cats_in_file}")
        print(f"requested categories {_categories}")
        assert set(_categories) == set(cats_in_file), "categories in binning file do not match the requested categories"


    # ensure that the output directory exists
    output_directory = os.path.expandvars(os.path.expanduser(output_directory))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # write each spin, mass and category combination
    datacard_args = []
    if binning_file == "":
        for category in categories: 
            datacard_args.append((
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
    else:
        for key in binnings:
            bin_edges = sorted(list(set(binnings[key][0]))) if len(binnings[key]) == 2 else sorted(list(set(binnings[key])))
            spin, mass = (int(i[1:]) for i in key.split("__")[1:])
            #spin, mass = re.search(r"s(\d+)_m(\d+)", key).groups()
            category = key.split("__")[0]
            datacard_args.append((
                sample_data,
                int(spin),
                int(mass),
                category,
                output_directory,
                output_pattern.format(spin=spin, mass=mass, category=category),
                variable_pattern.format(spin=spin, mass=mass),
                bin_edges,
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

    if binning_file == "":
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
            stop_reason = res[-2]
            #bin_counts = res[-1]
            key = f"{category}__s{spin}__m{mass}"
            # do not overwrite when edges are None (in case the datacard was skipped)
            if key in all_bin_edges and not edges:
                continue
            all_bin_edges[key] = edges, stop_reason#, bin_counts
        # write them
        with open(bin_edges_file, "w") as f:
            json.dump(all_bin_edges, f, indent=4)
    return datacard_results


def _write_datacard_mp(args: tuple[Any]) -> tuple[str, str]:
    return write_datacard(*args)