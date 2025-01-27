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
from collections import defaultdict, OrderedDict
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
from tautaunn.write_datacards_stack import categories, spins, masses, datacard_years, luminosities, shape_nuisances
from tautaunn.write_datacards_stack import expand_categories, make_list, load_sample_data, ShapeNuisance
from tautaunn.write_datacards_stack import br_hh_bbtt
from tautaunn.config import processes

from pathlib import Path
import matplotlib.pyplot as plt
from functools import reduce
from operator import add
from hist import Hist, Stack
import mplhep as hep
hep.style.use("CMS")


def loose_year(str) -> str:
    return re.sub(r"_2016APV|_2016|_2017|_2018", "", str)


def reduce_stack(stack: Stack,) -> Stack:
    main_bkgds = ['TT', 'ST', 'DY', 'W', 'QCD']
    bkgd_dict  = {loose_year(h.name): h for h in stack if any(loose_year(h.name) == s for s in main_bkgds)}
    others = reduce(add, (h for h in stack if loose_year(h.name) not in bkgd_dict))
    bkgd_dict["Others"] = others
    bkgd_dict = {k: v[0] for k, v in bkgd_dict.items()}
    # impose a fixed order of "Others", "QCD", "W", "ST", "DY", "TT"
    sorted_bkgd_dict = OrderedDict()
    for name in ["Others", "QCD", "W", "ST", "DY", "TT"]:
        if name in bkgd_dict:
            sorted_bkgd_dict[name] = bkgd_dict[name]
    return Stack.from_dict(sorted_bkgd_dict)


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



def write_datacards(
    spin: int | Sequence[int],
    mass: int | Sequence[int],
    category: str | Sequence[str],
    sample_names: list[str],
    skim_directory: list[str],
    eval_directory: list[str],
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

    year = _categories[0].split("_")[0]
    # assert that only one year is given
    assert all(cat.split("_")[0] == year for cat in _categories) 

    # input checks
    for spin in _spins:
        assert spin in spins
    for mass in _masses:
        assert mass in masses
    for category in _categories:
        assert category in categories
        
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
    print(f"going to load {len(all_matched_sample_names)} samples")
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
    sample_map: dict[str, list[str]],
    sample_data: dict[str, ak.Array],
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

    year = cat_data["year"]
    datacard_year = datacard_years[year]

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

#
    # write shapes
    #

    # just 5 equal-width bins for now
    bin_edges = np.linspace(0, 1, n_bins + 1) 

    # transpose the sample_map so that we have a "process -> year -> sample_names" mapping
    process_map = defaultdict(dict)
    for process_name, sample_names in sample_map.items():
        process_map[process_name] = sample_names

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
            for process_name, _map in process_map.items():
                if not nuisance.applies_to_process(process_name):
                    continue

                _hist_name, _process_name = hist_name, process_name
                if processes[process_name].get("data", False):
                    _hist_name = _process_name = "data_obs"
                full_hist_name = ShapeNuisance.create_full_name(_hist_name, year=datacard_year)
                h = hist.Hist.new.Variable(bin_edges, name=full_hist_name).Weight()
                hists[_process_name][(nuisance.get_combine_name(year=datacard_year), direction)] = h

            # fill histograms
            for process_name, sample_names in process_map.items():
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

                    mc_stack = Stack.from_dict(mc_hists)
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

    # return output paths
    return abs_shapes_path


def _write_datacard_mp(args: tuple[Any]) -> tuple[str, str]:
    return _write_datacard(*args)
