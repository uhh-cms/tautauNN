import os 
import itertools

import uproot
import numpy as np
import awkward as ak
import hist

from collections import defaultdict
from functools import reduce
from operator import mul
from fnmatch import fnmatch
from tqdm import tqdm

from tautaunn.shape_nuisances import shape_nuisances, ShapeNuisance
from tautaunn.write_datacards_stack import klub_weight_columns, klub_index_columns, klub_extra_columns, processes, datacard_years

from tautaunn.cat_selectors import selector, sel_baseline, category_factory
from tautaunn.config import luminosities, Sample, get_sample


_spins = [0,2]
_masses = [250,260,270,280,300,320,350,400,450,500,550,600,650,700,750,800,850,900,1000,1250,1500,1750,2000,2500,3000]


def skim_directory_to_year(skim_directory: str) -> str:
    skim_drectory_list = skim_directory.split("/")
    year_idx = skim_drectory_list.index("HHSkims") + 1
    year_suffix = skim_drectory_list[year_idx].split("_")[-1]
    if year_suffix in ["2016", "2017", "2018", "2016APV"]:
        return year_suffix
    elif year_suffix.startswith("UL"):
        return year_suffix.replace("UL", "20") 
    else:
        raise ValueError(f"could not determine year from skim directory {skim_directory}") 


def sample_name_to_process(sample_name: str) -> str:
    for process, process_dict in processes.items():
        if any([fnmatch(sample_name, pattern) for pattern in process_dict["sample_patterns"]]):
            return process
    raise ValueError(f"could not find a process for sample name {sample_name}")

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

del channel, name, sel, year

def load_klub_file(skim_directory: str,
                   sample_name: str,
                   file_name: str,
                   sum_weights: float,
                   is_data: bool = False,
                   treename: str = "HTauTauTree"):
    # define the branches to read here:
    klub_weight_column_patterns = klub_weight_columns + [f"{c}*" for c in klub_weight_columns] + ["IdFakeSF_deep_2d"]
    # all columns that should be loaded and kept later on
    persistent_columns = klub_index_columns + klub_extra_columns + sel_baseline.flat_columns
    # add all columns potentially necessary for selections
    persistent_columns += sum([
        cat["selection"].flat_columns
        for cat in categories.values()
    ], [])
    with uproot.open(os.path.join(skim_directory, sample_name, file_name)) as f:
        array = f[treename].arrays(filter_name=list(set(persistent_columns + (klub_weight_column_patterns))),
                                   cut=sel_baseline.str_repr.strip(),)
        
    if is_data:
        # fake weight for data
        array = ak.with_field(array, 1.0, "full_weight_nominal")
        persistent_columns.append("full_weight_nominal")
    else:
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
                    reduce(mul, (array[nuisance.get_varied_weight(c, direction)] for c in klub_weight_columns))/sum_weights,
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

    return array


def load_dnn_file(eval_directory: str,
                  sample_name: str,
                  file_name: str,
                  dnn_output_columns: list[str],
                  is_data: bool) -> ak.Array:

     
    # prepare expressions
    expressions = klub_index_columns + dnn_output_columns
    # extended output columns for variations if not data
    if not is_data:
        expressions += [f"{c}*" for c in dnn_output_columns]
    expressions = list(set(expressions))

    with uproot.open(os.path.join(eval_directory, sample_name, file_name)) as f:
        try: 
            array = f["hbtres"].arrays(
                filter_name=expressions,
                cut=sel_baseline.str_repr.strip(),
            )
        except uproot.exceptions.KeyInFileError:
            array = f["evaluations"].arrays(
                filter_name=expressions,
                cut=sel_baseline.str_repr.strip(),
            )
    return array

    
def load_file(
    skim_directory: str,
    eval_directory: str,
    sample_name: str,
    klub_file_name: str,
    eval_file_name: str,
    dnn_output_columns: list[str],
    is_data: bool,
    sum_weights: float = 1.0,
) -> tuple[ak.Array, float]:
    # load the klub file
    if is_data:
        assert sum_weights == 1.0
    else:
        assert sum_weights != 1.0

    klub_array = load_klub_file(skim_directory, sample_name, klub_file_name, sum_weights, is_data)

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

    return array

    
def fill_hists(binnings: dict,
               skim_directory: str,
               eval_directory: str,
               sample_name: str,
               klub_file_name: str,
               sum_weights: float = 1.0, 
               variable_pattern: str = "pdnn_m{mass}_s{spin}_hh"):

    # corresponding eval file name
    eval_file_name = klub_file_name.replace(".root", "_systs.root")
    # check if the sample is data
    is_data = False
    if any([any([fnmatch(sample_name, pattern) for pattern in processes[p]["sample_patterns"]])
            for p in processes if processes[p].get("data", False)]): 
        is_data = True

    dnn_output_columns = [
        variable_pattern.format(spin=spin, mass=mass)
        for spin, mass in itertools.product(_spins, _masses)
    ]

    # get the year
    year = skim_directory_to_year(skim_directory) 
    datacard_year = datacard_years[year]
    # get the process name
    process = sample_name_to_process(sample_name)
    # in case of signal reduce cols to the ones we need
    sample = get_sample(f"{year}_{sample_name}", silent=True)
    if sample is None:
        sample = Sample(sample_name, year=year)
    if sample.is_signal:
        dnn_output_columns = [c for c in dnn_output_columns if
                              ((f"_m{int(sample.mass)}_" in c)
                              and (f"_s{int(sample.spin)}_" in c))]

    # load the files
    array = load_file(skim_directory,
                      eval_directory,
                      sample_name,
                      klub_file_name,
                      eval_file_name,
                      dnn_output_columns,
                      is_data,
                      sum_weights)

    hists = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    print(f"filling histograms for {sample_name} in {year}")
    for key, bin_edges in tqdm(binnings.items()):
        cat_name, s, m = key.split("__") # cat_name is of the form year_channel_jet-cat_region__s{spin}_m{mass}
        jet_cat = cat_name.split("_")[2] # jet category like reolved{1,2}b, boosted
        spin, mass = s[1:], m[1:]
        if sample.is_signal and ((int(spin) != sample.spin) or (int(mass) != sample.mass)):
            continue
        for region in ["os_iso", "ss_iso", "os_noniso", "ss_noniso"]:
            cat = categories[cat_name.replace("os_iso", region)]
            cat_array = array[cat["selection"](array, year=year)]
            variable_name = variable_pattern.format(mass=mass, spin=spin)
            if is_data:
                h = hist.Hist.new.Variable(bin_edges, name=variable_name).Weight()
                h.fill(cat_array[variable_name], weight=cat_array["full_weight_nominal"])
                hists[cat['channel']][jet_cat][region][f"{process}__s{spin}__m{mass}"] = h
            else:
                # shape nuisances also contains nominal so we can loop over all and this
                # should get us all branches we need
                for nuisance in shape_nuisances.values():

                    if not nuisance.is_nominal and not nuisance.applies_to_channel(cat["channel"]): 
                        continue
                    
                    if not nuisance.applies_to_process(process):
                        continue

                    for direction in nuisance.get_directions():
                        hist_name = (variable_name if nuisance.is_nominal else f"{variable_name}_{nuisance.name}{direction}")
                        varied_variable_name = nuisance.get_varied_discriminator(variable_name, direction)
                        varied_weight_field = nuisance.get_varied_full_weight(direction)
                        full_hist_name = ShapeNuisance.create_full_name(hist_name, year=datacard_year)
                        combine_name = nuisance.get_combine_name(year=datacard_year)

                        h = hist.Hist.new.Variable(bin_edges, name=full_hist_name).Weight()
                        h.fill(cat_array[varied_variable_name], weight=cat_array[varied_weight_field])
                        hists[cat['channel']][jet_cat][region][f"{combine_name}__s{spin}__m{mass}"] = h
    return hists


def write_root_file(hists: dict,
                    filepath: str,):
    print(f"writing histograms to {filepath}")
    with uproot.recreate(filepath) as f:
        for channel, cat_dict in tqdm(hists.items()):
            for jet_cat, region_dict in cat_dict.items():
                for region, hist_dict in region_dict.items():
                    for name, hist in hist_dict.items():
                        f[f"{channel}/{jet_cat}/{region}/{name}"] = hist


def main():
    from argparse import ArgumentParser
    import json
    def make_parser():
        parser = ArgumentParser()
        parser.add_argument("--binning-file", "-b", type=str, required=True)
        parser.add_argument("--skim-directory", "-s", type=str, required=True)
        parser.add_argument("--eval-directory", "-e", type=str, required=True)
        parser.add_argument("--output-directory", "-o", type=str, required=True)
        parser.add_argument("--sample-name", "-n", type=str, required=True)
        parser.add_argument("--sum-weights", "-w", type=float, default=1.0)
        parser.add_argument("--klub-file-name", "-k", type=str, required=True)
        return parser
    
    parser = make_parser()
    args = parser.parse_args()
    with open(args.binning_file) as f:
        binnings = json.load(f)

    hists = fill_hists(binnings,
                       args.skim_directory,
                       args.eval_directory,
                       args.sample_name,
                       args.klub_file_name,)
    write_root_file(hists,
                    args.output_directory,
                    args.sample_name,
                    args.klub_file_name.replace(".root", "_hists.root"))


if __name__ == "__main__":
    main()

    


 
    
    