# coding: utf-8

"""
This script is used to determine the binning for the DNN output. The idea is to
just load the HH, TT and DY samples into memory (plus potentially all of their shifts)
and then derive a binning that asserts that none of the bins has an empty TT or DY contribution.
"""
import os
from functools import reduce
from operator import mul
import uproot
import numpy as np
import awkward as ak

from tautaunn.config import masses, spins, klub_index_columns, luminosities, btag_wps, pnet_wps
from cat_selectors import selector, sel_baseline

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
        filter_name=list(set(persistent_columns + ([] if is_data else klub_weight_column_patterns))),
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
    sum_gen_mc_weights = len(array) if is_data else float(f["h_eff"].values()[0])

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