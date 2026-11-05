#!/usr/bin/env python3
"""Add the non-resonant HH process to existing stacked datacards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import os
import shutil
import tempfile
from pathlib import Path

import hist
import numpy as np
from tqdm import tqdm
import uproot

import tautaunn.config as cfg
from tautaunn.cat_selectors import category_factory
from tautaunn.nuisances import shape_nuisances, rate_nuisances
from tautaunn.write_datacards_stack import load_sample_data


spins = tuple(cfg.spins)
masses = tuple(cfg.masses)

EVAL_DIRECTORY = (
    "/data/dust/user/riegerma/taunn_data/store/EvaluateSkims/"
    "hbtres_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_"
    "ED10_LU8x128_CTdense_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_"
    "RSv6_fi80_lbn_ft_lt20_lr1_LBdefault_daurot_fatjet_composite_FIx5_SDx5/prod9/"
)

BINNING_FILE = (
    "/data/dust/user/jwulff/inference/remodel_cards/new_remodel_post_merge/"
    "bin_edges.json"
)
INPUT_DIRECTORY = "/data/dust/user/jwulff/inference/remodel_cards/new_remodel_post_merge"


def _hist(values, weights, edges, name):
    result = hist.Hist.new.Variable(edges, name=name).Weight()
    result.fill(**{name: np.asarray(values), "weight": np.asarray(weights)})
    values = result.view().value
    mask = values <= 0
    values[mask] = 1.0e-5
    result.view().variance[mask] = 0.0
    return result


def _bin_edges(value):
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    return sorted(set(value))


def _variation_hist(data, nuisance, direction, variable, edges, name, scale):
    skim_syst = "nominal"
    if nuisance.skim_systs:
        skim_syst = nuisance.skim_systs[direction == "down"]
    varied_variable = nuisance.get_varied_discriminator(variable, direction)
    varied_weight = nuisance.get_varied_full_weight(direction)
    return _hist(
        data[skim_syst][varied_variable],
        data[skim_syst][varied_weight] * scale,
        edges,
        name,
    )


def _weight_variation_hists(data, nuisance, variable, edges, name, scale):
    nominal_weight = next(iter(nuisance.weight_variations))
    varied = []
    for weight_name in nuisance.weight_variations[nominal_weight]:
        varied.append(_hist(
            data["nominal"][variable],
            data["nominal"][nuisance.get_varied_full_weight(weight_name)] * scale,
            edges,
            name,
        ))
    if nuisance.name == "pdf_shape":
        if len(varied) != 103:
            raise ValueError(f"expected 103 PDF variations, found {len(varied)}")
        reference = varied[0].view().value
        hessian = np.stack([h.view().value for h in varied[1:101]])
        uncertainty = ((hessian - reference) ** 2).sum(axis=0) ** 0.5
        alpha_s = 0.5 * (varied[102].view().value - varied[101].view().value)
        uncertainty = (uncertainty ** 2 + alpha_s ** 2) ** 0.5
        up = reference + uncertainty
        down = reference - uncertainty
    elif nuisance.name == "qcd_scale_shape":
        reference = varied[0].view().value
        alternatives = np.stack([h.view().value for h in varied[1:]])
        up = np.maximum(reference, alternatives.max(axis=0))
        down = np.minimum(reference, alternatives.min(axis=0))
    else:
        raise NotImplementedError(f"unsupported varied shape nuisance {nuisance.name}")
    result_up = hist.Hist.new.Variable(edges, name=name).Weight()
    result_down = hist.Hist.new.Variable(edges, name=name).Weight()
    result_up.view().value[...] = up
    result_down.view().value[...] = down
    return result_up, result_down


def _shape_hists(data, category, variable, edges, year):
    channel = category["selection"].extra["channel"]
    scale = cfg.luminosities[year]
    result = {}
    for nuisance in shape_nuisances.values():
        if (
            nuisance.skip
            or not nuisance.applies_to_process("ggHH_hbbhtt")
            or not nuisance.applies_to_channel(channel)
        ):
            continue
        if nuisance.skim_systs and any(s not in data for s in nuisance.skim_systs):
            continue
        if nuisance.weight_variations:
            up, down = _weight_variation_hists(
                data, nuisance, variable, edges, variable, scale,
            )
            result[(nuisance.get_combine_name(year=year), "up")] = up
            result[(nuisance.get_combine_name(year=year), "down")] = down
            continue
        for direction in nuisance.get_directions():
            result[(
                nuisance.get_combine_name(year=year),
                direction,
            )] = _variation_hist(
                data,
                nuisance,
                direction,
                variable,
                edges,
                variable if nuisance.is_nominal else f"{variable}_{nuisance.name}{direction}",
                scale,
            )
    return result


def _rate_effect(name, category, year):
    channel = category["selection"].extra["channel"]
    category_name = category["name"]
    for nuisance in rate_nuisances.values():
        if nuisance.name != name:
            continue
        for effect in nuisance.rate_effects:
            if (
                effect.applies_to_channel(channel)
                and effect.applies_to_category(category_name)
                and effect.applies_to_year(year)
                and effect.applies_to_process("ggHH_hbbhtt")
            ):
                return effect.effect
    return "-"


def _update_card(path, process_name, rate, category, year, shape_names):
    lines = path.read_text().splitlines()
    process_lines = [i for i, line in enumerate(lines) if line.split()[:1] == ["process"]]
    if len(process_lines) != 2:
        raise ValueError(f"expected two process lines in {path}")
    if process_name in lines[process_lines[0]].split():
        return
    lines[process_lines[0]] += f"  {process_name}"
    existing_ids = [int(value) for value in lines[process_lines[1]].split()[1:]]
    lines[process_lines[1]] += f"  {max(existing_ids) + 1}"
    rate_line = next(i for i, line in enumerate(lines) if line.split()[:1] == ["rate"])
    lines[rate_line] += f"  {rate:.4f}"

    shape_names = set(shape_names)
    parameter_names = set()
    for i, line in enumerate(lines):
        fields = line.split()
        if len(fields) < 2 or fields[1] not in {"lnN", "shape"}:
            continue
        parameter_names.add(fields[0])
        if fields[1] == "shape":
            fields.append("1" if fields[0] in shape_names else "-")
        else:
            fields.append(_rate_effect(fields[0], category, year))
        lines[i] = "  ".join(fields)

    n_existing_processes = len(lines[process_lines[0]].split()) - 1
    missing_parameters = []
    for nuisance_name in shape_names - parameter_names:
        missing_parameters.append(
            "  ".join([nuisance_name, "shape"] + ["-"] * (n_existing_processes - 1) + ["1"]),
        )
    for nuisance in rate_nuisances.values():
        effect = _rate_effect(nuisance.name, category, year)
        if effect != "-" and nuisance.name not in parameter_names:
            missing_parameters.append(
                "  ".join([nuisance.name, "lnN"] + ["-"] * (n_existing_processes - 1) + [effect]),
            )
    if missing_parameters:
        insert_at = next(
            (i for i, line in enumerate(lines) if line.split()[:2] == ["*", "autoMCStats"]),
            len(lines),
        )
        lines[insert_at:insert_at] = missing_parameters
    path.write_text("\n".join(lines) + "\n")


def main(
    year="2018",
    spins=spins,
    masses=masses,
    eval_directory=EVAL_DIRECTORY,
    binning_file=BINNING_FILE,
    input_directory=INPUT_DIRECTORY,
    output_directory=".",
    cache_directory=None,
    skim_systs=("JERup", "JERdown"),
    n_parallel=4,
):
    _spins = list(spins) if isinstance(spins, (tuple, list, set)) else [spins]
    _masses = list(masses) if isinstance(masses, (tuple, list, set)) else [masses]
    eval_directory = Path(eval_directory) / year
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    if cache_directory is None:
        cache_directory = os.environ["TN_DATACARD_CACHE_DIR"]
    with open(binning_file) as f:
        binnings = json.load(f)

    variable_pattern = "pdnn_m{mass}_s{spin}_hh"
    dnn_output_columns = [
        variable_pattern.format(spin=spin, mass=mass)
        for spin, mass in itertools.product(_spins, _masses)
    ]
    skim_directory = Path(os.environ[f"TN_SKIMS_{year}"])
    sample_data = load_sample_data(
        skim_directory=str(skim_directory),
        eval_directory=str(eval_directory),
        year=year,
        sample_name="GluGluToHHTo2B2Tau",
        skim_systs=list(skim_systs),
        dnn_output_columns=dnn_output_columns,
        n_parallel=1,
        cache_directory=str(cache_directory),
    ) 

    jobs = []
    for channel in cfg.channels:
        for name, selection in category_factory(channel).items():
            category_name = f"{year}_{channel}_{name}"
            category = {"selection": selection, "name": name}
            selected = {
                syst: data[selection(data, year=year)]
                for syst, data in sample_data.items()
            }
            for spin, mass in itertools.product(_spins, _masses):
                key = f"{category_name}__s{spin}__m{mass}"
                if key not in binnings:
                    continue
                edges = _bin_edges(binnings[key])
                jobs.append((category_name, category, selected, spin, mass, edges))

    def process_card(job):
        category_name, category, selected, spin, mass, edges = job
        variable = variable_pattern.format(spin=spin, mass=mass)
        hists = _shape_hists(selected, category, variable, edges, year)
        nominal = hists[("nominal", "")]
        process_name = f"ggHH_{year}_hbbhtt"
        card_stem = f"cat_{category_name}_spin_{spin}_mass_{mass}"
        source_shapes_path = input_directory / f"shapes_{card_stem}.root"
        source_card_path = input_directory / f"datacard_{card_stem}.txt"
        shapes_path = output_directory / f"shapes_{card_stem}.root"
        card_path = output_directory / f"datacard_{card_stem}.txt"
        if not shapes_path.exists():
            if not source_shapes_path.exists():
                raise FileNotFoundError(f"missing input shapes file {source_shapes_path}")
            shutil.copy2(source_shapes_path, shapes_path)
        if not card_path.exists():
            if not source_card_path.exists():
                raise FileNotFoundError(f"missing input datacard {source_card_path}")
            shutil.copy2(source_card_path, card_path)
        if not shapes_path.exists() or not card_path.exists():
            raise FileNotFoundError(f"missing datacard inputs for {category_name}, s{spin}, m{mass}")
        with uproot.open(shapes_path) as source_file:
            with tempfile.NamedTemporaryFile(
                dir=shapes_path.parent,
                prefix=f".{shapes_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            try:
                with uproot.recreate(temporary_path) as root_file:
                    for key in source_file.keys():
                        key_without_cycle = key.split(";", 1)[0]
                        source_object = source_file[key]
                        if hasattr(source_object, "to_hist"):
                            root_file[key_without_cycle] = source_object.to_hist()
                    root_file[f"cat_{category_name}/{process_name}"] = nominal
                    for (nuisance, direction), shape in hists.items():
                        if nuisance == "nominal":
                            continue
                        root_file[
                            f"cat_{category_name}/{process_name}__"
                            f"{nuisance}{direction.capitalize()}"
                        ] = shape
                os.replace(temporary_path, shapes_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        shape_names = [
            nuisance
            for nuisance, direction in hists
            if nuisance != "nominal" and direction in {"up", "down"}
        ]
        _update_card(
            card_path,
            process_name,
            nominal.sum().value,
            category,
            year,
            shape_names,
        )

    if n_parallel < 1:
        raise ValueError("n_parallel must be at least 1")
    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = [executor.submit(process_card, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="writing HH shapes"):
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", default="2018")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--input-directory", default=INPUT_DIRECTORY)
    parser.add_argument("--eval-directory", default=EVAL_DIRECTORY)
    parser.add_argument("--binning-file", default=BINNING_FILE)
    parser.add_argument("--cache-directory", default=None)
    parser.add_argument("--spins", nargs="+", type=int, default=spins)
    parser.add_argument("--masses", nargs="+", type=int, default=masses)
    parser.add_argument("--n-parallel", type=int, default=4)
    args = parser.parse_args()
    main(**vars(args))
