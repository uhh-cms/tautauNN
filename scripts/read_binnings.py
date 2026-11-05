#!/usr/bin/env python3
"""Extract datacard histogram bin edges into a WriteDatacards JSON file."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import uproot


SHAPES_PATTERN = re.compile(
    r"^shapes_(?:cat_)?(?P<category>.+)_spin_(?P<spin>\d+)_mass_(?P<mass>\d+)\.root$",
)


def read_binnings(datacard_directory: str | Path) -> dict[str, list[float]]:
    directory = Path(datacard_directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"datacard directory does not exist: {directory}")

    binnings: dict[str, list[float]] = {}
    shape_files = sorted(directory.glob("shapes_*.root"))
    if not shape_files:
        raise FileNotFoundError(f"no shapes_*.root files found in {directory}")

    for shape_file in shape_files:
        match = SHAPES_PATTERN.fullmatch(shape_file.name)
        if match is None:
            raise ValueError(f"could not parse shape file name: {shape_file.name}")

        category = match.group("category")
        key = f"{category}__s{match.group('spin')}__m{match.group('mass')}"
        if key in binnings:
            raise ValueError(f"duplicate binning key {key}")

        root_key_prefix = f"cat_{category}/"
        with uproot.open(shape_file) as root_file:
            nominal_keys = [
                key.removesuffix(";1")
                for key in root_file.keys()
                if key.startswith(root_key_prefix)
                and "__" not in key[len(root_key_prefix):]
                and key[len(root_key_prefix):] != "data_obs"
            ]
            if not nominal_keys:
                raise ValueError(f"no nominal process histograms found in {shape_file}")

            reference_edges = np.asarray(root_file[nominal_keys[0]].axis().edges())
            for nominal_key in nominal_keys[1:]:
                edges = np.asarray(root_file[nominal_key].axis().edges())
                if not np.array_equal(edges, reference_edges):
                    raise ValueError(
                        f"nominal histograms have different binning in {shape_file}: "
                        f"{nominal_keys[0]} and {nominal_key}",
                    )

        binnings[key] = reference_edges.tolist()

    return dict(sorted(binnings.items()))


def main(datacard_directory: str, output: str | None = None) -> Path:
    directory = Path(datacard_directory)
    output_path = Path(output) if output else directory / "bin_edges.json"
    binnings = read_binnings(directory)
    with output_path.open("w") as file:
        json.dump(binnings, file, indent=4)
        file.write("\n")
    print(f"wrote {len(binnings)} binnings to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datacard_directory", help="directory containing shapes_*.root files")
    parser.add_argument(
        "-o",
        "--output",
        help="output JSON path (default: <datacard_directory>/bin_edges.json)",
    )
    args = parser.parse_args()
    main(args.datacard_directory, args.output)
