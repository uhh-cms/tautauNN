
import os
import argparse
import json

from glob import glob
from subprocess import Popen
from tqdm import tqdm
from pathlib import Path

import uproot

from tautaunn.write_datacards_stack import processes


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skim-directory", type=str, default=None, required=True)
    parser.add_argument("--output-directory", type=str, default=None, required=True)
    return parser


def merge_sample(destination,
                 skim_files,):

    if Path(destination).suffix != ".root":
        raise ValueError(f"Destination {destination} must be a root file") 
    if any([Path(file).suffix != ".root" for file in skim_files]):
        raise ValueError(f"Skim files must be root files")

    with open(os.devnull, "w") as devnull:
        process = Popen(["hadd", "-T", "-f", destination, *skim_files], stdout=devnull) 
    out, err = process.communicate()
    if process.returncode != 0:
        raise Exception(err)


def get_sumw(skim_directory: str, 
             output_directory: str):
    sum_weights = {}
    pbar = tqdm(processes.items())
    for process, process_data in pbar: 
        pbar.set_description(f"Processing {process}")
        if process_data.get("data", False):
            continue
        print(f"Processing {process}")
        for pattern in process_data["sample_patterns"]:
            samples = glob(os.path.join(skim_directory, pattern))
            for sample in samples:
                destination = os.path.join(output_directory, f"{os.path.basename(sample)}.root")
                skim_files = glob(os.path.join(sample, "output_*.root"))
                merge_sample(destination, skim_files)
                with uproot.open(destination) as file:
                    sum_weights[os.path.basename(sample)] = float(file["h_eff"].values()[0])
    return sum_weights
                
                
def main():
    parser = make_parser()
    args = parser.parse_args()
    sum_weights = get_sumw(args.skim_directory,
                           args.output_directory)
    from IPython import embed; embed()
    with open(os.path.join(args.output_directory, "sum_weights.json"), "w") as file:
        json.dump(sum_weights, file)
        

if __name__ == "__main__":
    main()
