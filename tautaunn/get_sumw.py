import concurrent.futures
import os
import argparse
import json

from glob import glob
from subprocess import Popen, PIPE
from tqdm import tqdm
from pathlib import Path

import uproot

from tautaunn.config import processes


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skim-directory", type=str, default=None, required=True)
    parser.add_argument("--output-directory", type=str, default=None, required=True)
    parser.add_argument("--num-workers", type=int, default=None, help="Number of workers to use for parallel processing")
    return parser


def merge_sample(destination, skim_files):
    if not destination.endswith(".root"):
        raise ValueError(f"Destination {destination} must be a root file")
    if any([Path(file).suffix != ".root" for file in skim_files]):
        raise ValueError(f"Skim files must be root files")

    with open(os.devnull, "w") as devnull:
        # from man hadd: -T  "Do not merge trees" 
        # only works now because the sum_w info is not in the HTauTauTree
        process = Popen(["hadd", "-T", "-f", destination, *skim_files], stdout=devnull)
    out, err = process.communicate()
    if process.returncode != 0:
        raise Exception(err)


def process_sample(sample, output_directory):
    destination = os.path.join(output_directory, f"{os.path.basename(sample)}.root")
    skim_files = glob(os.path.join(sample, "output_*.root"))
    merge_sample(destination, skim_files)
    with uproot.open(destination) as file:
        return os.path.basename(sample), float(file["h_eff"].values()[0])


def get_sumw(skim_directory: str, 
             output_directory: str,
             num_workers: int = None):
    sum_weights = {}
    samples_to_process = []

    for process, process_data in processes.items():
        if process_data.get("data", False):
            continue
        for pattern in process_data["sample_patterns"]:
            samples = glob(os.path.join(skim_directory, pattern))
            samples_to_process.extend(samples)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_sample, sample, output_directory): sample for sample in samples_to_process}
        with tqdm(total=len(futures), desc="Processing samples") as pbar:
            for future in concurrent.futures.as_completed(futures):
                sample, weight = future.result()
                sum_weights[sample] = weight
                pbar.set_description(f"{sample} finished!")
                pbar.update(1)

    return sum_weights
                
                
def main():
    parser = make_parser()
    args = parser.parse_args()
    sum_weights = get_sumw(args.skim_directory,
                           args.output_directory,
                           args.num_workers)
    filename = os.path.join(args.output_directory, "sum_weights.json")
    with open(filename, "w") as file:
        json.dump(sum_weights, file)
        

if __name__ == "__main__":
    main()
