# coding: utf-8

import os
import functools
import subprocess
import urllib.parse
from tqdm import tqdm

def _wget(url, path):
    # create the parent directory, remove the file if existing
    dirname = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif os.path.exists(path):
        os.remove(path)

    # build the wget command and run it
    cmd = ["wget", "-O", path, url]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as e:
        raise Exception(f"download of url '{url}' failed: {e}")

    return path

samples = [
            "SKIM_NMSSM_XToYHTo2Tau2B_MX-500_MY-60", "SKIM_NMSSM_XToYHTo2Tau2B_MX-500_MY-70", "SKIM_NMSSM_XToYHTo2Tau2B_MX-500_MY-80", "SKIM_NMSSM_XToYHTo2Tau2B_MX-500_MY-90", "SKIM_NMSSM_XToYHTo2Tau2B_MX-500_MY-100", "SKIM_NMSSM_XToYHTo2Tau2B_MX-500_MY-125", "SKIM_NMSSM_XToYHTo2Tau2B_MX-500_MY-150", "SKIM_NMSSM_XToYHTo2Tau2B_MX-500_MY-250", "SKIM_NMSSM_XToYHTo2Tau2B_MX-500_MY-300",
            "SKIM_NMSSM_XToYHTo2Tau2B_MX-1000_MY-60", "SKIM_NMSSM_XToYHTo2Tau2B_MX-1000_MY-70", "SKIM_NMSSM_XToYHTo2Tau2B_MX-1000_MY-80", "SKIM_NMSSM_XToYHTo2Tau2B_MX-1000_MY-90", "SKIM_NMSSM_XToYHTo2Tau2B_MX-1000_MY-100", "SKIM_NMSSM_XToYHTo2Tau2B_MX-1000_MY-125", "SKIM_NMSSM_XToYHTo2Tau2B_MX-1000_MY-150", "SKIM_NMSSM_XToYHTo2Tau2B_MX-1000_MY-250", "SKIM_NMSSM_XToYHTo2Tau2B_MX-1000_MY-300",
           ]

eos_data_url = "https://cernbox.cern.ch/remote.php/dav/public-files/jXKsLXsBOvFDXJF/{}/{}?access_token=null"

outputpath = "/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_llr_2018_forTraining"

quote = functools.partial(urllib.parse.quote, safe="")

url = lambda dirname, filename: eos_data_url.format(quote(dirname), quote(filename))

for sample in samples:
    print("Downloading sample " + sample)
    # _wget(url(sample, "goodfiles.txt"), os.path.join(outputpath, sample, "goodfiles.txt"))
    filenames = []
    # with open(os.path.join(outputpath, sample, "goodfiles.txt"), "r") as gf:
    #     for line in gf:
    #         line = line.strip()
    #         if not line:
    #             continue
    #         filenames.append(os.path.basename(line))
    # for filename in tqdm(filenames):
    index = 0
    while True:
        filename = f"output_{index}.root"
        try:
            print(f"Trying to download file {filename} of sample {sample}")
            _wget(url(sample, filename), os.path.join(outputpath, sample, filename))
        except:
            print(f"Could not download file {filename} of sample {sample}")
            break
        index += 1