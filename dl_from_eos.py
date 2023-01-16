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

samples = [#"SKIM_DYmerged", "SKIM_DY_NLO_PT100To250", "SKIM_ggF_BulkGraviton_m1000", "SKIM_ggF_BulkGraviton_m250", 
           "SKIM_ggF_BulkGraviton_m300", "SKIM_ggF_BulkGraviton_m450", "SKIM_ggF_BulkGraviton_m700", "SKIM_ggF_Radion_m1000", "SKIM_ggF_Radion_m250", "SKIM_ggF_Radion_m300", "SKIM_ggF_Radion_m450", "SKIM_ggF_Radion_m700", "SKIM_TT_fullyHad",
"SKIM_DY_NLO", "SKIM_DY_NLO_PT250To400", "SKIM_ggF_BulkGraviton_m1250", "SKIM_ggF_BulkGraviton_m2500", "SKIM_ggF_BulkGraviton_m3000", "SKIM_ggF_BulkGraviton_m500", "SKIM_ggF_BulkGraviton_m750", "SKIM_ggF_Radion_m1250", "SKIM_ggF_Radion_m2500", "SKIM_ggF_Radion_m3000", "SKIM_ggF_Radion_m500", "SKIM_ggF_Radion_m750", "SKIM_TT_fullyLep",
"SKIM_DY_NLO_0j", "SKIM_DY_NLO_PT400To650", "SKIM_ggF_BulkGraviton_m1500", "SKIM_ggF_BulkGraviton_m260", "SKIM_ggF_BulkGraviton_m320", "SKIM_ggF_BulkGraviton_m550", "SKIM_ggF_BulkGraviton_m800", "SKIM_ggF_Radion_m1500", "SKIM_ggF_Radion_m260", "SKIM_ggF_Radion_m320", "SKIM_ggF_Radion_m550", "SKIM_ggF_Radion_m800", "SKIM_TT_semiLep",
"SKIM_DY_NLO_1j", "SKIM_DY_NLO_PT50To100", "SKIM_ggF_BulkGraviton_m1750", "SKIM_ggF_BulkGraviton_m270", "SKIM_ggF_BulkGraviton_m350", "SKIM_ggF_BulkGraviton_m600", "SKIM_ggF_BulkGraviton_m850", "SKIM_ggF_Radion_m1750", "SKIM_ggF_Radion_m270", "SKIM_ggF_Radion_m350", "SKIM_ggF_Radion_m600", "SKIM_ggF_Radion_m850",
"SKIM_DY_NLO_2j", "SKIM_DY_NLO_PT650ToInf", "SKIM_ggF_BulkGraviton_m2000", "SKIM_ggF_BulkGraviton_m280", "SKIM_ggF_BulkGraviton_m400", "SKIM_ggF_BulkGraviton_m650", "SKIM_ggF_BulkGraviton_m900", "SKIM_ggF_Radion_m2000", "SKIM_ggF_Radion_m280", "SKIM_ggF_Radion_m400", "SKIM_ggF_Radion_m650", "SKIM_ggF_Radion_m900"]

eos_data_url = "https://cernbox.cern.ch/remote.php/dav/public-files/TJQEEcvlOQRUgjs/{}/{}?access_token=null"

outputpath = "/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_llr_2018_forTraining"

quote = functools.partial(urllib.parse.quote, safe="")

url = lambda dirname, filename: eos_data_url.format(quote(dirname), quote(filename))

for sample in samples:
    print("Downloading sample " + sample)
    _wget(url(sample, "goodfiles.txt"), os.path.join(outputpath, sample, "goodfiles.txt"))
    filenames = []
    with open(os.path.join(outputpath, sample, "goodfiles.txt"), "r") as gf:
        for line in gf:
            line = line.strip()
            if not line:
                continue
            filenames.append(os.path.basename(line))
    for filename in tqdm(filenames):
            _wget(url(sample, filename), os.path.join(outputpath, sample, filename))
