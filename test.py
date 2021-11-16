# coding: utf-8

import os
import functools
import subprocess
import urllib


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
        subprocess.check_call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        raise Exception(f"download of url '{url}' failed: {e}")

    return path


eos_data_url = "https://cernbox.cern.ch/index.php/s/OdVMUBKql0vKMG4/download?path={}&files={}"

quote = functools.partial(urllib.parse.quote, safe="")

url = lambda dirname, filename: eos_data_url.format(quote(dirname), quote(filename))

_wget(url("SKIM_DY", "goodfiles.txt"), "goodfiles.txt")
