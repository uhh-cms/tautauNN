
from __future__ import annotations

import os
import re
import time
import itertools
from collections import defaultdict
from fnmatch import fnmatch
import json

import luigi
import law
import numpy as np
import tensorflow as tf
import awkward as ak


from tautaunn.tasks.base import SkimWorkflow, MultiSkimTask, HTCondorWorkflow, Task
from tautaunn.tasks.training import MultiFoldParameters, ExportEnsemble
from tautaunn.util import calc_new_columns
from tautaunn.tf_util import get_device
import tautaunn.config as cfg

from tautaunn.tasks.datacards import EvaluateSkims
from tautaunn.write_datacards_stack import processes


class EvaluationParameters(MultiFoldParameters):

    spins = law.CSVParameter(
        cls=luigi.IntParameter,
        default=tuple(cfg.spins),
        description=f"spins to evaluate; default: {','.join(map(str, cfg.spins))}",
        brace_expand=True,
    )
    masses = law.CSVParameter(
        cls=luigi.FloatParameter,
        default=tuple(cfg.masses),
        description=f"masses to evaluate; default: {','.join(map(str, cfg.masses))}",
        brace_expand=True,
    )


_default_categories = ("{year}_*tau_resolved?b_os_iso", "{year}_*tau_boosted_os_iso")

class GetBinning(MultiSkimTask, EvaluationParameters):
    
    year = luigi.ChoiceParameter(
        default="2017",
        choices=("2016", "2016APV", "2017", "2018"),
        description="year to use; default: 2017",
    )
    binning = luigi.ChoiceParameter(
        default="flatsguarded",
        choices=("flatsguarded", "flats"),
        description="binning to use; choices: flatsguarded (on tt and dy); default: flatsguarded",
    )
    n_bins = luigi.IntParameter(
        default=10,
        description="number of bins to use; default: 10",
    )
    variable = luigi.Parameter(
        default="pdnn_m{mass}_s{spin}_hh",
        description="variable to use; template values 'mass' and 'spin' are replaced automatically; "
        "default: 'pdnn_m{mass}_s{spin}_hh'",
    )
    parallel_read = luigi.IntParameter(
        default=4,
        description="number of parallel processes to use for reading; default: 4",
    )
    parallel_write = luigi.IntParameter(
        default=4,
        description="number of parallel processes to use for writing; default: 4",
    )
    output_suffix = luigi.Parameter(
        default=law.NO_STR,
        description="suffix to append to the output directory; default: ''",
    )
    rewrite_existing = luigi.BoolParameter(
        default=False,
        significant=False,
        description="whether to rewrite existing datacards; default: False",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # reduce skim_names to only needed ones
        self.skim_names = [
            name for name in self.skim_names
            if any([s in name for s in ["TT_", "DY_", "Rad", "Grav"]]) and self.year == name.split("_")[0]
        ]

        self.categories = [c.format(year=self.year) for c in _default_categories]

            
    def requires(self):
        return {
            skim_name: EvaluateSkims.req(self, skim_name=skim_name)
            for skim_name in self.skim_names
        }
        #return
    
    def output(self):
        # prepare the output directory
        dirname = f"{self.year}/{self.binning}{self.n_bins}"
        if self.output_suffix not in ("", law.NO_STR):
            dirname += f"_{self.output_suffix.lstrip('_')}"
        d = self.local_target(dirname, dir=True)
        # hotfix location in case TN_STORE_DIR is set to Marcel's
        pathlist = d.path.split("/")
        path_user = pathlist[int(pathlist.index("user")+1)]
        if path_user != os.environ["USER"]: 
            new_path = d.path.replace(path_user, os.environ["USER"])
            print(f"changing output path from {d.path} to {new_path}")
            d = self.local_target(new_path, dir=True)
        return d.child("bin_edges.json", type="f")


    def run(self):
        from tautaunn.get_binning import get_binnings
        
        # prepare inputs
        # inp = self.input()
        
        # prepare skim and eval directories
        skim_directory = os.environ[f"TN_SKIMS_{self.year}"] 
        eval_dir = ("/nfs/dust/cms/user/riegerma/taunn_data/store/EvaluateSkims/"
                    "hbtres_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_"
                    "ED10_LU8x128_CTdense_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_RSv6_"
                    "fi80_lbn_ft_lt20_lr1_LBdefault_daurot_fatjet_composite_FIx5_SDx5/prod3_syst")
        if "max-" in os.environ["HOSTNAME"]:
            eval_dir = eval_dir.replace("nfs", "gpfs") 
        eval_directory = os.path.join(eval_dir, self.year)
        # define arguments
        binning_kwargs = dict(
            spin=list(self.spins),
            mass=list(self.masses),
            category=self.categories,
            skim_directory=skim_directory,
            eval_directory=eval_directory,
            output_file=self.output().path,
            variable_pattern=self.variable,
            # force using all samples, disabling the feature to select a subset
            # sample_names=[sample_name.replace("SKIM_", "") for sample_name in sample_names],
            binning=(self.n_bins, 0.0, 1.0, self.binning),
            n_parallel_read=self.parallel_read,
            n_parallel_write=self.parallel_write,
            cache_directory=os.path.join(os.environ["TN_DATA_DIR"], "datacard_cache"),
            skip_existing=not self.rewrite_existing,
        )

        # create the cards
        get_binnings(**binning_kwargs)


class GetSumW(Task):
    
    year = luigi.ChoiceParameter(
        default="2017",
        choices=("2016", "2016APV", "2017", "2018"),
        description="year to use; default: 2017",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def output(self):
        dirname = f"{self.year}"
        d = self.local_target(dirname, dir=True)
        return d.child("sum_weights.json", type="f")

    def run(self):
        from tautaunn.get_sumw import get_sumw

        # prepare inputs
        inp = self.input()

        skim_directory = os.environ[f"TN_SKIMS_{self.year}"]
        output_directory = self.output().abs_dirname
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # define arguments
        sum_weights = get_sumw(
            skim_directory=skim_directory,
            output_directory=output_directory,
        )
        filename = os.path.join(output_directory, "sum_weights.json")
        with open(filename, "w") as file:
            json.dump(sum_weights, file)


class FillHistsWorkflow(SkimWorkflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        #self.chunk_size = 10
        

class FillHists(FillHistsWorkflow, EvaluationParameters):
    categories = law.CSVParameter(
        default=_default_categories,
        description=f"comma-separated patterns of categories to produce; default: {','.join(_default_categories)}",
        brace_expand=True,
    )
    binning = luigi.ChoiceParameter(
        default="flatsguarded",
        choices=("flatsguarded", "flats"),
        description="binning to use; choices: flatsguarded (on tt and dy); default: flatsguarded",
    )
    n_bins = luigi.IntParameter(
        default=10,
        description="number of bins to use; default: 10",
    )
    variable = luigi.Parameter(
        default="pdnn_m{mass}_s{spin}_hh",
        description="variable to use; template values 'mass' and 'spin' are replaced automatically; "
        "default: 'pdnn_m{mass}_s{spin}_hh'",
    )
    binning_file = luigi.Parameter(
        default=law.NO_STR,
        description="path to a binning file; default: ''",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sample_name, self.skim_year = self.split_skim_name(self.skim_name)


    @property
    def priority(self):
        # higher priority value = picked earlier by scheduler
        # priotize (tt, ttH, ttZ > data > rest) across all years
        if re.match(r"^(TT_SemiLep|TT_FullyLep|ttHToTauTau|TTZToLLNuNu)$", self.sample.name):
            return 10
        if re.match(r"^(EGamma|MET|Muon|Tau)(A|B|C|D|E|F|G|H)$", self.sample.name):
            return 5
        return 1

    def requires(self):
        #evaluate_skims_requirement = EvaluateSkims.req(self, skim_name=self.skim_name)
        get_sumw_requirement = GetSumW.req(self, year=self.skim_year)
        if self.binning_file == law.NO_STR:
            get_binning_requirement = GetBinning.req(self,
                                                     year=self.skim_year,
                                                     spins=self.spins,
                                                     masses=self.masses,
                                                     binning=self.binning,
                                                     n_bins=self.n_bins,
                                                     variable=self.variable)
            return get_sumw_requirement, get_binning_requirement
        else:
            return get_sumw_requirement

    
    def store_parts(self):
        return super().store_parts()
        

    def output(self):
        return self.local_target(f"output_{self.branch_data}_hists.root")


    def run(self):
        from tautaunn.fill_hists import fill_hists, write_root_file
        inp = self.input()
        if isinstance(inp, tuple):
            assert inp[0].exists(), f"sum_weights file {inp[0]} does not exist"
            assert inp[1].exists(), f"binnings file {inp[1]} does not exist"
        else:
            assert inp.exists(), f"sum_weights file {inp} does not exist"
            if self.binning_file == law.NO_STR:
                raise ValueError("binning file is not provided")

        binnings_file = inp[1].path if isinstance(inp, tuple) else self.binning_file
        sum_weights_file = inp[0].path if isinstance(inp, tuple) else inp.path
        with open(binnings_file, "r") as file:
            binnings = json.load(file)
        with open(sum_weights_file, "r") as file:
            sum_weights = json.load(file)
        
        sample = cfg.Sample(self.sample_name, year=self.skim_year)
        if sample.is_data:
            sum_w = 1.0
        else:
            sum_w = sum_weights[self.sample_name]
        # hardcode eval dir 
        eval_dir = ("/nfs/dust/cms/user/riegerma/taunn_data/store/EvaluateSkims/"
                    "hbtres_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_"
                    "ED10_LU8x128_CTdense_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_RSv6_"
                    "fi80_lbn_ft_lt20_lr1_LBdefault_daurot_fatjet_composite_FIx5_SDx5/prod3_syst")

        
        hists = fill_hists(binnings=binnings,
                           skim_directory=self.skim_dir,
                           eval_directory=os.path.join(eval_dir, self.skim_year),
                           sample_name=self.sample_name,
                           klub_file_name=f"output_{self.branch_data}.root",
                           sum_weights=sum_w,)
        write_root_file(hists=hists,
                        filepath=self.output().path)
        

class FillHistsWrapper(MultiSkimTask, law.WrapperTask):

    def requires(self):
        # reduce requirements to only the skims that are actually listed in processes
        # otherwise the task complains that there's no sum_w present for the skim
        # get a flat list of all sample_patterns
        sample_patterns = list(itertools.chain.from_iterable([processes[p]["sample_patterns"] for p in processes]))
        return {
            skim_name: FillHists.req(self, skim_name=skim_name)
            for skim_name in self.skim_names
            if any(fnmatch(self.split_skim_name(skim_name)[0], pattern) for pattern in sample_patterns)
        }

                           
class MergeHists(MultiSkimTask, Task):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def requires(self):
        return FillHistsWrapper.req(self)
    
    def output(self):
        o_dict = {}
        for skim_name in self.skim_names:
            sample_name, skim_year = self.split_skim_name(skim_name)
            for process in processes:
                if any(fnmatch(sample_name, pattern) for pattern in processes[process]["sample_patterns"]):
                    if process not in o_dict:
                        o_dict[process] = self.local_target(f"{skim_year}/{process}_hists.root")
                    else:
                        continue
        return o_dict

    def run(self):

        from subprocess import Popen
        from glob import glob
        from tqdm import tqdm

        def merge_sample(destination,
                         skim_files,):
                        
            if not os.path.exists(os.path.dirname(destination)):
                os.makedirs(os.path.dirname(destination))

            with open(os.devnull, "w") as devnull:
                process = Popen(["hadd", "-f", "-j", "5", destination, *skim_files], stdout=devnull) 
            out, err = process.communicate()
            if process.returncode != 0:
                raise Exception(err)

        inp = self.input()
        pbar = tqdm(self.output().items())
        for process, output in pbar: 
            pbar.set_description(f"merging {process}")
            skim_files = []
            for skim_name in self.skim_names:
                sample_name, skim_year = self.split_skim_name(skim_name)
                if any(fnmatch(sample_name, pattern) for pattern in processes[process]["sample_patterns"]):
                    skim_files.append(glob(f"{inp[skim_name].collection.dir.path}/output_*_hists.root"))
            merge_sample(output.path, list(itertools.chain.from_iterable(skim_files)))
            