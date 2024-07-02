
from __future__ import annotations

import os
import re
import time
import itertools
from collections import defaultdict
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
    
    def output(self):
        # prepare the output directory
        dirname = f"{self.binning}{self.n_bins}"
        if self.output_suffix not in ("", law.NO_STR):
            dirname += f"_{self.output_suffix.lstrip('_')}"
        d = self.local_target(dirname, dir=True)
        # hotfix location in case TN_STORE_DIR is set to Marcel's
        pathlist = d.abs_dirname.split("/")
        path_user = pathlist[int(pathlist.index("user")+1)]
        if path_user != os.environ["USER"]: 
            abspath = d.abs_dirname.replace(path_user, os.environ["USER"])
            print(f"changing output path from {d.abs_dirname} to {abspath}")
            yn = input("continue? [y/n] ")
            if yn.lower() != "y":
                abspath = input("enter the correct path (should point to your $TN_STORE_DIR/GetBinning): ")
            d = law.LocalDirectoryTarget(abspath)
        return d.child("binnings.json", type="f")


    def run(self):
        from tautaunn.get_binning import get_binnings
        
        # prepare inputs
        inp = self.input()

        
        # prepare skim and eval directories
        skim_directory = os.environ[f"TN_SKIMS_{self.year}"] 
        eval_directory = inp[self.skim_names[0]].collection.dir.parent.path

        # define arguments
        binning_kwargs = dict(
            spin=list(self.spins),
            mass=list(self.masses),
            category=self.categories,
            skim_directory=skim_directory,
            eval_directory=eval_directory,
            output_directory=self.output().abs_dirname,
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
        
        self.chunk_size = 10
        

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def requires(self):
        evaluate_skims_requirement = EvaluateSkims.req(self, skim_name=self.skim_name)
        get_binning_requirement =  GetBinning.req(self,
                                                    year=self.year,
                                                    spins=self.spins,
                                                    masses=self.masses,
                                                    categories=self.categories,
                                                    binning=self.binning,
                                                    n_bins=self.n_bins,
                                                    variable=self.variable)
        get_sumw_requirement = GetSumW.req(self, year=self.year)

        # TODO: also add option for stacked histogramming to GetBinning and FillHists
        #if self.stacked:
            #get_binning_requirement = GetBinning.req(self,
                                                     #stacked=True,
                                                     #spins=self.spins,
                                                     #masses=self.masses,
                                                     #categories=self.categories,
                                                     #binning=self.binning,
                                                     #n_bins=self.n_bins,
                                                     #variable=self.variable)
            #get_sumw_requirement = {year: GetSumW.req(self, year=year)
                                    #for year in ["2016", "2016APV", "2017", "2018"]} 
        return (evaluate_skims_requirement, get_binning_requirement, get_sumw_requirement)

    
    def store_parts(self):
        return super().store_parts()
        

    def output(self):
        return law.FileCollection({num: self.local_target(f"output_{num}_hists.root", type="f")
                                   for num in self.skim_nums}) 

    def run(self):
        pass
        

