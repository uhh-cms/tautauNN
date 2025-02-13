
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

from subprocess import Popen
from glob import glob
from tqdm import tqdm
from pathlib import Path
import pickle


from tautaunn.tasks.base import SkimWorkflow, MultiSkimTask, HTCondorWorkflow, Task
from tautaunn.tasks.training import MultiFoldParameters
import tautaunn.config as cfg
from tautaunn.config import processes
#
from tautaunn.write_datacards import get_cache_path, expand_categories, write_datacards
from tautaunn.get_binning import get_binnings
from tautaunn.get_sumw import get_sumw
from tautaunn.cache_data import load_data
from tautaunn.fill_hists import fill_hists, write_root_file


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


# _default_categories = ("{year}_*tau_resolved?b_os_iso", "{year}_*tau_boosted_os_iso")
_default_categories = ("{year}_*tau_resolved1b_noak8_os_iso", "{year}_*tau_resolved2b_first_os_iso", "{year}_*tau_boosted_notres2b_os_iso")


class GetBinning(MultiSkimTask, EvaluationParameters):

    year = luigi.ChoiceParameter(
        default="2017",
        choices=("2016", "2016APV", "2017", "2018"),
        description="year to use; default: 2017",
    )
    categories = law.CSVParameter(
        default=_default_categories,
        description=f"comma-separated patterns of categories to produce; default: {','.join(_default_categories)}",
        brace_expand=True,
    )
    binning = luigi.ChoiceParameter(
        default="flatsguarded",
        choices=("flatsguarded", "flats_systs"),
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
        # return {
        # skim_name: EvaluateSkims.req(self, skim_name=skim_name)
        # for skim_name in self.skim_names
        # }
        return

    def output(self):
        # prepare the output directory
        dirname = f"{self.year}/{self.binning}{self.n_bins}"
        if self.output_suffix not in ("", law.NO_STR):
            dirname += f"_{self.output_suffix.lstrip('_')}"
        d = self.local_target(dirname, dir=True)
        # hotfix location in case TN_STORE_DIR is set to Marcel's
        pathlist = d.path.split("/")
        path_user = pathlist[int(pathlist.index("user") + 1)]
        if path_user != os.environ["USER"]:
            new_path = d.path.replace(path_user, os.environ["USER"])
            print(f"changing output path from {d.path} to {new_path}")
            d = self.local_target(new_path, dir=True)

        if self.spins == tuple(cfg.spins):
            spins_suffix = "all"
        elif len(self.spins) == 1:
            spins_suffix = f"{int(self.spins[0])}"
        else:
            spins_suffix = "-".join(map(str, (self.spins[0], self.spins[-1])))

        if self.masses == tuple(cfg.masses):
            masses_suffix = "all"
        elif len(self.masses) == 1:
            masses_suffix = f"{int(self.masses[0])}"
        else:
            masses_suffix = "-".join(map(str, (int(self.masses[0]), int(self.masses[-1]))))

        return d.child(f"bin_edges_y{self.year}_s{spins_suffix}_m{masses_suffix}.json", type="f")

    def run(self):

        # prepare inputs
        # inp = self.input()

        # prepare skim and eval directories
        skim_directory = os.environ[f"TN_SKIMS_{self.year}"]
        # new evals
        # eval_dir = ("/data/dust/user/riegerma/taunn_data/store/EvaluateSkims/"
        # "hbtres_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_"
        # "ED10_LU8x128_CTdense_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_RSv6_"
        # "fi80_lbn_ft_lt20_lr1_LBdefault_daurot_fatjet_composite_FIx5_SDx5/prod4_syst")

        # even newer evals
        eval_dir = ("/data/dust/user/riegerma/taunn_data/store/EvaluateSkims/"
                   "hbtres_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_"
                   "ED10_LU8x128_CTdense_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_RSv6_"
                   "fi80_lbn_ft_lt20_lr1_LBdefault_daurot_fatjet_composite_FIx5_SDx5/prod7")

        if "max-" in os.environ["HOSTNAME"]:
            eval_dir = eval_dir.replace("nfs", "gpfs")
        eval_directory = os.path.join(eval_dir, self.year)
        # define arguments
        binning_kwargs = dict(
            spin=list(self.spins),
            mass=list(self.masses),
            year=self.year,
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
    n_workers = luigi.IntParameter(
        default=4,
        description="number of workers to use for parallel processing; default: 4",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def output(self):
        dirname = f"{self.year}"
        d = self.local_target(dirname, dir=True)
        return d.child("sum_weights.json", type="f")

    def run(self):

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
            num_workers=self.n_workers,
        )
        # sort sum_weights by sample name
        sum_weights = dict(sorted(sum_weights.items()))
        filename = os.path.join(output_directory, "sum_weights.json")
        with open(filename, "w") as file:
            json.dump(sum_weights, file)


class CacheData(Task):

    year = luigi.ChoiceParameter(
        default="2017",
        choices=("2016", "2016APV", "2017", "2018"),
        description="year to use; default: 2017",
    )
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
    categories = law.CSVParameter(
        default=_default_categories,
        description=f"comma-separated patterns of categories to produce; default: {','.join(_default_categories)}",
        brace_expand=True,
    )
    variable_pattern = luigi.Parameter(
        default="pdnn_m{mass}_s{spin}_hh",
        description="variable to use; template values 'mass' and 'spin' are replaced automatically; "
        "default: 'pdnn_m{mass}_s{spin}_hh'",
    )
    parallel_read = luigi.IntParameter(
        default=4,
        description="number of parallel processes to use for reading; default: 4",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_dir = ("/data/dust/user/riegerma/taunn_data/store/EvaluateSkims/"
                    "hbtres_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_"
                    "ED10_LU8x128_CTdense_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_RSv6_"
                    "fi80_lbn_ft_lt20_lr1_LBdefault_daurot_fatjet_composite_FIx5_SDx5/prod7")

    def requires(self):
        pass

    def output(self):
        # get the hash
        cashe_path = get_cache_path(
            os.environ["TN_DATACARD_CACHE_DIR"],
            os.environ[f"TN_SKIMS_{self.year}"],
            os.path.join(self.eval_dir, self.year),
            self.year,
            "TT_SemiLep",
            [self.variable_pattern.format(mass=mass, spin=spin)
             for mass in self.masses for spin in self.spins],
        )
        h = Path(cashe_path).stem.split("_")[-1]
        # create an output FileCollection (each hypothesis gets its own file)
        return law.FileCollection({f"{self.year}_m{mass}_s{spin}":
            self.local_target(f"{self.year}_m{mass}_s{spin}_{h}.pkl")
            for mass in self.masses for spin in self.spins})

    def run(self):

        # cast arguments to lists
        _categories = expand_categories(self.categories)

        year = _categories[0].split("_")[0]
        # assert that only one year is given
        assert all(cat.split("_")[0] == year for cat in _categories)
        # get all the sample names
        sample_names = []
        for skim_name in os.listdir(os.path.join(self.eval_dir, self.year)):
            sample = cfg.get_sample(f"{self.year}_{skim_name}", silent=True)
            if sample is None:
                sample_name, skim_year = self.split_skim_name(f"{self.year}_{skim_name}")
                sample = cfg.Sample(sample_name, year=skim_year)
            skim_dir = os.path.join(cfg.skim_dirs[sample.year], sample.name)
            if os.path.isdir(skim_dir):
                sample_names.append(sample.name)
            else:
                raise ValueError(f"Cannot find skim directory for sample {sample.name} in year {sample.year}")

        paths_dict = load_data(year=self.year,
                               sample_names=sample_names,
                               skim_directory=os.environ[f"TN_SKIMS_{self.year}"],
                               eval_directory=os.path.join(self.eval_dir, self.year),
                               output_directory=self.output().first_target.absdirname,
                               cache_directory=os.environ["TN_DATACARD_CACHE_DIR"],
                               variable_pattern=self.variable_pattern,
                               n_parallel_read=self.parallel_read,
                               )
        return paths_dict


class WriteDatacardsJow(HTCondorWorkflow, law.LocalWorkflow):
    year = luigi.ChoiceParameter(
        default="2017",
        choices=("2016", "2016APV", "2017", "2018"),
        description="year to use; default: 2017",
    )
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
    variable_pattern = luigi.Parameter(
        default="pdnn_m{mass}_s{spin}_hh",
        description="variable to use; template values 'mass' and 'spin' are replaced automatically; "
        "default: 'pdnn_m{mass}_s{spin}_hh'",
    )
    n_bins = luigi.IntParameter(
        default=10,
        description="number of bins to use; default: 10",
    )
    binning = luigi.ChoiceParameter(
        default="flats_systs",
        choices=("flatsguarded", "flats_systs", "flats", "equald"),
        description="binning to use; choices: flatsguarded (on tt and dy); default: flatsguarded",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.categories = [cat.format(year=self.year) for cat in _default_categories]
        self.eval_dir = ("/data/dust/user/riegerma/taunn_data/store/EvaluateSkims/"
                    "hbtres_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_"
                    "ED10_LU8x128_CTdense_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_RSv6_"
                    "fi80_lbn_ft_lt20_lr1_LBdefault_daurot_fatjet_composite_FIx5_SDx5/prod7")

        # print(f"expecting the following hash")
        # h = get_cache_path(
        # os.environ["TN_DATACARD_CACHE_DIR"],
        # os.environ[f"TN_SKIMS_{self.year}"],
        # os.path.join(self.eval_dir, self.year),
        # self.year,
        # "TT_SemiLep",
        # [self.variable_pattern.format(mass=mass, spin=spin)
        # for mass in self.masses for spin in self.spins],
        # )
        # print(Path(h).stem.split("_")[-1])

    def requires(self):
        reqs = CacheData.req(self,
                             year=self.year,
                             spins=self.spins,
                             masses=self.masses,
                             categories=self.categories,
                             variable_pattern=self.variable_pattern)
        return reqs

    def output(self):
        categories = expand_categories(self.categories)
        # produces a filecollection for each channelxcategory combination consisting of datacard and shape files
        targets = {f"{cat}_datacard":
                   self.local_target(f"datacard_cat_{cat}_s{s}_m{m}.txt")
                   for m in self.masses for s in self.spins for cat in categories}
        targets.update({f"{cat}_shapes":
                        self.local_target(f"shapes_cat_{cat}_s{s}_m{m}.root")
                        for m in self.masses for s in self.spins for cat in categories})
        return law.SiblingFileCollection(targets)

    def create_branch_map(self):
        categories = expand_categories(self.categories)
        return [f"{s}_{m}" for m in self.masses for s in self.spins]

    def run(self):
        # load the sample_data
        paths_dict = self.input()
        spin, mass = self.branch_data.split("_")
        spin, mass = int(spin), int(mass)
        with open(paths_dict[f"{self.year}_m{mass}_s{spin}"].abspath, "rb") as file:
            sample_data = pickle.load(file)
        # get the categories
        write_datacards(sample_data=sample_data,
                        spin=spin,
                        mass=mass,
                        category=self.categories,
                        output_directory=self.output().first_target.absdirname,
                        binning=[self.n_bins, 0.0, 1.0, self.binning],
                        )


class FillHistsWorkflow(SkimWorkflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.chunk_size = 2


class FillHists(FillHistsWorkflow):
    categories = law.CSVParameter(
        default=_default_categories,
        description=f"comma-separated patterns of categories to produce; default: {','.join(_default_categories)}",
        brace_expand=True,
    )
    binning = luigi.ChoiceParameter(
        default="flatsguarded",
        choices=("flatsguarded", "flats_systs", "flats", "equald"),
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
        # evaluate_skims_requirement = EvaluateSkims.req(self, skim_name=self.skim_name)
        get_sumw_requirement = GetSumW.req(self, year=self.skim_year)
        if self.binning_file == law.NO_STR and self.binning != "equald":
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
        if self.chunk_size > 1:
            return law.FileCollection({branch: self.local_target(f"output_{branch}_hists.root") for branch in self.branch_data})
        else:
            return self.local_target(f"output_{self.branch_data}_hists.root")

    def run(self):
        inp = self.input()
        output = self.output()
        if isinstance(inp, tuple):
            assert inp[0].exists(), f"sum_weights file {inp[0]} does not exist"
            assert inp[1].exists(), f"binnings file {inp[1]} does not exist"
        else:
            assert inp.exists(), f"sum_weights file {inp} does not exist"
            if self.binning_file == law.NO_STR and self.binning != "equald":
                raise ValueError("binning file is not provided")

        if self.binning != "equald":
            binnings_file = inp[1].path if isinstance(inp, tuple) else self.binning_file
            with open(binnings_file, "r") as file:
                # json file now also includes a string that is the stopping reason
                binnings = json.load(file)
            binnings = {key: val[0] for key, val in binnings.items()}
        else:
            print(f"\nusing equal-distance binning with {self.n_bins} bins\n")
            binnings = None

        sum_weights_file = inp[0].path if isinstance(inp, tuple) else inp.path
        with open(sum_weights_file, "r") as file:
            sum_weights = json.load(file)

        sample = cfg.Sample(self.sample_name, year=self.skim_year)
        if sample.is_data:
            sum_w = 1.0
        else:
            sum_w = sum_weights[self.sample_name]

        # hardcode eval dir
        eval_dir = ("/data/dust/user/riegerma/taunn_data/store/EvaluateSkims/"
                   "hbtres_PSnew_baseline_LSmulti3_SSdefault_FSdefault_daurot_composite-default_extended_pair_"
                   "ED10_LU8x128_CTdense_ACTelu_BNy_LT50_DO0_BS4096_OPadamw_LR1.0e-03_YEARy_SPINy_MASSy_RSv6_"
                   "fi80_lbn_ft_lt20_lr1_LBdefault_daurot_fatjet_composite_FIx5_SDx5/prod7")

        print(f"chunk size: {self.chunk_size}", f"branch data: {self.branch_data}")
        if self.chunk_size > 1:
            for branch in self.branch_data:
                hists = fill_hists(skim_directory=self.abs_skim_dir,
                                   eval_directory=os.path.join(eval_dir, self.skim_year),
                                   sample_name=self.sample_name,
                                   klub_file_name=f"output_{branch}.root",
                                   category=self.categories,
                                   binnings=binnings,
                                   n_bins=self.n_bins,
                                   sum_weights=sum_w,)
                write_root_file(hists=hists,
                                filepath=output[branch].path)
        else:
            hists = fill_hists(skim_directory=self.abs_skim_dir,
                            eval_directory=os.path.join(eval_dir, self.skim_year),
                            sample_name=self.sample_name,
                            klub_file_name=f"output_{self.branch_data}.root",
                            category=self.categories,
                            binnings=binnings,
                            n_bins=self.n_bins,
                            sum_weights=sum_w,)
            write_root_file(hists=hists,
                            filepath=output.path)


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


class MergeHists(HTCondorWorkflow, law.LocalWorkflow):

    """
    class to merge the histograms that were filled in FillHists

    ------------

    2017_{TT, ST} take around 15 GB of memory

    2017_DY takes around 30 GB of memory

    not sure how it will look if we stack all years together
    """

    skim_names = law.CSVParameter(
        default=("201*_*",),
        description="skim name pattern(s); default: 201*_*",
        brace_expand=True,
    )

    hadd_parallel = luigi.IntParameter(
        default=4,
        description="number of parallel hadd processes to use; default: 4",
    )

    @classmethod
    def resolve_param_values(cls, params: dict) -> dict:
        params = super().resolve_param_values(params)

        # resolve skim_names based on directories existing on disk
        resolved_skim_names = []
        for year, sample_names in cfg.get_all_skim_names().items():
            for sample_name in sample_names:
                if law.util.multi_match((skim_name := f"{year}_{sample_name}"), params["skim_names"]):
                    resolved_skim_names.append(skim_name)
        params["skim_names"] = tuple(resolved_skim_names)

        return params

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # def requires(self):
        # return FillHistsWrapper.req(self)
        # pass

    def output(self):
        o_dict = {}
        for skim_name in self.skim_names:
            sample_name, skim_year = self.split_skim_name(skim_name)
            if "DY" in sample_name:
                o_dict[sample_name] = self.local_target(f"{skim_year}/{sample_name}_hists.root")
            for process in processes:
                if any(fnmatch(sample_name, pattern) for pattern in processes[process]["sample_patterns"]):
                    if process not in o_dict:
                        o_dict[process] = self.local_target(f"{skim_year}/{process}_hists.root")
                    else:
                        continue
        return o_dict

    def create_branch_map(self):
        dy_skims = [skim_name for skim_name in self.skim_names if "DY" in skim_name]
        sample_names = [self.split_skim_name(skim_name)[0] for skim_name in self.skim_names]
        requested_processes = list(set([process for process in processes
                               if any(fnmatch(sample_name, pattern)
                               for pattern in processes[process]["sample_patterns"]
                               for sample_name in sample_names)]))

        signal_processes = [process for process in requested_processes if processes[process].get("signal", False)]
        background_processes = [process for process in requested_processes
                                if (
                                    not processes[process].get("data", False) and
                                    not processes[process].get("signal", False) and
                                    not process == "QCD" and
                                    not process == "DY"
                                )]
        data_processes = [process for process in requested_processes if processes[process].get("data", False)]

        branch_map = {}
        if len(background_processes) > 0:
            branch_map.update({i: process for i, process in enumerate(background_processes)})
        if len(data_processes) > 0:
            branch_map.update({i + len(background_processes): process for i, process in enumerate(data_processes)})
        if len(signal_processes) > 0:
            branch_map.update({len(background_processes) + len(data_processes): signal_processes})
        if len(dy_skims) > 0:
            branch_map.update({i + len(background_processes) + len(data_processes) + len(signal_processes): sample for i, sample in enumerate(dy_skims)})
        return branch_map

    def run(self):

        def merge_sample(destination,
                         skim_files,
                         num_processes=4):

            if not os.path.exists(os.path.dirname(destination)):
                os.makedirs(os.path.dirname(destination))

            with open(os.devnull, "w") as devnull:
                process = Popen(["hadd", "-f", "-j", str(num_processes), destination, *skim_files], stdout=devnull)
            out, err = process.communicate()
            if process.returncode != 0:
                raise Exception(err)

        inp = self.input()
        output = self.output()
        if isinstance(self.branch_data, list):
            for process in tqdm(self.branch_data):
                skim_files = []
                for skim_name in self.skim_names:
                    sample_name, skim_year = self.split_skim_name(skim_name)
                    if any(fnmatch(sample_name, pattern) for pattern in processes[process]["sample_patterns"]):
                        skim_files.append(glob(f"{inp[skim_name].collection.dir.path}/output_*_hists.root"))
                merge_sample(output[process].path,
                             list(itertools.chain.from_iterable(skim_files)),)
        else:
            tic = time.time()
            print(f"merging {self.branch_data}")
            if "DY" in self.branch_data:
                skim_files = inp[self.branch_data].collection.dir.glob("output_*_hists.root")
                merge_sample(output[self.branch_data].path,
                             skim_files,
                             self.hadd_parallel)
                toc = time.time()
                print(f"merging took {(time.time()-tic):.2f} seconds")
            else:
                skim_files = []
                for skim_name in self.skim_names:
                    sample_name, skim_year = self.split_skim_name(skim_name)
                    if any(fnmatch(sample_name, pattern) for pattern in processes[self.branch_data]["sample_patterns"]):
                        skim_files.append(glob(f"{inp[skim_name].collection.dir.path}/output_*_hists.root"))
                merge_sample(output[self.branch_data].path,
                            list(itertools.chain.from_iterable(skim_files)),
                            self.hadd_parallel)
                toc = time.time()
                print(f"merging took {(toc-tic):.2f} seconds")

        # pbar = tqdm(self.output().items())
        # for process, output in pbar:
            # pbar.set_description(f"merging {process}")
            # skim_files = []
            # for skim_name in self.skim_names:
                # sample_name, skim_year = self.split_skim_name(skim_name)
                # if any(fnmatch(sample_name, pattern) for pattern in processes[process]["sample_patterns"]):
                # skim_files.append(glob(f"{inp[skim_name].collection.dir.path}/output_*_hists.root"))
            # merge_sample(output.path, list(itertools.chain.from_iterable(skim_files)))
