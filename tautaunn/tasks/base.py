# coding: utf-8

from __future__ import annotations

import os
import re
import glob
import math

import luigi
import law

import tautaunn.config as cfg


law.contrib.load("tasks", "htcondor", "slurm", "git", "root", "tensorflow")

_default_htcondor_flavor = law.config.get_expanded("analysis", "htcondor_flavor", "naf")
_default_slurm_flavor = law.config.get_expanded("analysis", "slurm_flavor", "maxwell")
_default_slurm_partition = law.config.get_expanded("analysis", "slurm_partition", "allgpu")


class Task(law.Task):

    version = luigi.Parameter(
        description="mandatory version that is encoded into output paths",
    )

    task_namespace = None
    message_cache_size = 25
    local_workflow_require_branches = False
    output_collection_cls = law.SiblingFileCollection
    default_store = "$TN_STORE_DIR"

    @classmethod
    def modify_param_values(cls, params: dict) -> dict:
        params = super().modify_param_values(params)
        params = cls.resolve_param_values(params)
        return params

    @classmethod
    def resolve_param_values(cls, params: dict) -> dict:
        return params

    @classmethod
    def req_params(cls, inst: Task, **kwargs) -> dict:
        # always prefer certain parameters given as task family parameters (--TaskFamily-parameter)
        _prefer_cli = law.util.make_set(kwargs.get("_prefer_cli", [])) | {
            "version", "workflow", "job_workers", "poll_interval", "walltime", "max_runtime",
            "retries", "acceptance", "tolerance", "parallel_jobs", "shuffle_jobs", "htcondor_cpus",
            "htcondor_gpus", "htcondor_memory", "htcondor_pool",
        }
        kwargs["_prefer_cli"] = _prefer_cli

        # build the params
        params = super().req_params(inst, **kwargs)

        return params

    @classmethod
    def split_skim_name(cls, skim_name: str) -> tuple[str, str, str, cfg.Sample]:
        m = re.match(rf"^({'|'.join(cfg.skim_dirs.keys())})_(.+)$", skim_name)
        if not m:
            raise ValueError(f"invalid skim name format '{skim_name}'")
        skim_year = m.group(1)
        sample_name = m.group(2)
        return sample_name, skim_year

    def store_parts(self) -> law.util.InsertableDict:
        parts = law.util.InsertableDict()

        # in this base class, just add the task class name
        parts["task_family"] = self.task_family

        # add the version when set
        if self.version is not None:
            parts["version"] = self.version

        return parts

    def local_path(self, *path, store=None, fs=None):
        # if no fs is set, determine the main store directory
        parts = ()
        if not fs:
            parts += (store or self.default_store,)

        # concatenate all parts that make up the path and join them
        parts += tuple(self.store_parts().values()) + path
        path = os.path.join(*map(str, parts))

        return path

    def local_target(self, *path, **kwargs):
        _dir = kwargs.pop("dir", False)
        store = kwargs.pop("store", None)
        fs = kwargs.get("fs", None)

        # select the target class
        cls = law.LocalDirectoryTarget if _dir else law.LocalFileTarget

        # create the local path
        path = self.local_path(*path, store=store, fs=fs)

        # create the target instance and return it
        return cls(path, **kwargs)


class BundleRepo(Task, law.git.BundleGitRepository, law.tasks.TransferLocalFile):

    version = None
    exclude_files = ["models", "old", ".law", ".data", ".github"]

    def get_repo_path(self):
        # required by BundleGitRepository
        return os.environ["TN_DIR"]

    def single_output(self):
        repo_base = os.path.basename(self.get_repo_path())
        return self.local_target(f"{repo_base}.{self.checksum}.tgz")

    def get_file_pattern(self):
        path = os.path.expandvars(os.path.expanduser(self.single_output().path))
        return self.get_replicated_path(path, i=None)

    def output(self):
        return law.tasks.TransferLocalFile.output(self)

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        # create the bundle
        bundle = law.LocalFileTarget(is_tmp="tgz")
        self.bundle(bundle)

        # log the size
        self.publish_message(f"size is {law.util.human_bytes(bundle.stat().st_size, fmt=True)}")

        # transfer the bundle
        self.transfer(bundle)


class HTCondorWorkflow(Task, law.htcondor.HTCondorWorkflow):

    transfer_logs = luigi.BoolParameter(
        default=True,
        significant=False,
        description="transfer job logs to the output directory; default: True",
    )
    max_runtime = law.DurationParameter(
        default=2.0,
        unit="h",
        significant=False,
        description="maximum runtime; default unit is hours; default: 2",
    )
    htcondor_cpus = luigi.IntParameter(
        default=law.NO_INT,
        significant=False,
        description="number of CPUs to request; empty value leads to the cluster default setting; "
        "empty default",
    )
    htcondor_gpus = luigi.IntParameter(
        default=law.NO_INT,
        significant=False,
        description="number of GPUs to request; empty value leads to the cluster default setting; "
        "empty default",
    )
    htcondor_memory = law.BytesParameter(
        default=law.NO_FLOAT,
        unit="MB",
        significant=False,
        description="requested memeory in MB; empty value leads to the cluster default setting; "
        "empty default",
    )
    htcondor_flavor = luigi.ChoiceParameter(
        default=_default_htcondor_flavor,
        choices=("naf", "cern", law.NO_STR),
        significant=False,
        description="the 'flavor' (i.e. configuration name) of the batch system; choices: "
        f"naf,cern,NO_STR; default: '{_default_htcondor_flavor}'",
    )

    exclude_params_branch = {
        "max_runtime", "htcondor_cpus", "htcondor_gpus", "htcondor_memory", "htcondor_flavor",
    }

    # mapping of environment variables to render variables that are forwarded
    htcondor_forward_env_variables = {
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # cached BundleRepo requirement to avoid race conditions during checksum calculation
        self.bundle_repo_req = BundleRepo.req(self)

    def htcondor_workflow_requires(self):
        reqs = super().htcondor_workflow_requires()
        reqs["repo"] = self.bundle_repo_req
        return reqs

    def htcondor_output_directory(self):
        # the directory where submission meta data and logs should be stored
        return self.local_target(dir=True)

    def htcondor_bootstrap_file(self):
        return law.JobInputFile(law.util.rel_path(__file__, "remote_bootstrap.sh"), share=True, render_job=True)

    def htcondor_job_config(self, config, job_num, branches):
        if self.htcondor_flavor == "cern":
            # https://batchdocs.web.cern.ch/local/submit.html#os-selection-via-containers
            config.custom_content.append(("MY.WantOS", "el9"))
        elif self.htcondor_flavor == "naf":
            # https://confluence.desy.de/display/IS/BIRD
            config.custom_content.append(("Request_OpSysAndVer", "\"RedHat9\""))

        # maximum runtime, compatible with multiple batch systems
        if self.max_runtime is not None and self.max_runtime > 0:
            max_runtime = int(math.floor(self.max_runtime * 3600)) - 1
            config.custom_content.append(("+MaxRuntime", max_runtime))
            config.custom_content.append(("+RequestRuntime", max_runtime))

        # request cpus
        if self.htcondor_cpus is not None and self.htcondor_cpus > 0:
            config.custom_content.append(("RequestCpus", self.htcondor_cpus))

        # request gpus
        if self.htcondor_gpus is not None and self.htcondor_gpus > 0:
            # e.g. https://confluence.desy.de/display/IS/GPU+on+NAF
            config.custom_content.append(("Request_GPUs", self.htcondor_gpus))

        # request memory
        if self.htcondor_memory is not None and self.htcondor_memory > 0:
            config.custom_content.append(("Request_Memory", self.htcondor_memory))

        # render variables
        config.render_variables["tn_bootstrap_name"] = "htcondor"
        if self.htcondor_flavor not in ("", law.NO_STR):
            config.render_variables["tn_htcondor_flavor"] = self.htcondor_flavor
        config.render_variables["tn_repo_bundle"] = self.bundle_repo_req.output().path

        # forward env variables
        for ev, rv in self.htcondor_forward_env_variables.items():
            config.render_variables[rv] = os.environ[ev]

        return config

    def htcondor_use_local_scheduler(self):
        # remote jobs should not communicate with ther central scheduler but with a local one
        return True

    def htcondor_destination_info(self, info: dict[str, str]) -> dict[str, str]:
        return super().htcondor_destination_info(info)


class SlurmWorkflow(law.slurm.SlurmWorkflow):

    transfer_logs = luigi.BoolParameter(
        default=True,
        significant=False,
        description="transfer job logs to the output directory; default: True",
    )
    max_runtime = law.DurationParameter(
        default=2.0,
        unit="h",
        significant=False,
        description="maximum runtime; default unit is hours; default: 2",
    )
    slurm_partition = luigi.Parameter(
        default=_default_slurm_partition,
        significant=False,
        description=f"target queue partition; default: {_default_slurm_partition}",
    )
    slurm_flavor = luigi.ChoiceParameter(
        default=_default_slurm_flavor,
        choices=("maxwell",),
        significant=False,
        description="the 'flavor' (i.e. configuration name) of the batch system; choices: "
        f"maxwell; default: '{_default_slurm_flavor}'",
    )

    exclude_params_branch = {"max_runtime", "slurm_partition", "slurm_flavor"}

    # mapping of environment variables to render variables that are forwarded
    slurm_forward_env_variables = {
        "TN_DIR": "tn_dir",
    }

    def slurm_output_directory(self):
        # the directory where submission meta data and logs should be stored
        return self.local_target(dir=True)

    def slurm_bootstrap_file(self):
        return law.JobInputFile(law.util.rel_path(__file__, "remote_bootstrap.sh"), share=True, render_job=True)

    def slurm_job_config(self, config, job_num, branches):
        # forward kerberos proxy
        if "KRB5CCNAME" in os.environ:
            kfile = os.environ["KRB5CCNAME"]
            kerberos_proxy_file = os.sep + kfile.split(os.sep, 1)[-1]
            if os.path.exists(kerberos_proxy_file):
                config.input_files["kerberosproxy_file"] = law.JobInputFile(
                    kerberos_proxy_file,
                    share=True,
                    render=False,
                )

                # set the pre command to extend potential afs permissions
                config.render_variables["tn_pre_setup_command"] = "aklog"

        # set job time
        if self.max_runtime is not None:
            job_time = law.util.human_duration(
                seconds=int(math.floor(self.max_runtime * 3600)) - 1,
                colon_format=True,
            )
            config.custom_content.append(("time", job_time))

        # set nodes
        config.custom_content.append(("nodes", 1))

        # custom, flavor dependent settings
        if self.slurm_flavor == "maxwell":
            # nothing yet
            pass

        # render variales
        config.render_variables["tn_bootstrap_name"] = "slurm"
        if self.slurm_flavor not in ("", law.NO_STR):
            config.render_variables["tn_slurm_flavor"] = self.slurm_flavor

        # custom tmp dir since slurm uses the job submission dir as the main job directory, and law
        # puts the tmp directory in this job directory which might become quite long; then,
        # python's default multiprocessing puts socket files into that tmp directory which comes
        # with the restriction of less then 80 characters that would be violated, and potentially
        # would also overwhelm the submission directory
        config.render_variables["law_job_tmp"] = "/tmp/law_$( basename \"$LAW_JOB_HOME\" )"  # noqa

        # forward env variables
        for ev, rv in self.slurm_forward_env_variables.items():
            config.render_variables[rv] = os.environ[ev]

        return config


class SkimTask(Task):

    skim_name = luigi.Parameter(
        description="the name and year of a skim in the format '<YEAR>_<SAMPLE>'; no default",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # get the skim directory and the sample name from the skim_name
        self.sample_name, self.skim_year = self.split_skim_name(self.skim_name)
        self.skim_dir = cfg.skim_dirs[self.skim_year]

        # check if the sample is already registered in the config,
        # and if not, it is likely a background sample, so create it
        self.sample = cfg.get_sample(self.skim_name, silent=True)
        if self.sample is None:
            self.sample = cfg.Sample(self.sample_name, year=self.skim_year)

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        parts["sample_directory"] = self.sample.directory_name
        parts.insert_after("version", "year", self.sample.year)
        return parts

    def get_skim_file(self, num):
        return law.LocalFileTarget(os.path.join(self.skim_dir, self.sample.directory_name, f"output_{num}.root"))


class MultiSkimTask(Task):

    skim_names = law.CSVParameter(
        default=("201*_*",),
        description="skim name pattern(s); default: 201*_*",
        brace_expand=True,
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

        # get all corresponding sample objects, creating new ones if they do not exist yet
        self.samples = [
            cfg.get_sample(skim_name, silent=True) or cfg.Sample(*SkimTask.split_skim_name(skim_name))
            for skim_name in self.skim_names
        ]


class SkimWorkflow(SkimTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow):

    chunk_size = 0

    # files known to be broken and should be skipped
    broken_files = {
        # "2016APV_TT_FullyLep": [13, 38, 55],
        # "2016APV_DY_PtZ0To50": [180],
        # "2016APV_TT_SemiLep": [77],
        # "2016_DY_0J": [190, 266, 279, 287],
        # "2016_DY_1J": [
        #     3, 4, 7, 9, 10, 12, 17, 21, 22, 28, 29, 33, 36, 43, 44, 57, 63, 64, 70, 90, 91, 92, 97, 104, 107, 108, 109,
        #     118, 127, 136, 143, 144, 146, 147, 151, 152, 154, 157, 160, 161, 162, 165, 166, 171, 172, 173, 175, 176,
        #     178, 182, 183, 184, 185, 188, 189, 191, 196, 197,
        # ],
        # "2016_WJets_HT0To70": [15],
        # "2016_TT_FullyLep": [50],
        # "2018_ttHToTauTau": [358, 402, 419, 445, 459, 472, 515, 540],
        # "2018_DY_Incl": [
        #     206, 214, 222, 230, 245, 268, 414, 422, 430, 438, 452, 461, 467, 469, 475, 494, 495, 504, 507, 508, 514,
        #     562, 581, 582,
        # ],
        # "2018_METB": [1, 39, 77, 115, 153, 190, 227, 264, 301, 338],
        # "2018_TT_FullyLep": [
        #     25, 59, 62, 69, 91, 95, 117, 141, 163, 185, 201, 221, 236, 256, 278, 304, 324, 347, 369, 389, 406, 420, 430,
        #     477, 499, 517, 521, 523, 524, 542, 570, 581, 590, 1858, 1887, 1932, 1949, 2028, 2048, 2068, 2106, 2143,
        #     2186, 2230, 2295, 2314,
        # ],
        # "2018_ST_tW_antitop": [4],
        # "2018_WplusHToTauTau": [3, 12],
        # "2018_TTWJetsToLNu": [2, 165, 183, 196],
        # "2018_ZHToTauTau": [9],
        # "2018_METD": [9, 34, 77, 106, 135, 161, 177, 195, 222, 232, 263, 290, 296, 314, 331, 391],
        # "2018_METC": [191, 228, 269, 309, 348],
        # "2018_MuonB": [51, 81, 97, 129, 159, 174, 189],
        # "2018_MuonC": [12, 356],
        # "2018_TT_Hadronic": [3, 115],
        # "2018_TTZToLLNuNu": [2, 13, 25, 40, 51, 67, 81, 95, 111, 124, 175, 211, 225, 238],
        # "2018_EGammaD": [9, 56, 72, 171, 185, 224],
        # "2018_EGammaA": [188, 212, 219, 229, 241, 252, 263, 271, 300, 306],
        # "2018_ST_tW_top": [3, 7, 8, 13],
        # "2018_EWKWPlus2Jets_WToLNu": [0, 8],
        # "2018_ZH_HToBB_ZToLL": [5, 76],
        # "2018_GluGluHToTauTau": [9],
        # "2018_ttHToNonbb": [9],
        # "2018_WminusHToTauTau": [8, 9, 10, 20],
        # "2018_EGammaC": [348],
        # "2018_ttHTobb": [78, 91, 104, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117],
        # "2018_TauD": [580],
        # "2018_DY_PtZ250To400": [1, 2, 127],
    }

    @law.workflow_property(attr="_skim_nums", cache=True)
    def skim_nums(self):
        nums = sorted(
            int(os.path.basename(path)[7:-5])
            for path in glob.glob(os.path.join(self.skim_dir, self.sample.directory_name, "output_*.root"))
        )
        # filter out broken files
        broken = self.broken_files.get(self.sample.skim_name, [])
        nums = [num for num in nums if num not in broken]
        # complain if no skim files are found
        if not nums:
            raise Exception(f"no skim files found in '{os.path.join(self.skim_dir, self.sample.directory_name)}'")
        return nums

    def create_branch_map(self):
        # create a branch map with one branch per skim file if the chunk size is 0
        if self.chunk_size == 0:
            return self.skim_nums
        # otherwise, create chunks
        return list(law.util.iter_chunks(self.skim_nums, self.chunk_size))

    def htcondor_destination_info(self, info):
        info["sample"] = self.sample.skim_name
        return info
