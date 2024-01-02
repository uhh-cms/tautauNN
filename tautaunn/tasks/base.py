# coding: utf-8

from __future__ import annotations

import os
import glob
import fnmatch
import math

import luigi
import law


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
            config.custom_content.append(("requirements", "(OpSysAndVer == \"CentOS7\")"))  # noqa

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


class SkimDirTask(Task):

    skim_dir = luigi.Parameter(
        default="$TN_SKIMS_2017",
        description="path to the skim directory; default: TN_SKIMS_2017",
    )

    @classmethod
    def resolve_param_values(cls, params: dict) -> dict:
        params = super().resolve_param_values(params)
        params["skim_dir"] = os.path.abspath(os.path.expandvars(os.path.expanduser(params["skim_dir"])))
        return params


class SkimTask(SkimDirTask):

    skim_name = luigi.Parameter()

    def store_parts(self) -> law.util.InsertableDict:
        parts = super().store_parts()
        parts["skim_name"] = self.skim_name
        return parts

    def get_skim_file(self, num):
        return law.LocalFileTarget(os.path.join(self.skim_dir, self.skim_name, f"output_{num}.root"))


class MultiSkimTask(SkimDirTask):

    skim_names = law.CSVParameter(
        default=("SKIM_*",),
        description="skim name pattern(s); default: SKIM_*",
    )
    skip_skim_names = law.CSVParameter(
        default=(),
        description="skim name pattern(s) to skip; default: empty",
    )

    @classmethod
    def resolve_param_values(cls, params: dict) -> dict:
        params = super().resolve_param_values(params)

        # expand skim names
        skim_names = []
        for pattern in params["skim_names"]:
            paths = glob.glob(os.path.join(params["skim_dir"], pattern))
            skim_names.extend(sorted(map(os.path.basename, paths)))
        params["skim_names"] = tuple(sorted(set(skim_names), key=skim_names.index))

        return params

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # determine selected skim names
        self.selected_skim_names = [
            skim_name
            for skim_name in self.skim_names
            if not any(fnmatch.fnmatch(skim_name, pattern) for pattern in self.skip_skim_names)
        ]


class SkimWorkflow(SkimTask, law.LocalWorkflow, HTCondorWorkflow, SlurmWorkflow):

    @law.workflow_property(attr="_skim_nums", cache=True)
    def skim_nums(self):
        return [
            int(os.path.basename(path)[7:-5])
            for path in glob.glob(os.path.join(self.skim_dir, self.skim_name, "output_*.root"))
        ]

    def create_branch_map(self):
        return self.skim_nums
