#!/usr/bin/env bash

action() {
    # Sets up environment variables to use in code and runs a minimal software setup based on conda.
    # Variables start with "TN_" (from TautauNn). When they are already set before this setup is
    # invoked, they are not overwritten, meaning that e.g. software install directories can be set
    # beforehand to a location of your choice. Example:
    #
    #   > TN_DATA_DIR=/my/personal/data_directory source setup.sh
    #
    # Variables defined by the setup:
    #   TN_DIR
    #       The absolute directory of the repository. Used to infer file locations relative to it.
    #   TN_DATA
    #       The main data directory where outputs, software, etc can be stored in. Internally, this
    #       serves as a default for e.g. $TN_SOFTWARE_DIR or $TN_CONDA_DIR.
    #   TN_SOFTWARE_DIR
    #       The directory where general software is installed.
    #   TN_CONDA_DIR
    #       The directory where conda and conda envs are installed.

    #
    # prepare local variables
    #

    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local micromamba_url="https://micro.mamba.pm/api/micromamba/linux-64/latest"
    local pyv="3.11"
    local user_name="$( whoami )"
    local remote_env="$( [ -z "${TN_REMOTE_ENV}" ] && echo "false" || echo "true" )"


    #
    # host and user depdenent variables
    #

    local host_matched="false"
    if [[ "$( hostname )" = max-*.desy.de ]]; then
        # maxwell
        export TN_DATA_DIR="/gpfs/dust/cms/user/${user_name}/taunn_data"
        export TN_SOFTWARE_DIR="${TN_DATA_DIR}/software_maxwell"
        export TN_REG_MODEL_DIR="/gpfs/dust/cms/user/riegerma/taunn_data/reg_models"
        export TN_REG_MODEL_DIR_TOBI="/gpfs/dust/cms/user/kramerto/taunn_data/store/RegTraining"
        export TN_SKIMS_2016APV="/gpfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_UL16APV"
        export TN_SKIMS_2016="/gpfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_UL16"
        export TN_SKIMS_2017="/gpfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_UL17"
        export TN_SKIMS_2018="/gpfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_UL18"
        export TN_SLURM_FLAVOR="maxwell"
        export TN_SLURM_PARTITION="allgpu"
        host_matched="true"
    elif [[ "$( hostname )" = naf*.desy.de ]] || [[ "$( hostname )" = bird*.desy.de ]] || [[ "$( hostname )" = batch*.desy.de ]]; then
        # naf
        export TN_DATA_DIR="/nfs/dust/cms/user/${user_name}/taunn_data"
        export TN_SOFTWARE_DIR="${TN_DATA_DIR}/software_naf"
        export TN_REG_MODEL_DIR="/nfs/dust/cms/user/riegerma/taunn_data/reg_models"
        export TN_REG_MODEL_DIR_TOBI="/nfs/dust/cms/user/kramerto/taunn_data/store/RegTraining"
        export TN_SKIMS_2016APV="/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_UL16APV"
        export TN_SKIMS_2016="/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_UL16"
        export TN_SKIMS_2017="/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_UL17"
        export TN_SKIMS_2018="/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_UL18"
        export TN_HTCONDOR_FLAVOR="naf"
        host_matched="true"
    elif [[ "$( hostname )" = *.cern.ch ]]; then
        # lxplus
        export TN_DATA_DIR="/eos/user/${user_name:0:1}/${user_name}/taunn_data"
        export TN_SOFTWARE_DIR="${TN_DATA_DIR}/software_lxplus"
        export TN_REG_MODEL_DIR=""
        export TN_REG_MODEL_DIR_TOBI=""
        export TN_SKIMS_2016APV="/eos/user/l/lportale/hhbbtautau/skims/SKIMS_UL16APV"
        export TN_SKIMS_2016="/eos/user/l/lportale/hhbbtautau/skims/SKIMS_UL16"
        export TN_SKIMS_2017="/eos/user/l/lportale/hhbbtautau/skims/SKIMS_UL17"
        export TN_SKIMS_2018="/eos/user/l/lportale/hhbbtautau/skims/SKIMS_UL18"
        export TN_HTCONDOR_FLAVOR="cern"
        host_matched="true"
    fi

    # complain when in a remote env and none of the above host patterns matched
    if ! ${host_matched}; then
        echo "no host pattern matched for host '$( hostname )'"
        if [ ! -z "${TN_REMOTE_ENV}" ]; then
            >&2 echo "pattern match required in remote env '${TN_REMOTE_ENV}', stopping setup"
            return "1"
        fi
    fi


    #
    # global variables
    # (TN = TautauNn)
    #

    # start exporting variables, potentially giving priority to already exported ones
    export TN_DIR="${this_dir}"
    export TN_DATA_DIR="${TN_DATA_DIR:-${TN_DIR}/data}"
    export TN_STORE_DIR="${TN_STORE_DIR:-${TN_DATA_DIR}/store}"
    export TN_SOFTWARE_DIR="${TN_SOFTWARE_DIR:-${TN_DATA_DIR}/software}"
    export TN_CONDA_DIR="${TN_CONDA_DIR:-${TN_SOFTWARE_DIR}/conda}"
    export TN_JOB_DIR="${TN_JOB_DIR:-${TN_DATA_DIR}/jobs}"
    export TN_LOCAL_SCHEDULER="${TN_LOCAL_SCHEDULER:-true}"
    export TN_SCHEDULER_HOST="${TN_SCHEDULER_HOST:-$( hostname )}"
    export TN_SCHEDULER_PORT="8088"
    export TN_WORKER_KEEP_ALIVE="${TN_WORKER_KEEP_ALIVE:-"${remote_env}"}"
    export TN_HTCONDOR_FLAVOR="${TN_HTCONDOR_FLAVOR:-naf}"
    export TN_SLURM_FLAVOR="${TN_SLURM_FLAVOR:-maxwell}"
    export TN_SLURM_PARTITION="${TN_SLURM_PARTITION:-allgpu}"

    # external variable defaults
    export LANGUAGE="${LANGUAGE:-en_US.UTF-8}"
    export LANG="${LANG:-en_US.UTF-8}"
    export LC_ALL="${LC_ALL:-en_US.UTF-8}"
    export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
    export VIRTUAL_ENV_DISABLE_PROMPT="${VIRTUAL_ENV_DISABLE_PROMPT:-1}"
    export TF_CPP_MIN_LOG_LEVEL="3"
    export MAMBA_ROOT_PREFIX="${TN_CONDA_DIR}"
    export MAMBA_EXE="${MAMBA_ROOT_PREFIX}/bin/micromamba"
    export CUDA_VISIBLE_DEVICES="0"


    #
    # minimal local software setup
    #

    export PYTHONPATH="${TN_DIR}:${PYTHONPATH}"

    # increase stack size
    ulimit -s unlimited

    # conda base environment
    local conda_missing="$( [ -d "${TN_CONDA_DIR}" ] && echo "false" || echo "true" )"
    if ${conda_missing}; then
        echo "installing conda/micromamba at ${TN_CONDA_DIR}"
        (
            mkdir -p "${TN_CONDA_DIR}"
            cd "${TN_CONDA_DIR}"
            curl -Ls "${micromamba_url}" | tar -xvj -C . "bin/micromamba"
            ./bin/micromamba shell hook -y --prefix="${TN_CONDA_DIR}" &> "micromamba.sh"
            mkdir -p "etc/profile.d"
            mv "micromamba.sh" "etc/profile.d"
            cat << EOF > ".mambarc"
changeps1: false
always_yes: true
channels:
  - conda-forge
EOF
        )
    fi

    # initialize conda
    source "${TN_CONDA_DIR}/etc/profile.d/micromamba.sh" "" || return "$?"
    micromamba activate || return "$?"
    echo "initialized conda/micromamba"

    # install packages
    if ${conda_missing}; then
        echo
        echo "setting up conda/micromamba environment"

        # conda packages (nothing so far)
        micromamba install \
            libgcc \
            bash \
            "python=${pyv}" \
            git \
            git-lfs \
            || return "$?"
        micromamba clean --yes --all

        # pip packages
        pip install --no-cache-dir -U pip setuptools wheel
        pip install --no-cache-dir -U \
            "ipython" \
            "notebook" \
            "tensorflow[and-cuda]" \
            "tensorboard_plugin_profile" \
            "numpy" \
            "scipy" \
            "scikit-learn" \
            "nvidia-pyindex" \
            "nvidia-tensorrt" \
            "uproot" \
            "awkward" \
            "livelossplot" \
            "tqdm" \
            "matplotlib" \
            "mplhep" \
            "flake8" \
            "flake8-commas" \
            "flake8-quotes" \
            "cmsml" \
            "hist" \
            "vector" \
            "shap" \
            "uniplot" \
            "git+https://github.com/riga/law.git@master" \
            || return "$?"
    fi


    #
    # law setup
    #

    export LAW_HOME="${LAW_HOME:-${TN_DIR}/.law}"
    export LAW_CONFIG_FILE="${LAW_CONFIG_FILE:-${TN_DIR}/law.cfg}"

    if which law &> /dev/null; then
        # source law's bash completion scipt
        source "$( law completion )" ""

        # silently index
        law index -q
    fi
}

# entry point
action "$@"
