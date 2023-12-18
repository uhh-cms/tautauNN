#!/usr/bin/env bash

action() {
    # Sets up environment variables to use in code and runs a minimal software setup based on conda.
    # Variables start with "TN_" (from TautauNn). When they are already set before this setup is
    # invoked, they are not overwritten, meaning that e.g. software install directories can be set
    # beforehand to a location of your choice. Example:
    #
    #   > TN_DATA_BASE=/my/personal/data_directory source setup.sh
    #
    # Variables defined by the setup:
    #   TN_BASE
    #       The absolute directory of the repository. Used to infer file locations relative to it.
    #   TN_DATA
    #       The main data directory where outputs, software, etc can be stored in. Internally, this
    #       serves as a default for e.g. $TN_SOFTWARE_BASE or $TN_CONDA_BASE.
    #   TN_SOFTWARE_BASE
    #       The directory where general software is installed.
    #   TN_CONDA_BASE
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


    #
    # host and user depdenent variables
    #

    if [[ "$( hostname )" = max-*.desy.de ]]; then
        # maxwell
        export TN_DATA_BASE="/gpfs/dust/cms/user/${user_name}/taunn_data"
        export TN_SOFTWARE_BASE="${TN_DATA_BASE}/software_maxwell"
        export TN_SKIMS_2017="/gpfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v4_17Jul23"
    elif [[ "$( hostname )" = naf-*.desy.de ]]; then
        # naf
        export TN_DATA_BASE="/nfs/dust/cms/user/${user_name}/taunn_data"
        export TN_SOFTWARE_BASE="${TN_DATA_BASE}/software_naf"
        export TN_SKIMS_2017"/nfs/dust/cms/user/kramerto/hbt_resonant_run2/HHSkims/SKIMS_uhh_2017_v4_17Jul23"
    elif [[ "$( hostname )" = lxplus*.cern.ch ]]; then
        # lxplus (use EOS!)
        # TODO: does this also match htcondor worker nodes?
        export TN_DATA_BASE="/eos/user/${user_name:0:1}/${user_name}/taunn_data"
        export TN_SOFTWARE_BASE="${TN_DATA_BASE}/software_lxplus"
        export TN_SKIMS_2017="/eos/user/t/tokramer/hhbbtautau/skims/2017"
    fi


    #
    # global variables
    # (TN = TautauNn)
    #

    # start exporting variables, potentially giving priority to already exported ones
    export TN_BASE="${this_dir}"
    export TN_DATA_BASE="${TN_DATA_BASE:-${TN_BASE}/data}"
    export TN_SOFTWARE_BASE="${TN_SOFTWARE_BASE:-${TN_DATA_BASE}/software}"
    export TN_CONDA_BASE="${TN_CONDA_BASE:-${TN_SOFTWARE_BASE}/conda}"

    # external variable defaults
    export LANGUAGE="${LANGUAGE:-en_US.UTF-8}"
    export LANG="${LANG:-en_US.UTF-8}"
    export LC_ALL="${LC_ALL:-en_US.UTF-8}"
    export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
    export VIRTUAL_ENV_DISABLE_PROMPT="${VIRTUAL_ENV_DISABLE_PROMPT:-1}"
    export TF_CPP_MIN_LOG_LEVEL="3"
    export MAMBA_ROOT_PREFIX="${TN_CONDA_BASE}"
    export MAMBA_EXE="${MAMBA_ROOT_PREFIX}/bin/micromamba"


    #
    # minimal local software setup
    #

    # increase stack size
    ulimit -s unlimited

    # conda base environment
    local conda_missing="$( [ -d "${TN_CONDA_BASE}" ] && echo "false" || echo "true" )"
    if ${conda_missing}; then
        echo "installing conda/micromamba at ${TN_CONDA_BASE}"
        (
            mkdir -p "${TN_CONDA_BASE}"
            cd "${TN_CONDA_BASE}"
            curl -Ls "${micromamba_url}" | tar -xvj -C . "bin/micromamba"
            ./bin/micromamba shell hook -y --prefix="${TN_CONDA_BASE}" &> "micromamba.sh"
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
    source "${TN_CONDA_BASE}/etc/profile.d/micromamba.sh" "" || return "$?"
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
        pip install -U pip setuptools wheel
        pip install -U \
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
            || return "$?"
    fi
}

# entry point
action "$@"
