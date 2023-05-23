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

    local miniconda_source="https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh"
    local pyv="3.9"


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


    #
    # minimal local software setup
    #

    # increase stack size
    ulimit -s unlimited

    # conda base environment
    local conda_missing="$( [ -d "${TN_CONDA_BASE}" ] && echo "false" || echo "true" )"
    if ${conda_missing}; then
        echo "installing conda at ${TN_CONDA_BASE}"
        (
            wget "${miniconda_source}" -O setup_miniconda.sh &&
            bash setup_miniconda.sh -b -u -p "${TN_CONDA_BASE}" &&
            rm setup_miniconda.sh &&
            cat << EOF >> "${TN_CONDA_BASE}/.condarc"
changeps1: false
channels:
- conda-forge
- defaults
EOF
        )
    fi

    # initialize conda
    local __conda_setup="$( "${TN_CONDA_BASE}/bin/conda" "shell.$( ${shell_is_zsh} && echo "zsh" || echo "bash" )" "hook" 2> /dev/null )"
    if [ "$?" = "0" ]; then
        eval "${__conda_setup}"
    else
        if [ -f "${TN_CONDA_BASE}/etc/profile.d/conda.sh" ]; then
            . "${TN_CONDA_BASE}/etc/profile.d/conda.sh"
        else
            export PATH="${TN_CONDA_BASE}/bin:${PATH}"
        fi
    fi
    echo "initialized conda with python ${pyv}"

    # install packages
    if ${conda_missing}; then
        echo
        echo "setting up conda environment"

        # conda packages (nothing so far)
        # conda install --yes ...  || return "$?"

        # pip packages
        pip install -U pip
        pip install \
            'tensorflow==2.10.*' \
            'awkward==1.10.*' \
            'ipython' \
            'uproot4' \
            matplotlib \
            numpy \
            livelossplot \
            tqdm \
            mplhep \
            scikit-learn \
        || return "$?"
    fi
}

# entry point
action "$@"
