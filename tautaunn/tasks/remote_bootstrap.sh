#!/usr/bin/env bash

bootstrap_htcondor() {
    # set env variables
    export TN_REMOTE_ENV="htcondor"
    export TN_DIR="${LAW_JOB_HOME}/repo"

    # load the repo bundle
    (
        echo -e "\nfetching repository bundle {{tn_repo_bundle}} ..."
        mkdir -p "${TN_DIR}" &&
        cd "${TN_DIR}" &&
        cp "{{tn_repo_bundle}}" "repo.tgz" &&
        tar -xzf "repo.tgz" &&
        rm "repo.tgz" &&
        echo "done fetching repository bundle"
    ) || return "$?"

    # optional custom command before the setup is sourced
    {{tn_pre_setup_command}}

    # source the default repo setup
    echo -e "\nsource repository setup ..."
    source "${TN_DIR}/setup.sh" "" || return "$?"
    echo "done sourcing repository setup"

    return "0"
}

# Bootstrap function for slurm jobs.
bootstrap_slurm() {
    # set env variables
    export TN_REMOTE_ENV="slurm"
    export TN_DIR="{{tn_dir}}"
    export KRB5CCNAME="FILE:{{kerberosproxy_file}}"
    [ ! -z "{{vomsproxy_file}}" ] && export X509_USER_PROXY="{{vomsproxy_file}}"

    # optional custom command before the setup is sourced
    {{tn_pre_setup_command}}

    # source the default repo setup
    echo -e "\nsource repository setup ..."
    source "${TN_DIR}/setup.sh" "" || return "$?"
    echo "done sourcing repository setup"

    return "0"
}

# job entry point
bootstrap_{{tn_bootstrap_name}} "$@"
