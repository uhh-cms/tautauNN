#!/usr/bin/env bash

fix_skim_file() {
    # check argument and file existence
    local file_name="$1"
    if [ -z "${file_name}" ]; then
        >&2 echo "file name is not specified"
        return "1"
    elif [ ! -f "${file_name}" ]; then
        >&2 echo "file ${file_name} does not exist"
        return "2"
    fi

    # temporary file name
    local fixed_file_name="$( dirname "${file_name}" )/fixed_$( basename "${file_name}" )"

    # let ROOTs TFile::Recover fix the file
    root -b -q -e "TFile fin(\"${file_name}\", \"READ\"); TTree* tin = (TTree*)fin.Get(\"HTauTauTree\"); TFile fout(\"${fixed_file_name}\", \"RECREATE\"); fout.cd(); TTree* tout = tin->CloneTree(-1, \"fast\"); tout->Write(); fout.Close(); fin.Close();" || return "$?"

    # move back the file
    mv "${fixed_file_name}" "${file_name}"
}

fix_skim_files_in_dir() {
    # check argument and directory existence
    local dir_name="$1"
    if [ -z "${dir_name}" ]; then
        >&2 echo "directory name is not specified"
        return "1"
    elif [ ! -d "${dir_name}" ]; then
        >&2 echo "directory ${dir_name} does not exist"
        return "2"
    fi

    # loop over output files in the directory and check if they are a) broken, and b) can be recovered
    local f
    local test_result
    for f in ${dir_name}/output_*.root; do
        echo "test ${f}"
        test_result="$( 2>&1 root -b -q -e "TFile fin(\"${f}\", \"READ\")" )"
        if [ ! -z "$( echo "${test_result}" | grep "trying to recover" )" ]; then
            echo "file ${f} is broken, trying to recover"
            if [ ! -z "$( echo "${test_result}" | grep "successfully recovered" )" ]; then
                # file is recoverable
                fix_skim_file "${f}"
            else
                >&2 echo "file ${f} cannot be recovered"
            fi
        fi
    done
}

fix_skim_files_in_dir "$@"
