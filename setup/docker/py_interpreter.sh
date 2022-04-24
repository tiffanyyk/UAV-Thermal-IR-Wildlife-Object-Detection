#!/bin/bash -l

readonly SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE}"c)")
# source ${SCRIPT_DIR}/env_variables.sh

/opt/conda/bin/python3.6 "$@"
