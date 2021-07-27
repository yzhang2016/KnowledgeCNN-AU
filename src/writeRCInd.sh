#!/usr/bin/env sh

R_C=$1
FILE="variables.py"

/bin/cat <<EOM >$FILE
RC=$R_C
EOM
