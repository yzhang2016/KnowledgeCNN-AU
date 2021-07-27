#!/usr/bin/env sh

R_C=$1
FILE="iters.py"

/bin/cat <<EOM >$FILE
iter=$R_C
EOM
