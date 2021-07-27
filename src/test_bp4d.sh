#!/usr/bin/env sh

set -e

#0.0 4.0 8.0 12.0 16.0 22.0

for RC in 8.0
do
	for iter in 89 90 91 92 93 94 95 96 97 98
	do
		sh writeRCInd.sh $RC
		sh writeIteration.sh $iter
		python ShollowCNN_AUModel_eval_weak.py
		echo "training AU & $VAR done\n"
	done
done
