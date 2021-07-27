#!/usr/bin/env sh

set -e

#0.0 4.0 8.0 12.0 16.0 22.0

for RC in 4.0 8.0 12.0 16.0 24.0
do
	sh writeRCInd.sh $RC
	python ShollowCNN_AUModel_weak_train.py
	echo "training annotation rate $RC done\n"
done

