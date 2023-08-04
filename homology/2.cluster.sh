# !/bin/bash

# Author: Shijie Xu
# Date: 2022-12-17
# Last modified: 2023-01-23
# Description: This script is used to create a cluster on metalpdb database.

# set variables
# in_path="tmp/in.txt"
# db90="tmp/db90.txt"

# since it is small, we do not need a step-by-step clustering
in="tmp/in.txt"
db25="tmp/db25.txt"

# cd-hit with 90% identity, coverage 0.3
# cd-hit -i $in_path -o $db90 -c 0.9 -n 5 -aS 0.3 -aL 0.3 -g 1 -G 0 -d 0 -p 1 -T 16 -M 0
# # cd-hit with 60% identity, coverage 0.3
# cd-hit -i $db90 -o $db60 -c 0.6 -n 4 -aS 0.3 -aL 0.3 -g 1 -G 0 -d 0 -p 1 -T 16 -M 0
# psi-cd-hit with 25% identity, coverage 0.3
# https://github.com/weizhongli/cdhit/blob/master/psi-cd-hit/README.psi-cd-hit
./cdhit/psi-cd-hit/psi-cd-hit.pl -i $in -o $db25 -c 0.25 -para 8 -blp 4 -ce 1e-6 -g 1 