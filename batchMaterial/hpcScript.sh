#!/bin/bash
#PBS -j oe
#PBS -l nodes=18
#PBS -N test
#PBS -o $PBS_JOBID.out
#PBS -e $PBS_JOBID.err

module load python/2.7
module load scipy/0.10.0-r7012
module load numpy/1.5.1

cd $PBS_O_WORKDIR 

for i in {1..18}
do
    python hsmmHPC.py $i &
done
wait
