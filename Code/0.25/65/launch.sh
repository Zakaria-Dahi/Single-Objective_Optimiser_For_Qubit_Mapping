#!/usr/bin/env bash
# Leave only one comment symbol on selected options
# Those with two commets will be ignored:
# The name to show in queue lists for this job:
#SBATCH -J T65-25P

# Number of desired cpus (can be in any node):
#SBATCH --ntasks=30

# Number of desired cpus (all in same node):
##SBATCH --cpus-per-task=120

# Amount of RAM needed for this job:
#SBATCH --mem=2gb

# The time the job will be running:
#SBATCH --time=168:00:00

# To use GPUs you have to request them:
##SBATCH --gres=gpu:1

# If you need nodes with special features uncomment the desired constraint line:
# * to request only the machines with 80 cores and 2TB of RAM
##SBATCH --constraint=bigmem
# * to request only machines with 16 cores and 64GB with InfiniBand network
##SBATCH --constraint=cal
# * to request only machines with 24 cores and Gigabit network
##SBATCH --constraint=dx
##SBATCH --constraint=ssd

# Set output and error files
##SBATCH --error=job.%J.err
##SBATCH --output=job.%J.out

# Leave one comment in following line to make an array job. Then N jobs will be launched. In each one SLURM_ARRAY_TASK_ID will take one value from 1 to 100
#SBATCH --array=1-32

# To load some software (you can show the list with 'module avail'):
# module load software


# the program to execute with its parameters:
source ~/env/bin/activate
origin=`pwd`
dir_temp=${LOCALSCRATCH}${USER}/${SLURM_JOB_ID}/trabajo_${SLURM_ARRAY_TASK_ID}
mkdir -p ${dir_temp}
cp main.py ${dir_temp}
cd ${dir_temp}
mkdir exec
time python main.py ${SLURM_ARRAY_TASK_ID}
mv result* exec
zip -r exec_65qubits_${SLURM_ARRAY_TASK_ID}.zip exec/
mv exec_65qubits_${SLURM_ARRAY_TASK_ID}.zip  $origin