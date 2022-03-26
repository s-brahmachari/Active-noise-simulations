#!/bin/bash -l

#SBATCH --job-name=RCpass
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=10G
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

module load singularity

# RUN THE PYTHON SCRIPTS WITHIN THE OPENMM CONTAINER
singularity exec /public/apps/singularity/containers/AMD/rocm420/openmm_hip.rocm420.ubuntu18.sif ./python_venv.sh
