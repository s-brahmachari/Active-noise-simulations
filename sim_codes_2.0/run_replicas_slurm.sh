#!/bin/bash -l

for replica in {1..50}; do

python_venv="#!/bin/bash -l
source ~/venv/containers/openmm/bin/activate

#python run_active_MiChroM.py replica_$replica

#python analyze_trajectory.py replica_$replica/traj_chr10_0.cndb

python analyze_MiChroM_parallel.py replica_$replica/traj_chr10_0.cndb 16

#python run_active_MiChroM.py test/
"
echo "$python_venv">"job_$replica.sh"
chmod u+x "job_$replica.sh"

slurm_file_content="#!/bin/bash -l

#SBATCH --job-name=f1ta1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --tasks-per-node=1
#SBATCH --mem=20G
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

module load singularity

# RUN THE PYTHON SCRIPTS WITHIN THE OPENMM CONTAINER
singularity exec /public/apps/singularity/containers/AMD/rocm420/openmm_hip.rocm420.ubuntu18.sif ./job_$replica.sh"

echo "$slurm_file_content">"sub_$replica.slurm"
sbatch "sub_$replica.slurm"


done
