#!/bin/bash -l

save_dest=~/Active_fluctuations/data_new/SAC_G1200_N100_R020/anneal
run_code_home=~/Active_fluctuations/Active-noise-simulations/sim_codes
analyze_code_home=~/Active_fluctuations/analysis_codes

name=SAC_anneal
seq=allA_seq.txt
top=chromosome_top.txt
finit=init_structure_G100_N100_R020.npy #sample_snap_1200.npy
ftype=type_table.csv

G=1200
kr=30.0
kb=10.0
Esoft=4.0
R0=20.0
blocksize=100
dt=0.001
#rm -r $path0
#mkdir ~/Active_fluctuations/data_new/SAC_G1200_N100_R020/
mkdir $save_dest
#mkdir $save_dest/analysis

Ti=400
Tf=200

for replica in {1..10}; do
 
sim_home=$save_dest/replica_$replica

mkdir $sim_home

cd $sim_home
cp $run_code_home/run_sims.py $sim_home
cp $run_code_home/input_files/$seq $sim_home
cp $run_code_home/input_files/$top $sim_home
cp $run_code_home/ActivePolymer.py $sim_home 

python_venv="#!/bin/bash -l
source ~/venv/containers/openmm/bin/activate
python3 run_sims.py -name $name -dt $dt -ftop $top  -fseq $seq -rep $replica -Ta 1.0 -G $G -F 0.0 -temp 400.0 -kb $kb -Esoft $Esoft -nblocks 400000 -blocksize $blocksize -R0 $R0 -kr $kr -savename SAC_anneal_400.0

python3 run_sims.py -name $name -dt $dt -ftop $top  -fseq $seq -rep $replica -Ta 1.0 -G $G -F 0.0 -temp 350.0 -kb $kb -Esoft $Esoft -nblocks 300000 -blocksize $blocksize -R0 $R0 -kr $kr -savename SAC_anneal_350.0 -finit SAC_anneal_400.0_lastFrame.npy

python3 run_sims.py -name $name -dt $dt -ftop $top  -fseq $seq -rep $replica -Ta 1.0 -G $G -F 0.0 -temp 300.0 -kb $kb -Esoft $Esoft -nblocks 200000 -blocksize $blocksize -R0 $R0 -kr $kr -savename SAC_anneal_300.0 -finit SAC_anneal_350.0_lastFrame.npy

python3 run_sims.py -name $name -dt $dt -ftop $top  -fseq $seq -rep $replica -Ta 1.0 -G $G -F 0.0 -temp 250.0 -kb $kb -Esoft $Esoft -nblocks 100000 -blocksize $blocksize -R0 $R0 -kr $kr -savename SAC_anneal_250.0 -finit SAC_anneal_300.0_lastFrame.npy

python3 run_sims.py -name $name -dt $dt -ftop $top  -fseq $seq -rep $replica -Ta 1.0 -G $G -F 0.0 -temp 200.0 -kb $kb -Esoft $Esoft -nblocks 100000 -blocksize $blocksize -R0 $R0 -kr $kr -savename SAC_anneal_200.0 -finit SAC_anneal_300.0_lastFrame.npy

"
echo "$python_venv">"python_venv.sh"
chmod u+x "python_venv.sh"

slurm_file_content="#!/bin/bash -l

#SBATCH --job-name=SAC_anneal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=10G
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=23:55:00

module load singularity

# RUN THE PYTHON SCRIPTS WITHIN THE OPENMM CONTAINER
singularity exec /public/apps/singularity/containers/AMD/rocm420/openmm_hip.rocm420.ubuntu18.sif ./python_venv.sh"

echo "$slurm_file_content">"run_sim.slurm"

sbatch "run_sim.slurm"
((ii+=1))

done

echo Total number of runs: $ii

