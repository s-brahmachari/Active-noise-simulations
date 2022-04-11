#!/bin/bash -l

save_dest=~/Active_fluctuations/data/SAC_exp_G1200_N100_R020_Es0
run_code_home=~/Active_fluctuations/Active-noise-simulations/sim_codes
analyze_code_home=~/Active_fluctuations/analysis_codes

name=SAC_exp
seq=allA_seq.txt
top=chromosome_top.txt
finit=init_structure_G1200_N100_R020.npy #sample_snap_1200.npy
ftype=type_table.csv

G=1200
kr=30.0
kb=10.0
Esoft=0.0
R0=20.0
nblocks=1000000
blocksize=100
dt=0.001
#rm -r $path0
mkdir $save_dest
mkdir $save_dest/analysis

ii=0
# for Esoft in 0; do
# mkdir $save_dest/Esoft_$Esoft

for T in 200.0; do
mkdir $save_dest/T_$T

for F in 0.0; do
#for F in 0.1 0.3 0.7 1.0 2.0 5.0; do
mkdir $save_dest/T_$T/F_$F

for Ta in 1.0; do
#for Ta in 0.01 0.1 1.0 10.0 50.0 100.0 1000.0; do
mkdir $save_dest/T_$T/F_$F/Ta_$Ta

# for kb in 10.0; do
# mkdir $save_dest/T_$T/F_$F/Ta_$Ta/kb_$kb

for replica in 1; do
 
# sim_home=$save_dest/T_$T/F_$F/Ta_$Ta/kb_$kb
sim_home=$save_dest/T_$T/F_$F/Ta_$Ta/replica_$replica

mkdir $sim_home

cd $sim_home
cp $run_code_home/run_sims.py $sim_home
cp $run_code_home/input_files/$seq $sim_home
cp $run_code_home/input_files/$top $sim_home
cp $run_code_home/input_files/$finit $sim_home
cp $run_code_home/ActivePolymer.py $sim_home 

cp $run_code_home/AnalyzeTrajectory.py $sim_home
cp $run_code_home/run_analyze.py $sim_home
cp $run_code_home/input_files/$ftype $sim_home


python_venv="#!/bin/bash -l
source ~/venv/containers/openmm/bin/activate
python3 run_sims.py -name $name -dt $dt -ftype $ftype -ftop $top  -fseq $seq -rep $replica -Ta $Ta -G $G -F $F -temp $T -kb $kb -Esoft $Esoft -nblocks $nblocks -blocksize $blocksize -R0 $R0 -kr $kr -finit $finit

python3 run_analyze.py -s $save_dest/analysis/ -VCV -gyr -RDP -MSD -rep $replica -bondlen
"
echo "$python_venv">"python_venv.sh"
chmod u+x "python_venv.sh"

slurm_file_content="#!/bin/bash -l

#SBATCH --job-name=SAC0
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
done
done
done

echo Total number of runs: $ii

