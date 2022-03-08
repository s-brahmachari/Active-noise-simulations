#!/bin/bash -l

save_dest=~/Active_fluctuations/data/ROUSE_chain_confined_PD
run_code_home=~/Active_fluctuations/Active-noise-simulations/sim_codes
analyze_code_home=~/Active_fluctuations/analysis_codes

#script=run_sims_dumbbell.py
run_script=run_sims.py
# analyze_script=Rouse_mode_correlations.py
analyze_script=RDP_RG_MSD_polymer.py

name=Rouse_chain
seq=chr_seq.txt
top=chromosome_top.txt
finit=None #snap800AB.npy

Na=1200
kr=30
# kb=10.0
Esoft=0
nblocks=1020000
blocksize=100
dt=0.001
#rm -r $path0
mkdir $save_dest
mkdir $save_dest/analysis

ii=0
# for Esoft in 0; do
# mkdir $save_dest/Esoft_$Esoft

for T in 100.0 150.0 200.0 250.0 300.0; do
mkdir $save_dest/T_$T

#0.02 0.2 2.0
for F in 0.01 0.1 1.0 10.0; do
rm -r $save_dest/T_$T/F_$F
mkdir $save_dest/T_$T/F_$F

#0.1 2.0 20.0 200.0
for Ta in 1.0 10.0 100.0 1000.0; do
mkdir $save_dest/T_$T/F_$F/Ta_$Ta

#10.0
for R0 in 500.0; do
mkdir $save_dest/T_$T/F_$F/Ta_$Ta/R0_$R0

for kb in 10.0; do
mkdir $save_dest/T_$T/F_$F/Ta_$Ta/R0_$R0/kb_$kb

for replica in 1; do 
sim_home=$save_dest/T_$T/F_$F/Ta_$Ta/R0_$R0/kb_$kb/replica_$replica

mkdir $sim_home

cd $sim_home
cp $run_code_home/$run_script $sim_home
cp $run_code_home/input_files/$seq $sim_home
cp $run_code_home/input_files/$top $sim_home
cp $run_code_home/ActivePolymer.py $sim_home 

# cp $analyze_code_home/$analyze_script $sim_home

# cp $code_home/type_table.csv ./
# cp $code_home/$finit ./

python_venv="#!/bin/bash -l
source ~/venv/containers/openmm/bin/activate
python3 $run_script -name $name -dt $dt -ftop $top  -fseq $seq -rep $replica -Ta $Ta -Na $Na -F $F -temp $T -kb $kb -Esoft $Esoft -nblocks $nblocks -blocksize $blocksize -R0 $R0 -finit $finit -kr $kr

# python3 $analyze_script -top $top -s $save_dest/analysis/ -if_RDP 1
"
echo "$python_venv">"python_venv.sh"
chmod u+x "python_venv.sh"

slurm_file_content="#!/bin/bash -l

#SBATCH --job-name=RCact
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=20G
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

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
done
done

echo Total number of runs: $ii
