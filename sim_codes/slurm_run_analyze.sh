#!/bin/bash -l

save_dest=~/Active_fluctuations/data_new/RC_G1200_N100_free
run_code_home=~/Active_fluctuations/Active-noise-simulations/sim_codes
analyze_code_home=~/Active_fluctuations/analysis_codes

name=RC
seq=allA_seq.txt
top=chromosome_top.txt
#top=chr_G1200_N40_top.txt
ftype=type_table.csv

G=1200
kr=30.0
kb=10.0
Esoft=0.0
R0=50.0
nblocks=500000
blocksize=100
dt=0.001

mkdir $save_dest
mkdir $save_dest/analysis

ii=0

for T in 200.0; do
mkdir $save_dest/T_$T

#for F in 0.1 0.2 0.5 1.0 1.5 2.0 3.0 4.0; do
for F in 0.1 0.5 1.0 2.0 4.0; do
#for F in 0.0; do
mkdir $save_dest/T_$T/F_$F

#for Ta in 0.1 0.3 1.0 3.0 10.0 30.0 100.0; do

#if (( $(echo "$F==0.0"|bc -l) && $(echo "$Ta!=1.0") )) 
#then
#continue
#fi
for Ta in 0.1 1.0 10.0 30.0 100.0; do
#for Ta in 1.0; do 
sim_home=$save_dest/T_$T/F_$F/Ta_$Ta
mkdir $sim_home
cd $sim_home

python_venv="#!/bin/bash -l

for replica in {1..4} ; do
rep_home=$save_dest/T_$T/F_$F/Ta_$Ta/replica_"'$replica'"

mkdir "'$rep_home'"

cd "'$rep_home'"
cp $run_code_home/run_sims.py "'$rep_home'"
cp $run_code_home/input_files/$seq "'$rep_home'"
cp $run_code_home/input_files/$top "'$rep_home'"
cp $run_code_home/ActivePolymer.py "'$rep_home'" 

cp $run_code_home/AnalyzeTrajectory.py "'$rep_home'"
cp $run_code_home/run_analyze.py "'$rep_home'"
#cp $run_code_home/input_files/$ftype "'$rep_home'"


source ~/venv/containers/openmm/bin/activate

python3 run_sims.py -anneal -name $name -dt $dt -ftype $ftype -ftop $top  -fseq $seq -rep "'$replica'" -Ta $Ta -G $G -F $F -temp $T -kb $kb -Esoft $Esoft -nblocks $nblocks -blocksize $blocksize -R0 $R0 -outpath "'$rep_home'"/

python3 run_analyze.py -s $save_dest/analysis/ -rep "'$replica'" -gyr -VCV -MSD -relMSD -SXp -bondlen

done
"
echo "$python_venv">"python_venv.sh"
chmod u+x "python_venv.sh"

slurm_file_content="#!/bin/bash -l

#SBATCH --job-name=RC_free
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --mem=10G
#SBATCH --export=ALL
#SBATCH --gres=gpu:1
#SBATCH --time=23:55:00

module load singularity

# RUN THE PYTHON SCRIPTS WITHIN THE OPENMM CONTAINER
singularity exec /public/apps/singularity/containers/AMD/rocm420/openmm_hip.rocm420.ubuntu18.sif ./python_venv.sh"

echo "$slurm_file_content">"run_sim.slurm"

echo $F $Ta
sbatch "run_sim.slurm"
((ii+=1))

done
done
done

echo Total number of runs: $ii

