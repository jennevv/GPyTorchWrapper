#!/bin/bash -l 
#SBATCH --clusters=wice
#SBATCH --account=lp_jenne
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=0
#SBATCH --mem-per-cpu=1000M
#SBATCH --job-name=gp-training
#SBATCH --time=01:00:00
#SBATCH --output=/user/leuven/347/vsc34721/gp_training_%j.out

Input=data/arh2p/data/processed/arh2p_xyz_ang_s0_lr.csv
Format=csv
Config=data/arh2p/docs/config_s0.yml
Output=model_arh2_s0_ang_morse_poly1xmatern52_1
Dir=/data/leuven/347/vsc34721/arh2p/

WorkDir=gp_training
Script=$VSC_DATA/GPyTorchWrapper/training_gpytorch.py

source activate GPyTorchWrapper

if [ ! -d $VSC_SCRATCH/$WorkDir ]
then 
    mkdir $VSC_SCRATCH/$WorkDir
fi

job_dir="$SLURM_JOB_ID"

mkdir $VSC_SCRATCH/"$WorkDir"/"$job_dir"

if [ -d $VSC_SCRATCH/"$WorkDir"/"$job_dir" ]; then
    echo "Job directory created: $job_dir"
else
    echo "Failed to create job directory"
    exit 1
fi

cp $Input $VSC_SCRATCH/"$WorkDir"/"$job_dir"
cp $Config $VSC_SCRATCH/"$WorkDir"/"$job_dir"

cd $VSC_SCRATCH/"$WorkDir"/"$job_dir"

python "$Script" -i "$(basename $Input)" -f "$Format" -c "$(basename $Config)" -o "$Output" -d ./

cp *.pth $Dir
