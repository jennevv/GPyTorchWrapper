#!/bin/bash

# HELP
Help()
{
	# Display HELP
	echo "Usage: gp_training [OPTIONS]"
	echo "options:"
	echo "-h        Print this help text"
	echo "-i        Input/training data"
	echo "-f        Format of the input data"
	echo "          DEFAULT=csv"
	echo "-c        Configuration file specifying training/testing options"
	echo "-o        Name for the model file. Should end in .pth"
	echo "          DEFAULT=model.pth"
	echo "-d        Directory where the model file will be saved"
	echo "          DEFAULT=./"
	echo "-w        Walltime for the calculation"
	echo "          DEFAULT=01:00:00"
	echo "-m        Amount of memory per cpu"
	echo "          DEFAULT=1000M"
	echo "-n        Number of cores per task"
	echo "          DEFAULT=1"
	echo "-g        Number of gpus per node"
	echo "          DEFAULT=0"
	echo "-t        Number of tasks - default=1"
	echo "          default=1"
	echo "-l        Cluster where you want to run the calculation"
	echo "          default=wice"
	echo "-p        Partition"
	echo "          DEFAULT=batch"
	echo "-a        Account from which to pull credits"
	echo "          DEFAULT=lp_jenne"
	echo "-j        Job name"
	echo "          DEFAULT=gp-training"
	echo
}

# Main Program
## Set the submission script
SubScript=$VSC_DATA/scripts/sub_gp_training.sh

## Set the default values
Format=csv
Output=model
Dir=./
Time=01:00:00
Mem=1000M
Cores=1
GPUS=0
Tasks=1
Cluster=wice
Partition=batch
Acc=lp_jenne
Job=gp-training

## Get the options from the command line
while getopts ":hi:f:c:o:d:w:m:n:g:t:l:p:a:j:" option; do
	case $option in
		h) # display help
		   Help
		   exit;;
		i) # input data
		   Input=$OPTARG;;
		f) # format input data 
		   Format=$OPTARG;;
		c) # configuration file 
		   Config=$OPTARG;;
		o) # output file 
		   Output=$OPTARG;;
		d) # output directory
		   Dir=$OPTARG;;
		w) # walltime
		   Time=$OPTARG;;
		m) # memory per cpu
		   Mem=$OPTARG;;
		n) # cores
		   Cores=$OPTARG;;
		g) # gpus
		   GPUS=$OPTARG;;
		t) # tasks
		   Tasks=$OPTARG;;
		l) # cluster/location
		   Cluster=$OPTARG;;
		p) # partition
		   Partition=$OPTARG;;
		a) # account
		   Acc=$OPTARG;;
		j) # job name
		   Job=$OPTARG;;
	       \?) # Invalid option
		   echo "Error: Invalid option"
		   exit;;
	esac
done

## Check if the Input and Config variables are specified
if [ -z "$Input" ]; 
then
	echo "Error: The input data file is not specified."
	exit
elif [ -z "$Config" ];
then
	echo "Error: The config file is not specified."
	exit
fi

## Change the variables in the submit script
sed -i "s|^\(#SBATCH --clusters=\).*|\1$Cluster|" $SubScript
sed -i "s|^\(#SBATCH --partition=\).*|\1$Partition|" $SubScript
sed -i "s|^\(#SBATCH --account=\).*|\1$Acc|" $SubScript
sed -i "s|^\(#SBATCH --ntasks=\).*|\1$Tasks|" $SubScript
sed -i "s|^\(#SBATCH --cpus-per-task=\).*|\1$Cores|" $SubScript
sed -i "s|^\(#SBATCH --gpus-per-node=\).*|\1$GPUS|" $SubScript
sed -i "s|^\(#SBATCH --mem-per-cpu=\).*|\1$Mem|" $SubScript
sed -i "s|^\(#SBATCH --job-name=\).*|\1$Job|" $SubScript
sed -i "s|^\(#SBATCH --time=\).*|\1$Time|" $SubScript
sed -i "s|^\(Input=\).*|\1$Input|" $SubScript
sed -i "s|^\(Format=\).*|\1$Format|" $SubScript
sed -i "s|^\(Config=\).*|\1$Config|" $SubScript
sed -i "s|^\(Output=\).*|\1$Output|" $SubScript
sed -i "s|^\(Dir=\).*|\1$Dir|" $SubScript

echo "TASKS=$Tasks"
echo "CORES=$Cores"
echo "MEM=$Mem" 
echo "TIME=$Time"
echo

## Run the submit script
sbatch "$SubScript"
