#!/bin/bash
#SBATCH -J epic-odise
#SBATCH --comment "odise run epic"

#SBATCH --qos level0
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --gres=gpu
#SBATCH --partition=gpu
#SBATCH --account=research
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00

#SBATCH -o "/home/lipuhao/dev/MAH/DataPreprocess/odise/outputs/slurm/%x.%j.out"
#SBATCH -e "/home/lipuhao/dev/MAH/DataPreprocess/odise/outputs/slurm/%x.%j.err"


