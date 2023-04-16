#!/bin/bash
#SBATCH -J epic-odise
#SBATCH --comment "odise run epic"

#SBATCH --qos level1
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --gres=gpu
#SBATCH --partition=gpu
# SBATCH -w lambda-hyperplane01
#SBATCH --account=research
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00

#SBATCH -o "/home/lipuhao/dev/MAH/DataPreprocess/ODISE/outputs/slurm/%x.%j.out"
#SBATCH -e "/home/lipuhao/dev/MAH/DataPreprocess/ODISE/outputs/slurm/%x.%j.err"

echo "running slurm file"
# python scripts/helloworld.py
# python demo/demo_seq.py 
# python demo/demo.py --input demo/examples/coco.jpg --output demo/coco_pred.jpg --vocab "black pickup truck, pickup truck; blue sky, sky"
python demo/demo_seq.py --batch_size 8 --basedir /scratch/generalvision/EgoCentric/EPIC-KITCHEN --part P01 --clip P01_01 
echo "exit python file" 