#!/bin/bash
#SBATCH --account=punim0811
#SBATCH --job-name=v_final
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-user=zhuol7@student.unimelb.edu.au
#SBATCH --mail-type=FAIL,END
module load Python/3.6.4-intel-2017.u2-GCC-6.2.0-CUDA10
module load Tensorflow/1.13.1-intel-2017.u2-GCC-6.2.0-CUDA10-Python-3.6.4-GPU
python main.py > output_vfinal.txt