#!/bin/bash
#SBATCH --job-name=Everglades   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH --time=5:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/Everglades_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/Everglades_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=4

source activate Zooniverse_pytorch

#comet debug
python everglades.py
