#!/bin/bash
#SBATCH --job-name=birddetector   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ran
#SBATCH --cpus-per-task=10
#SBATCH --mem=50GB
#SBATCH --time=12:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/DeepForest_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/DeepForest_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=2

source activate Zooniverse_pytorch

#comet debug
cd ~/BirdDetector/
python random_weight.py
