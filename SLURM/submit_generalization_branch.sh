#!/bin/bash

# Command line args for commit has and number of gpus
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=birddetector   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ran
#SBATCH --cpus-per-task=20
#SBATCH --mem=90GB
#SBATCH --time=12:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/DeepForest_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/DeepForest_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=$2

source activate Zooniverse_pytorch

#comet debug
#export COMET_LOGGING_FILE=comet.log
#export COMET_LOGGING_FILE_LEVEL=debug
#NCCL_DEBUG=INFO python generalization.py
cd ~/BirdDetector/
git checkout $1
python generalization.py
EOT

