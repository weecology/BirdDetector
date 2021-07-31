#!/bin/bash

# Command line args for commit has and number of gpus
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=birddetector   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ran
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/DeepForest_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/DeepForest_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=$2

ulimit -c 0
NCCL_DEBUG=INFO

source activate Zooniverse_pytorch

cd ~/BirdDetector/
module load git
git checkout $1
python single_run.py $3
EOT

