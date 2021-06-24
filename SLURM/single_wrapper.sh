#Wrapper for submitting seperate SLURM runs
#run from top dir
for DATASET in newmexico neill seabirdwatch penguins pfeifer palmyra monash mckellar USGS terns hayes
do
    sbatch SLURM/submit_single.sh $1 $2 $DATASET
done

sbatch SLURM/combined.sbatch