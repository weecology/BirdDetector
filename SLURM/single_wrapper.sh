#Wrapper for submitting seperate SLURM runs
for DATASET in neill seabirdwatch penguins pfeifer palmyra monash mckellar USGS terns hayes
do
    sbatch submit_single.sh $1 $2 $DATASET
done

sbatch combined.sh