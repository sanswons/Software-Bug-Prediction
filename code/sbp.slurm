#!/bin/bash
#note - there can be no empty lines between #SBATCH directives.
#SBATCH --job-name=$swoonna-gpu
#SBATCH --output=output/encoder.out
#SBATCH --error=output/encoder.err
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=10    # CPU cores/threads
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sanjana_woona@yahoo.co.in
#SBATCH --partition=96x24gpu4
# The following designates you are using GPU 1 - Tesla P100 model
# Please only use 1GPU for 1 job at a time.
#SBATCH --gres=gpu:p100:1

chmod +x encoder.sh
./encoder.sh
