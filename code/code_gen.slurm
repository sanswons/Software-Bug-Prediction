#!/bin/bash
#note - there can be no empty lines between #SBATCH directives.
#SBATCH --job-name=$swoonna-gpu
#SBATCH --output=output/decoder.out
#SBATCH --error=output/decoder.err
#SBATCH --nodes=1
# allocate 5GB or RAM on node. You must declare --mem in all scripts
#SBATCH --mem=5GB
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sanjana_woona@yahoo.co.in
#SBATCH --partition=96x24gpu4
# The following designates you are using GPU 1 - Tesla P100 model
# Please only use 1GPU for 1 job at a time.
#SBATCH --gres=gpu:p100:1

chmod +x sampling.sh
./sampling.sh
