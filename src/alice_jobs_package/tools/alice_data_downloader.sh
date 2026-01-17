#!/bin/bash
#SBATCH --job-name=AliceDownload
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgaliceai3-gpu-a100
#SBATCH --output="./results/alice_data_downloader.out"
#SBATCH --error="./results/alice_data_downloader.err"

cp $SLURM_SUBMIT_DIR

module purge
module load Miniconda3
module load NVHPC/22.7

source /net/software/v1/software/Miniconda3/4.9.2/etc/profile.d/conda.sh
conda activate torch-gpu

export XLA_FLAGS=--xla_gpu_cuda_data_dir=${SCRATCH}/.conda/envs/torch-gpu

which python 

#Should be configured by human
python ./alice_data_downloader.py --destinstion-path /net/pr2/projects/plgrid/plggalice_ai/alice_storage/data/data_14_jan_26 --site_sonar_year 2025 --data_type BOTH --to_csv --chunked_csv 10
