#!/bin/bash
#SBATCH --job-name=mGPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --partition=XXX
#SBATCH --account=XXX
#SBATCH --output="./logs/alice_config_generator.out"
#SBATCH --error="./logs/alice_config_generator.err"

cd $SLURM_SUBMIT_DIR

### GETING IP AND PORT FROM SLURM CONFIG
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
ips=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
ips_array=($ips)
head_node_ip=${ips_array[0]}
rdvz_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
###

### SETTING DISTRIBUTED ARGS
DISTRIBUTED_ARGS="
    --nnodes $SLURM_NNODES \
    --nproc_per_node $SLURM_GPUS_ON_NODE \
    --rdzv_endpoint $head_node_ip:$rdvz_port 
    --rdzv_id $SLURM_JOB_ID 
    --rdzv-backend c10d
"
###

### ENV SETUP
module purge
module load Miniconda3
module load NVHPC/22.7

source /net/software/v1/software/Miniconda3/4.9.2/etc/profile.d/conda.sh
conda activate torch-gpu

export XLA_FLAGS=--xla_gpu_cuda_data_dir=${SCRATCH}/.conda/envs/torch-gpu

which python 
###

srun torchrun $DISTRIBUTED_ARGS ./alice_torchrun.py \
--training_args_mode FILE \
--training_args_path training_config.json