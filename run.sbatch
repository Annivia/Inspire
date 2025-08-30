#!/bin/bash
#SBATCH --job-name=BPT
#SBATCH --output=results/run-%J.out
#SBATCH --error=results/run-%J.err
#SBATCH --cpus-per-task=12
#SBATCH --time=0:05:00
#SBATCH --account=bfbo-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1

module load cuda/12.6.1
source ~/.bashrc
conda activate inspire-safe

NUM_GPUS=$SLURM_GPUS_ON_NODE
echo "Number of GPUs allocated: $NUM_GPUS"

python - <<'EOF'
import torch
print("Is CUDA available?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
EOF

export PYTHONPATH=/u/xzhang42/Inspire/LIBERO:$PYTHONPATH
export PYTHONPATH=/u/xzhang42/Inspire/vq_bet_official:$PYTHONPATH
export PYTHONPATH=/u/xzhang42/Inspire:$PYTHONPATH

echo "Transferring data..."
rsync -av /projects/bfbo/xzhang42/Inspire /work/nvme/bfbo/xzhang42/
echo "Data Transfer Completed..."

export PRISMATIC_DATA_ROOT=/work/nvme/bfbo/xzhang42/Inspire 

bash /u/xzhang42/Inspire/vla_scripts/eval/eval_baseline_libero90.sh

echo "Transferring results..."
rsync -av /work/nvme/bfbo/xzhang42/Inspire/results /projects/bfbo/xzhang42/Inspire
echo "Results Transfer Completed..."