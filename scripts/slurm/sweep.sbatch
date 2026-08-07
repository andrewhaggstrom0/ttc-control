#!/bin/bash
#SBATCH --job-name=ttc-sweep
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=experiments/logs/sweep_%A_%a.out
#SBATCH --error=experiments/logs/sweep_%A_%a.err

set -euo pipefail
mkdir -p experiments/logs

# Headless: no display, no wasted GPU on a compositor.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false

TASKS=(reach-v3 push-v3 door-open-v3 drawer-close-v3 peg-insert-side-v3 pick-place-v3)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

source ~/.bashrc
conda activate ttc

echo "task=$TASK node=$(hostname) started=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/run_sweep.py \
  --task "$TASK" \
  --n-episodes 50 \
  --k-max 64 \
  --out-dir experiments/rollouts

python scripts/analysis/plot_k_sweep.py \
  "experiments/rollouts/${TASK}_rollouts.jsonl" \
  --out "experiments/figs/${TASK}_k_sweep.png" \
  --title "$TASK"

echo "finished=$(date -Is)"
