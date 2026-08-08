#!/bin/bash
#SBATCH --job-name=ttc-sweep
#SBATCH -A engr-acad-cse5100
#SBATCH -p condo-cse5100
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=experiments/logs/sweep_%A_%a.out
#SBATCH --error=experiments/logs/sweep_%A_%a.err

set -euo pipefail
mkdir -p experiments/logs experiments/rollouts experiments/figs

# Headless rendering; block ~/.local from shadowing the conda env.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=.

TASKS=(peg-insert-side-v3 basketball-v3 stick-pull-v3 \
       shelf-place-v3 disassemble-v3 bin-picking-v3)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

# Planning horizon: past ~4 steps the dynamics model drifts >2 sigma, so the
# learned arm plans 4 and the value function bootstraps the tail.
PLAN_H=4

source ~/.bashrc
conda activate ttc

echo "task=$TASK node=$(hostname) started=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Beta scales with each task's return spread, so calibrate per task rather than
# reusing one task's numbers. Costs ~2 min; prevents a sweep that measures
# nothing because the penalty was mis-scaled.
BETAS=$(python scripts/calibrate_beta.py --task "$TASK" --horizon $PLAN_H \
        | grep 'suggested:' | sed 's/.*--betas //')
echo "calibrated betas: $BETAS"

python scripts/run_sweep.py \
  --task "$TASK" \
  --policy   experiments/ckpt/bc_${TASK}.pt \
  --dynamics experiments/ckpt/dyn_${TASK}.pt \
  --value    experiments/ckpt/val_${TASK}.pt \
  --plan-horizon $PLAN_H \
  --betas $BETAS \
  --n-episodes 50 \
  --k-max 64 \
  --out-dir experiments/rollouts

python scripts/analysis/plot_k_sweep.py \
  "experiments/rollouts/${TASK}_rollouts.jsonl" \
  --out "experiments/figs/${TASK}_k_sweep.png" \
  --title "$TASK"

echo "finished=$(date -Is)"
