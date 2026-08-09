#!/bin/bash
#SBATCH --job-name=ttc-frac
#SBATCH -A engr-acad-cse5100
#SBATCH -p condo-cse5100
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=experiments/logs/frac_%A_%a.out
#SBATCH --error=experiments/logs/frac_%A_%a.err

set -eo pipefail
cd /home/compute/a.haggstrom/ttc-control
mkdir -p experiments/logs experiments/rollouts_frac experiments/figs

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=.
export CONDA_PREFIX=/home/compute/a.haggstrom/.conda/envs/ttc
export PATH="$CONDA_PREFIX/bin:$PATH"

FRACS=(0.1 0.3 1.0)
FRAC=${FRACS[$SLURM_ARRAY_TASK_ID]}
PLAN_H=4

# Two tasks only. The dose-response needs three model-quality points per task,
# so six tasks would be 18 sweeps; two tasks with clear signal is enough, and
# these are the two where the effect was largest and cleanest.
for TASK in disassemble-v3 stick-pull-v3; do
  echo "=== task=$TASK frac=$FRAC started=$(date -Is)"
  DYN=experiments/ckpt/dyn_${TASK}_frac${FRAC}.pt

  python -m ttc.train.train_dynamics --tasks $TASK --epochs 100 \
    --frac $FRAC --out $DYN

  python scripts/measure_model_error.py --task $TASK --dynamics $DYN \
    --horizon $PLAN_H | tee -a experiments/logs/model_error.tsv

  BETAS=$(python scripts/calibrate_beta.py --task "$TASK" --horizon $PLAN_H \
          --dynamics $DYN | grep 'suggested:' | sed 's/.*--betas //')

  python scripts/run_sweep.py \
    --task "$TASK" \
    --policy   experiments/ckpt/bc_${TASK}.pt \
    --dynamics $DYN \
    --value    experiments/ckpt/val_${TASK}.pt \
    --plan-horizon $PLAN_H \
    --betas $BETAS \
    --arms first learned \
    --n-episodes 50 --k-max 64 \
    --out-dir experiments/rollouts_frac/frac${FRAC}
done
echo "finished=$(date -Is)"
