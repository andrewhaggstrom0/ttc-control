#!/bin/bash
#SBATCH --job-name=ttc-degrade
#SBATCH -A engr-acad-cse5100
#SBATCH -p condo-cse5100
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --output=experiments/logs/degrade_%A_%a.out
#SBATCH --error=experiments/logs/degrade_%A_%a.err

set -eo pipefail
cd /home/compute/a.haggstrom/ttc-control
mkdir -p experiments/logs experiments/rollouts_degrade_bball

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=.
export CONDA_PREFIX=/home/compute/a.haggstrom/.conda/envs/ttc
export PATH="$CONDA_PREFIX/bin:$PATH"

# Verified ladder (median sigma error at h=4): 0.191 / 0.440 / 1.121 / ~2+.
# Degrades on data, capacity, and training length together -- subsampling alone
# did nothing, since 10% of 13k transitions still fits a 400-unit MLP.
# batch-size 64 is REQUIRED: the default 512 with drop_last yields zero batches
# on the small fracs and silently trains nothing.
FRACS=(1.0   0.2   0.05  0.03)
HIDDEN=(400  128    48    32)
EPOCHS=(60    40    25    15)
I=$SLURM_ARRAY_TASK_ID
FRAC=${FRACS[$I]}; HID=${HIDDEN[$I]}; EP=${EPOCHS[$I]}
TAG="d${I}"
PLAN_H=4

# ONE task. The dose-response must be within-task: pooling disassemble and
# stick-pull last time produced a Simpson's paradox that reversed the sign.
TASK=basketball-v3

echo "task=$TASK frac=$FRAC hidden=$HID epochs=$EP tag=$TAG started=$(date -Is)"
DYN=experiments/ckpt/dyn_${TASK}_${TAG}.pt

python -m ttc.train.train_dynamics --tasks $TASK --epochs $EP \
  --frac $FRAC --hidden $HID --batch-size 64 --out $DYN

python scripts/measure_model_error.py --task $TASK --dynamics $DYN \
  --horizon $PLAN_H | tee -a experiments/logs/degrade_error_bball.tsv

BETAS=$(python scripts/calibrate_beta.py --task "$TASK" --horizon $PLAN_H \
        --dynamics $DYN | grep 'suggested:' | sed 's/.*--betas //')
echo "betas: $BETAS"

python scripts/run_sweep.py \
  --task "$TASK" \
  --policy   experiments/ckpt/bc_${TASK}.pt \
  --dynamics $DYN \
  --value    experiments/ckpt/val_${TASK}.pt \
  --plan-horizon $PLAN_H --betas $BETAS \
  --arms first learned \
  --n-episodes 50 --k-max 64 \
  --out-dir experiments/rollouts_degrade_bball/${TAG}

echo "finished=$(date -Is)"
