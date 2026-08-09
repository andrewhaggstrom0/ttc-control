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

# NOTE: no `set -u`. WashU's /etc/bashrc reads BASHRCSOURCED without a default,
# so `set -u` plus any bashrc sourcing kills the job before it starts.
set -eo pipefail

cd /home/compute/a.haggstrom/ttc-control
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

# Skip `conda activate` entirely: conda here is a system install whose base is
# /usr, with no usable profile.d script, and activation is unnecessary for a
# batch job. Putting the env's bin on PATH makes `python` the env interpreter.
export CONDA_PREFIX=/home/compute/a.haggstrom/.conda/envs/ttc
export PATH="$CONDA_PREFIX/bin:$PATH"


echo "task=$TASK node=$(hostname) started=$(date -Is)"
echo "python: $(which python)"
python -c "import torch; print('cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Beta scales with each task's return spread, so calibrate per task rather than
# reusing one task's numbers.
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
