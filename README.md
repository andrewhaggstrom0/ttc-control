# ttc-control

**Test-time search degrades control policies when the verifier is learned.**

Language models improve when you let them spend more compute at inference:
sample many candidates, score them, execute the best. Porting that to control is
straightforward to state — sample K candidate action chunks, rank them, execute
the winner — and several recent robotics systems report gains from doing it.

This repo evaluates that procedure to K=64 across six Meta-World manipulation
tasks and finds that it **consistently harms performance**. Pooled success falls
from 0.62 at K=1 to 0.40 at K=64, below the 0.58 obtained by ignoring the scores
and picking at random. An oracle verifier that scores candidates by branching the
true simulator rises to 0.74 on the same candidate sets, so the failure is the
verifier, not the search.

## Results

| Arm | K=1 | K=64 |
|---|---|---|
| no search | 0.62 | 0.62 |
| random selection | 0.62 | 0.58 |
| **learned verifier** | 0.62 | **0.40** |
| self-consistency (no verifier) | 0.62 | 0.61 |
| oracle (true dynamics) | 0.62 | 0.74 |

*Pooled over six tasks, 50 episodes per task per cell (300 per cell), candidates
nested across K and paired across arms by episode seed.*

**The verifier cannot find the good candidates.** Auditing 435 decision points
against true simulator rollouts: as K grows the best available candidate improves
from 12.45 to 14.50 in true 16-step return, while the learned verifier's pick
stays flat at ~12.3 — statistically indistinguishable from an average candidate
(~12.5). Normalized regret sits at 0.93–1.08 at every K.

**Harm compounds rather than striking once.** Per-decision cost is roughly 5%
relative to an average candidate. Episodes contain 31–45 decisions. That product,
not any single catastrophic choice, produces the 22-point drop — which is also
the likely reason evaluations at K≤32 on shorter tasks report gains.

**Verifier-free selection is immune.** Choosing the medoid of the candidate set
uses no learned scorer, so it has no proxy to over-optimize: 0.62 → 0.61 pooled,
and 0.78 → 0.86 on disassemble. It requires no extra model and is the practical
recommendation where verifier quality is uncertain.

**Negative results worth stating.** Penalizing ensemble disagreement does not
help across two orders of magnitude in penalty strength. Varying dynamics-model
error over a 12× range shows no systematic relationship with degradation
(underpowered: 4 rungs per task, SE ≈ 0.07).

## Method

Six Meta-World V3 tasks selected for long horizons (114–145 demo steps) and a
difficulty spread: `peg-insert-side`, `basketball`, `stick-pull`, `shelf-place`,
`disassemble`, `bin-picking`. Short tasks were excluded — at an execution stride
of 8 they afford too few decision points for per-decision effects to accumulate.

Per-task behavior-cloning policies (MLP, 3×512 hidden, Mish, tanh output) predict
16-step action chunks from the 39-dim state, trained on 100 scripted-expert
demonstrations, reaching 0.40–0.65 success. Candidates are Gaussian perturbations
(sigma=0.15) of the predicted chunk.

The learned verifier scores candidates by rolling a 5-member probabilistic
dynamics ensemble forward 4 steps, with a learned value function bootstrapping
the remainder. The horizon is calibrated against ground truth: median prediction
error is 0.34 sigma at 1 step, 1.52 at 4, and 6.91 at 16, so 4 is the longest
horizon under a 2 sigma bar.

### Exact simulator branching

The oracle arm requires that scoring a candidate leave no trace on the trunk
trajectory. Two details matter and neither is obvious:

- State is saved with `mjSTATE_INTEGRATION`, not `mjSTATE_FULLPHYSICS`. The
  latter omits `qacc_warmstart`, the constraint solver's cached initial guess; a
  differing warmstart makes the solver converge along a slightly different path.
- `mj_forward` is required after restore so observations read correct forward
  kinematics — but it overwrites `qacc_warmstart` as a side effect, so the
  warmstart is reapplied afterwards.

Mutable Python state is rolled back on every wrapper in the chain (time limits,
success latches, cached previous observations), not just `env.unwrapped`.
Restoration is verified bit-exact under repeated nested branching, which is the
stricter test — drift only accumulates across cycles.

## Layout

ttc/
envs/ env factory; SnapshotWrapper (exact save/restore)
policies/ BC chunk regressor
models/ dynamics ensemble, value function
search/ Selector interface + oracle/learned/value/consistency arms
eval/ closed-loop rollout with per-candidate score logging; metrics
train/ policy, dynamics, value training
scripts/
run_sweep.py main K sweep driver
check_headroom.py base-policy success gate (target 40-60%)
check_pairing.py asserts episodes are paired across arms and K
calibrate_beta.py measures return/disagreement scales per task
audit_selection.py selection regret against oracle scores
audit_success.py candidate picks audited against eventual success
analysis/ summary tables and paper figures
viz/ MuJoCo decision-point rendering, futures plots
slurm/ sbatch job scripts
paper/ LaTeX draft


## Reproducing

```bash
conda create -n ttc python=3.11 -y && conda activate ttc
conda env config vars set PYTHONNOUSERSITE=1 -n ttc   # see notes below
pip install -e ".[dev]"
pip install "metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git"

export MUJOCO_GL=egl
python -m pytest tests/test_snapshot.py -v      # verify exact restore FIRST

TASKS="peg-insert-side-v3 basketball-v3 stick-pull-v3 \
       shelf-place-v3 disassemble-v3 bin-picking-v3"
python -m ttc.data.collect --tasks $TASKS --n-episodes 100 --noise-std 0.0

for t in $TASKS; do
  python -m ttc.train.train_bc       --tasks $t --out experiments/ckpt/bc_$t.pt
  python -m ttc.train.train_dynamics --tasks $t --out experiments/ckpt/dyn_$t.pt
  python -m ttc.train.train_value    --tasks $t --out experiments/ckpt/val_$t.pt
done

PYTHONPATH=. python scripts/check_headroom.py   # gate: 40-60% base success
sbatch scripts/slurm/sweep.sbatch
PYTHONPATH=. python scripts/analysis/summarize.py
```

## Notes for anyone building on this

Things that cost real time, recorded so they don't have to again:

- **Offline loss says nothing about closed-loop competence.** A diffusion policy
  trained here showed smoothly declining loss to 0.128 while scoring 0-10%
  success. The cause was a noise schedule whose terminal alpha-bar was 0.37, so
  training never saw pure noise while sampling started from it. A positive
  control — plain MLP regression on an easy task, 90% success — isolated the
  sampler in minutes after two days of chasing the policy.
- **Test repeated state restoration, not single restoration.** The one-shot
  roundtrip passed while eight nested branch-and-restore cycles failed.
- **Seed the environment per episode.** Meta-World draws goal configurations at
  construction, so unseeded per-episode construction gives every arm a different
  task. `scripts/check_pairing.py` catches this by asserting the no-search arm is
  invariant to K.
- **Watch normalization floors.** Meta-World's observation has constant
  dimensions whose std is the 1e-6 clamp floor. Dividing by it inflated the
  disagreement metric by ~1e6 and made multi-step rollouts diverge to NaN.
- **Calibrate hyperparameters against measured scales.** The disagreement
  penalty beta needed values of 6-102, not the 0-5 that seemed natural; the
  sweep would have measured nothing.
- **`pip list` shows what pip installed, not what Python imports.** On shared
  clusters `~/.local` precedes the conda env on `sys.path`. Hence
  `PYTHONNOUSERSITE=1` above.
- **Don't pool tasks when testing a within-task relationship.** Doing so produced
  a Simpson's paradox that reversed the sign of a correlation.
- **Check that training actually happened.** `--frac 0.02` with `batch_size=512`
  and `drop_last=True` yields zero batches; the loss printed `nan` and the
  checkpoint saved random initialization, which then measured as a plausible
  "degraded model."

## Status

Experiments complete. Paper draft in `paper/`; four provisional citations still
need verification. Rollout JSONLs are gitignored and are not reproducible without
a day of GPU time — back them up separately.
