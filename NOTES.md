
## Session end 2026-08-07

Next action: fix episode pairing in run_sweep.
  - env_fn() is called fresh per episode with no construction seed, so
    Meta-World's RandomTaskSelectWrapper draws a different goal each time
  - symptom: `first` success varies with K (0.4/0.5/0.2/0.3) though it
    always returns candidate 0; first@K=1 != oracle@K=1
  - fix: thread the episode index into env construction, then assert
    first@K=1 == oracle@K=1 before running anything else

Then: full sweep, all six tasks, all arms, K up to 64.

Deferred:
  - dynamics logvar pinned at -10 clamp; check disagreement isn't degenerate
  - disassemble value loss 0.086 vs basketball 0.0038
  - diffusion policy via `diffusers` DDPMScheduler as a second policy arm
