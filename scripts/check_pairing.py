"""Assert that episodes are paired across selectors and K.

Two invariants that MUST hold:
  1. `first` success is identical for every K (it always returns candidate 0)
  2. first@K=1 == oracle@K=1 (with one candidate there is nothing to select)

If either fails, every downstream comparison is contaminated by task-instance
noise rather than measuring the selector. Do not run the full sweep until this
passes.
"""

from __future__ import annotations

import argparse
import sys

from ttc.eval.metrics import load, success_vs_k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    a = p.parse_args()
    s = success_vs_k(load(a.jsonl))

    ok = True
    first = s.get("first", {})
    vals = {k: v["success"] for k, v in sorted(first.items())}
    print("first  success by K:", vals)
    if len(set(vals.values())) > 1:
        print("  FAIL: `first` varies with K -> episodes not paired")
        ok = False
    else:
        print("  ok: invariant to K")

    if "oracle" in s:
        f1 = first.get(1, {}).get("success")
        o1 = s["oracle"].get(1, {}).get("success")
        print(f"first@K=1={f1}  oracle@K=1={o1}")
        if f1 != o1:
            print("  FAIL: K=1 arms disagree -> episodes not paired")
            ok = False
        else:
            print("  ok: K=1 arms agree")

    print("\nPAIRING OK" if ok else "\nPAIRING BROKEN - do not run the sweep")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
