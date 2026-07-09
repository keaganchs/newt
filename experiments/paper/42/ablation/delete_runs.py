#!/usr/bin/env python3
"""Delete W&B runs whose *group* matches one of a list of regexes.

These groups were produced by the buggy artifact-name path (a '/' in the wandb
group crashed the first checkpoint save), so the runs are dead and need clearing
before relaunch.

SAFE BY DEFAULT: this only lists matches. Nothing is deleted until you re-run
with --execute. Deletion via the W&B API is permanent.

Usage:
    # dry run -- just show what would be deleted (default)
    python delete_runs.py

    # actually delete
    python delete_runs.py --execute

    # override project/entity if needed
    python delete_runs.py --entity trm-dynamics --project "TRM Dynamics" --execute
"""
import argparse
import re
import sys

# Group-name patterns to delete (matched with re.fullmatch against run.group).
PATTERNS = [
    # r"abl_latent_dim_xl/newt_xl_.*ld",
    # r"abl_latent_dim/srm_.*ld_4h3l",
    # r"abl_latent_dim/trm_.*ld_4h3l",
    # r"abl_srm_truncation/srm_384ld_4h3l_trunc.*",
    # r"abl_regularization/smp_.*ld_4h3l_.*",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", default="trm-dynamics")
    p.add_argument("--project", default="TRM Dynamics")
    p.add_argument("--execute", action="store_true",
                   help="Actually delete. Without this it is a dry run.")
    p.add_argument("--delete-artifacts", action="store_true",
                   help="Also delete each run's logged artifacts (passes "
                        "delete_artifacts=True to run.delete()).")
    args = p.parse_args()

    import wandb

    compiled = [re.compile(pat) for pat in PATTERNS]

    api = wandb.Api()
    path = f"{args.entity}/{args.project}"
    print(f"[delete-runs] scanning {path!r}")
    print(f"[delete-runs] {'EXECUTE -- will DELETE' if args.execute else 'DRY RUN -- no deletions'}")
    print()

    runs = api.runs(path)  # all runs in the project

    matched = []
    for run in runs:
        group = run.group or ""
        if any(c.fullmatch(group) for c in compiled):
            matched.append(run)

    if not matched:
        print("[delete-runs] no runs matched. Nothing to do.")
        return 0

    # Group the matches for a readable summary.
    by_group: dict[str, list] = {}
    for run in matched:
        by_group.setdefault(run.group or "", []).append(run)

    for group in sorted(by_group):
        rs = by_group[group]
        print(f"  group {group!r}  ({len(rs)} run(s)):")
        for run in rs:
            print(f"      - {run.name}  [{run.id}]  state={run.state}")
    print()
    print(f"[delete-runs] total matched: {len(matched)} run(s) across "
          f"{len(by_group)} group(s).")

    if not args.execute:
        print("[delete-runs] dry run. Re-run with --execute to delete.")
        return 0

    print("[delete-runs] deleting...")
    failures = 0
    for run in matched:
        try:
            run.delete(delete_artifacts=args.delete_artifacts)
            print(f"      deleted {run.name} [{run.id}]")
        except Exception as e:  # keep going; report at the end
            failures += 1
            print(f"      FAILED {run.name} [{run.id}]: {e}", file=sys.stderr)

    deleted = len(matched) - failures
    print(f"[delete-runs] done. deleted={deleted} failed={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
