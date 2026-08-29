"""``nguasach`` command-line entry point.

    nguasach run <stage> --config configs/xxx.yaml
    nguasach all         --config configs/xxx.yaml
    nguasach show        --config configs/xxx.yaml     # print result summary

Stages: data, semantics, ipa, phonetics, align  (associate, report to come).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# This box's locale is GBK; several third-party libs (panphon) open data files
# without an explicit encoding and crash. Re-exec once in UTF-8 mode.
if sys.flags.utf8_mode == 0 and os.environ.get("NGUASACH_UTF8_REEXEC") != "1":
    os.environ["NGUASACH_UTF8_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable, "-X", "utf8", "-m", "nguasach.cli", *sys.argv[1:]])

from .config import Config  # noqa: E402

STAGES = ["data", "translate-qc", "ipa", "phonetics", "semantics",
          "align", "mantel", "associate", "baselines", "report"]
_MODULE = {"align": "crossval", "associate": "association", "translate-qc": "translate_qc"}
_TAKES_JOBS = {"align", "mantel", "associate", "baselines"}


def _import(stage: str):
    from importlib import import_module

    return import_module(f".{_MODULE.get(stage, stage)}", "nguasach")


def _run_stage(stage: str, cfg: Config, n_jobs: int) -> dict:
    mod = _import(stage)
    t = time.time()
    rep = mod.run(cfg, n_jobs=n_jobs) if stage in _TAKES_JOBS else mod.run(cfg)
    rep["_seconds"] = round(time.time() - t, 1)
    return rep


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nguasach")
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("run", "all", "show"):
        sp = sub.add_parser(cmd)
        if cmd == "run":
            sp.add_argument("stage", choices=STAGES)
        sp.add_argument("--config", default="configs/default.yaml")
        sp.add_argument("--jobs", type=int, default=1)

    args = p.parse_args(argv)
    cfg = Config.load(args.config)
    print(f"[nguasach] config={cfg.name} fingerprint={cfg.fingerprint()}", file=sys.stderr)

    if args.cmd == "show":
        path = cfg.paths.resolve("results") / "accuracy_by_pair.json"
        if not path.exists():
            print("no results yet; run `nguasach all` first", file=sys.stderr)
            return 1
        data = json.loads(path.read_text(encoding="utf-8"))
        for r in sorted(data["pairs"], key=lambda x: x["p_perm"]):
            print(f"{r['family']:12} {r['source']:>10} -> {r['target']:<10} "
                  f"acc={r['acc_mean']:.3f} null={r['null_mean']:.3f} "
                  f"p={r['p_perm']:.4f} q={r.get('q_fdr', float('nan')):.4f}")
        return 0

    stages = STAGES if args.cmd == "all" else [args.stage]
    for stage in stages:
        rep = _run_stage(stage, cfg, args.jobs)
        print(f"[{stage}] {rep['_seconds']}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
