"""Stage ``report``: assemble the results tables the manuscript needs.

Reads ``results/accuracy_by_pair.json`` and (if present)
``results/association_z.json``; writes ``results/summary.md`` and prints a
compact terminal digest. Figures come from ``notebooks/02_figures.ipynb``.
"""

from __future__ import annotations

import json

from .config import Config


def _fmt_pair(r: dict) -> str:
    lo, hi = r.get("boot_ci95", [float("nan")] * 2)
    star = " *" if r.get("q_fdr", 1) < 0.05 else ""
    return (
        f"| {r['source']} → {r['target']} | {r['acc_mean']:.3f} "
        f"[{lo:.3f}, {hi:.3f}] | {r['acc_clean_mean']:.3f} | {r['null_mean']:.3f} "
        f"| {r['p_perm']:.4f} | {r.get('q_fdr', float('nan')):.4f}{star} | "
        f"{r['mean_rank']:.0f} | {r['n_collision_total']} |"
    )


def run(cfg: Config) -> dict:
    rdir = cfg.paths.resolve("results")
    acc = json.loads((rdir / "accuracy_by_pair.json").read_text(encoding="utf-8"))
    assoc_path = rdir / "association_z.json"
    assoc = json.loads(assoc_path.read_text(encoding="utf-8")) if assoc_path.exists() else None

    base_path = rdir / "baselines.json"
    base_rows = json.loads(base_path.read_text(encoding="utf-8"))["rows"] if base_path.exists() else []
    base = {(r["baseline"], r["source"], r["target"]): r for r in base_rows}

    lines: list[str] = []
    lines.append(f"# Results — config `{acc['config']}` (`{acc['config_fingerprint']}`)\n")
    lines.append(
        f"map={acc['map']}, k={acc['k']}, folds={acc['folds']}, "
        f"null_iters={acc['null_iters']}, bootstrap_iters={acc['bootstrap_iters']}\n"
    )

    for fam in ("confirmatory", "exploratory"):
        pairs = sorted(
            (p for p in acc["pairs"] if p["family"] == fam),
            key=lambda x: (x["p_perm"], -x["acc_mean"]),
        )
        if not pairs:
            continue
        lines.append(f"\n## {fam.capitalize()} — retrieval (BH-FDR within this family)\n")
        lines.append(
            "| pair | acc@k [95% CI] | acc clean | null | p_perm | q_FDR | mean rank | collisions |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        lines.extend(_fmt_pair(r) for r in pairs)
        n_sig = sum(1 for r in pairs if r.get("q_fdr", 1) < 0.05)
        lines.append(f"\n{n_sig}/{len(pairs)} pairs significant at q<0.05.\n")

    if base_rows:
        kinds = sorted({r["baseline"] for r in base_rows})
        lines.append(f"\n## Baseline comparison — phonetic vs {', '.join(kinds)}\n")
        lines.append("| pair | phonetic acc | " + " | ".join(f"{k} acc" for k in kinds) + " | null |")
        lines.append("|---|---|" + "---|" * len(kinds) + "---|")
        for p in sorted((x for x in acc["pairs"]), key=lambda x: -x["acc_mean"]):
            cells = []
            for k in kinds:
                b = base.get((k, p["source"], p["target"]))
                cells.append(f"{b['acc_mean']:.3f}" if b else "—")
            lines.append(
                f"| {p['source']} → {p['target']} | {p['acc_mean']:.3f} | "
                + " | ".join(cells) + f" | {p['null_mean']:.3f} |"
            )
        lines.append(
            "\nphonetic − editdist is the retrieval accuracy not explained by raw "
            "orthographic string overlap (cognates / borrowing).\n"
        )

    mant_path = rdir / "mantel.json"
    if mant_path.exists():
        mant = json.loads(mant_path.read_text(encoding="utf-8"))
        lines.append(f"\n## Form–meaning correlation (Mantel, n={mant['n_subsample']} concepts)\n")
        lines.append("| analysis | unit | r | p | r \\| orthography | p (partial) | note |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in mant["rows"]:
            note = "orth control degenerate" if r.get("orth_control_degenerate") else ""
            s1 = "*" if r["p_perm"] < 0.05 else ""
            s2 = "*" if r["p_partial"] < 0.05 else ""
            lines.append(
                f"| {r['analysis']} | {r['unit']} | {r['r']:+.4f}{s1} | {r['p_perm']:.4f} "
                f"| {r['r_partial_orth']:+.4f}{s2} | {r['p_partial']:.4f} | {note} |"
            )
        lines.append(
            "\nA within-language *form~meaning* r that stays significant in the "
            "*| orthography* column is sound–meaning systematicity not attributable "
            "to spelling / cognate overlap.\n"
        )

    if assoc:
        lines.append(f"\n## Phoneme–meaning association ({assoc['n_poles']} poles, "
                     f"null_iters={assoc['null_iters']})\n")
        lines.append(f"Significant phoneme×pole cells (q<0.10): "
                     f"**{assoc['n_significant_total']}** total.\n")
        sig = [c for c in assoc["cells"] if c["significant"]][:40]
        if sig:
            lines.append("| language | pole | phoneme | z | n | q_FDR |")
            lines.append("|---|---|---|---|---|---|")
            for c in sig:
                lines.append(
                    f"| {c['language']} | {c['pole']} | /{c['phoneme']}/ | "
                    f"{c['z']:+.2f} | {c['count']} | {c['q_fdr']:.4f} |"
                )

    (rdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (cfg.paths.resolve("interim") / "report.done").write_text(
        cfg.fingerprint(), encoding="utf-8"
    )

    digest = {
        "stage": "report",
        "confirmatory_significant": sum(
            1 for p in acc["pairs"] if p["family"] == "confirmatory" and p.get("q_fdr", 1) < 0.05
        ),
        "exploratory_significant": sum(
            1 for p in acc["pairs"] if p["family"] == "exploratory" and p.get("q_fdr", 1) < 0.05
        ),
        "association_significant": assoc["n_significant_total"] if assoc else None,
        "summary_md": str(rdir / "summary.md"),
    }
    print(json.dumps(digest, indent=2))
    return digest
