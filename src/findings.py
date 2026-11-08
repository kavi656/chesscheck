"""Group REAL hypotheses into distinct findings.

Several REAL hypotheses can be the same fact said twice ("White outperforms" / "Black underperforms") or
nested variants ("lost, back within 5 min" / "lost, back within 2 min"). Counting them as separate
discoveries would oversell the result, so we merge them:

  same effect sign (nested variants): overlap = |T_A ∩ T_B| / min(|T_A|, |T_B|), rescaled as
      (overlap - chance) / (1 - chance), chance = max(|T_A|, |T_B|) / N (what independent sets would give).
  opposite sign (mirror claims): Jaccard(T_A, C_B), i.e. A's treated rows are essentially B's control rows.
  (T = treated rows, C = control rows.)
  A and B are the same finding if similarity >= 0.5 (union-find, so the relation is transitive).

Computed on the explore split only. Writes results/findings.json.
Usage: python -m src.findings
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.hypothesis import groups, validate
from src.judge import load_data
from src.ledger import Ledger

OUT = Path("results/findings.json")
THRESHOLD = 0.5


def similarity(ta, ca, sa, tb, cb, sb) -> float:
    """Overlap coefficient, rescaled so chance overlap (independent sets) scores 0 and nesting scores 1."""
    if sa != sb:
        # mirror claim ("White +" vs "Black -"): A's treated rows must be essentially B's control rows
        union = (ta | cb).sum()
        return float((ta & cb).sum() / union) if union else 0.0
    gb = tb
    na, ng, n = ta.sum(), gb.sum(), len(ta)
    if not na or not ng:
        return 0.0
    overlap = (ta & gb).sum() / min(na, ng)
    chance = max(na, ng) / n
    return float((overlap - chance) / (1 - chance)) if chance < 1 else 0.0


def cluster(ids, masks, signs, threshold=THRESHOLD):
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            s = max(similarity(*masks[a], signs[a], *masks[b], signs[b]),
                    similarity(*masks[b], signs[b], *masks[a], signs[a]))
            pairs.append((a, b, round(s, 3)))
            if s >= threshold:
                parent[find(a)] = find(b)
    groups_: dict[str, list[str]] = {}
    for i in ids:
        groups_.setdefault(find(i), []).append(i)
    return sorted(groups_.values(), key=lambda g: (-len(g), g[0])), pairs


def main(ledger: str = "results/ledger_main.jsonl") -> None:
    st = Ledger(ledger).state()
    real = {k: s for k, s in st.items() if s["verdict"] == "REAL"}
    df = load_data()
    ex = df[df["split"] == "explore"].reset_index(drop=True)
    masks, signs, info = {}, {}, {}
    for hid, s in real.items():
        h = validate({**s["raw"], "id": hid})
        masks[hid] = groups(ex, h)
        signs[hid] = int(np.sign(s["stages"]["validate"]["effect"]))
        info[hid] = {"statement": h.statement, "vault_effect": s["stages"]["vault"]["effect"],
                     "n_treated_explore": int(masks[hid][0].sum())}
    ids = sorted(real)
    clusters, pairs = cluster(ids, masks, signs)
    findings = []
    for g in clusters:
        lead = max(g, key=lambda i: info[i]["n_treated_explore"])  # broadest member names the finding
        findings.append({"lead": lead, "statement": info[lead]["statement"], "members": g,
                         "member_statements": {i: info[i]["statement"] for i in g}})
    out = {"n_real": len(ids), "n_findings": len(findings), "threshold": THRESHOLD,
           "findings": findings, "pair_similarity": [p for p in pairs if p[2] > 0.2]}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"{len(ids)} REAL hypotheses -> {len(findings)} distinct findings")
    for f in findings:
        print(f"  [{len(f['members'])}] {f['statement'][:80]}  ({', '.join(f['members'])})")


if __name__ == "__main__":
    main()
