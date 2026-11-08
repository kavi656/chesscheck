# ChessCheck: an honest AI researcher for Lichess blitz

An LLM proposes hypotheses about Lichess rated blitz games. A pre-registered statistical pipeline tests each one and labels it **REAL**, **LUCKY** or **REJECTED**, and the system **proves its own false-discovery rate** on shuffled data.

Every design and statistics decision is in [NOTES.md](NOTES.md), with a one-line *why* for each. The demo script is in [DEMO.md](DEMO.md).

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY (and optionally LICHESS_TOKEN)
```

## Run each stage
| Stage | Command | What it produces |
|---|---|---|
| 1. Ingest | `python -m src.ingest --month 2026-10 --max-day 8` | streams the Lichess dump; rated-blitz headers to `data/raw/*.parquet` (~50 min, network-bound) |
| 1. Features + splits | `python -m src.features` | `data/processed/player_games.parquet`, `split_boundaries.json`, `bot_accounts.json` |
| 2. Tester | `python -m src.tester` | the three hand-written hypotheses on explore, plus a null p-value histogram |
| 3. Honest pipeline | `python -m src.judge` | `results/ledger_hand.jsonl`, `results/vault_access_hand.jsonl` |
| 4. Calibration | `python -m src.calibrate --seeds 50 --plant-seeds 50`<br>`python -m src.calibrate --seeds 50 --reuse-power` (null runs on the full bank) | `results/calibration.json` (false REAL rate, power curve) |
| 5. LLM proposer | `python -m src.proposer --n 20 [--fresh]`<br>`python -m src.proposer --offline --n 20` (no API key) | `results/ledger_main.jsonl`; prompts and raw responses in `results/proposer/` |
| 6. Naive vs honest | `python -m src.naive --data real --refiner llm`<br>`python -m src.naive --data null --refiner auto --seeds 20` | `results/naive_*.json` |
| Findings | `python -m src.findings` | merges REAL hypotheses into distinct findings, giving `results/findings.json` |
| Extra power curve | `python -m src.calibrate --power-only 14 --plant-seeds 20` | `results/power_14.json` (power on a rarer hypothesis) |
| 7. Dashboard | `streamlit run src/app.py` | runs from cached results; no API calls |
| 8. Live vault | `python -m src.live commit [--final-hours 3]`, then `python -m src.live run --minutes 60`, then `python -m src.live verify` | `commitment.json` + SHA-256; `results/live/` |
| 9. Ask a question | dashboard tab *Ask a question* (needs `ANTHROPIC_API_KEY`) | the question is logged in `results/ledger_main.jsonl` and counted in BH |
| Tests | `python -m pytest -q` | unit tests for features, tester, judge, calibration, naive, proposer and live |

## How a hypothesis is judged
1. **Logged first:** an append-only JSONL ledger records every proposal, including invalid and duplicate ones.
2. **Explore** (first 50% of the time): difference in player-adjusted residual, with two-way (player, game) clustered SEs.
3. **Correction:** Benjamini–Hochberg at q = 0.05 over *every hypothesis in the ledger*, plus |effect| ≥ `min_effect`.
4. **Validate** (next 30%): the same pre-registered spec must hold, in the same direction, at p < 0.05 and above min_effect.
5. **Skeptic checks:** time stability, rating-band stability, the effect surviving without the top 1% most active players, and no leaky features.
6. **Vault** (last 20%): one read per hypothesis, enforced by a `Vault` class that logs every access.
7. **Verdict:** REAL / LUCKY (with the stage that killed it) / REJECTED / UNDERPOWERED / INVALID / DUPLICATE.

## Results
| | |
|---|---|
| Data | Lichess Oct 2026, 11.1M rated blitz games (BOT and >200-games-a-day accounts excluded), 22.2M player-game rows |
| Main ledger | 57 hypotheses (18 hand, 20 template, 19 Claude) → **14 REAL (7 distinct findings)**, 19 LUCKY (12 of them TOO SMALL), 23 REJECTED, 1 UNDERPOWERED |
| Null calibration | **0 false REAL in 2,800 null tests** (57 hypotheses × 50 outcome-shuffled datasets; 0/50 runs with any REAL). Naive p<0.05 on the same nulls: 5.4% false discoveries per hypothesis (3.0 per run) |
| Power (planted tilt effect) | 10% / 100% / 100% / 100% at δ = 0.005 / 0.01 / 0.02 / 0.04 (min_effect = 0.005) |
| Naive vs honest on null data | deterministic p-hacker: 131 "discoveries" → **0** REAL; Claude with 3 refinement rounds: 6 → **0** REAL |
| Naive vs honest on real data | p-hacker 14 → 8 REAL; Claude 19 → 7 REAL (the rest LUCKY: vault failures and effects below min_effect) |

The most important lesson is in [NOTES.md § Test](NOTES.md). Our first test was a within-player permutation test, and it found "tilt" on pure noise at p = 10⁻²¹³ (the Miller–Sanjurjo hot-hand bias). Our first null couldn't see the artefact, so calibration would have passed. We fixed both.
