# Design & stats decisions (one line *why* each)

## Data
- **Source:** Lichess standard rated dump `2026-10`, streamed over HTTP and decompressed on the fly. Nothing is stored except parsed headers. *Why:* the 31 GB file never has to fit on disk.
- **Only the first 8 UTC days (Oct 1–8)** are ingested. *Why:* the stream is network-bound at ~2.5 MB/s compressed (~11k games/s), so the full month would take ~3.5 h. Eight days still gives millions of blitz games and dense per-player sequences.
- **Rated blitz = `Event` starts with "Rated Blitz"** (lobby + arena games). Arena games are tagged `event_type=tournament`, because berserk and arena pairing could behave differently.
- **Headers parsed with a byte regex**, with no chess parser and no moves. *Why:* this is ~10× faster, and the brief only needs headers.
- **Elo expected score uses the 400-point logistic** even though Lichess uses Glicko-2. *Why:* this is what the spec defines. Systematic miscalibration of this formula (e.g. big favourites over-performing) would be a *genuine* finding about the formula, not luck.

- **BOT accounts are excluded** (all games involving them). The 3,000 most active accounts were checked via `POST /api/users` (title == "BOT"), giving 94 bots in the first 3 days. *Why:* bots play nonstop and can't tilt; on early data they produced 117-game streaks and 2,000+-game sessions. The list is cached in `data/processed/bot_accounts.json`.
- **Accounts with >200 blitz games in any UTC day are also excluded** (15 accounts on early data). *Why:* >13 h of nonstop 3+0 is not human play. They are almost certainly unflagged engines. A simple rule is easier to defend than a classifier.

## Features
- **Every sequence feature uses only games that started before the current one.** *Why:* this rules out look-ahead leakage by construction.
- **`minutes_since_prev_game` is start-to-start** and includes the previous game's duration. *Why:* headers have no end time.
- **`streak` counts consecutive decisive results ending at the previous game.** A draw resets it to 0.
- **`session_index` is the game's position in the current session** (1 = first), with sessions split by start-to-start gaps > 30 min.
- **`termination` (of the current game) is kept but flagged `leaky`.** *Why:* it is only known after the game ends. Any hypothesis using it fails the skeptic leakage check. `prev_termination` is the safe version.
- **`start_minute_parity`** exists only to host the deliberately silly control hypothesis.

## Splits
- **Oct 1 is burn-in only.** Its games feed sequence features of later games but are never analysed. *Why:* on day 1, prev_result/streak are artificially "none"/0 because history before the window is unknown.
- **The remaining time is split by game start time at row-count quantiles 50/30/20, floored to the hour.** *Why:* the split is by time, never random (players' form is autocorrelated), and quantiles give the requested proportions with only 7 analysable days. Exact boundaries are in `data/processed/split_boundaries.json` and copied below after ingest.
- **Actual boundaries** (UTC; 11.10M games after exclusions, 22.2M player-game rows):

  | Split | From | To | Rows | Players |
  |---|---|---|---|---|
  | burn-in | Oct 1 00:00 | Oct 2 00:00 | 2.78M | 358k |
  | explore | Oct 2 00:00 | Oct 5 14:00 | 9.61M (49%) | 572k |
  | validate | Oct 5 14:00 | Oct 7 16:00 | 5.91M (30%) | 477k |
  | vault | Oct 7 16:00 | Oct 8 24:00 | 3.91M (20%) | 414k |

- **Exclusions on the full data:** 163 accounts (121 BOT-titled, 42 with >200 games in a day), covering 71,480 games.
- **Stage-1 sanity:** mean residual = −5e-11 (zero by construction); White +0.0170, Black −0.0170.
- A consequence: weekday hypotheses can be untestable in later splits (e.g. no Sunday in the vault). They are marked UNDERPOWERED at that stage, which is honest.

## Test (revised after a bug the system itself would have hidden)
- **First version: a within-player permutation test** (shuffle treatment labels within player, effect = mean residual minus the player's own mean). **It was wrong** for any condition built from the player's own results (prev_result, streak):
  - On pure noise, with i.i.d. games and constant ratings, it reported "players do *better* after a loss" at **+0.047 (p≈10⁻²¹³)** and "anti hot hand" at **−0.089**.
  - This is the Miller & Sanjurjo (2018) hot-hand bias, a.k.a. Nickell (1981) dynamic-panel bias. The player's mean includes the very games that define the label, so after-loss rows are compared against a mean dragged down by those losses.
  - The original calibration null (permute residuals, keep labels fixed) **did not contain the artefact**, so calibration would have passed while the real-data pipeline confidently called the artefact REAL. That is the "patterns you shouldn't trust" problem in miniature.
- **The fix: outcome `y_adj` = residual − the player's mean residual over games more than K=6 positions away, in the same split** (`tester.far_baseline`).
  - Subtracting a player baseline removes between-player confounding (between-player SD of true mean residual ≈ 0.028, which is not negligible: overrated players lose more *and* keep underperforming).
  - Excluding the ±6 neighbouring games removes the mechanical dependence between the label and the baseline. *Why K=6:* every outcome-derived condition depends on at most the last |k|+1 games, so streak thresholds are capped at |5| in the schema.
  - It is unbiased if a player's games are i.i.d. around their own level. `tests/test_tester.py::test_plain_player_mean_is_biased_far_baseline_is_not` demonstrates both versions.
  - Rows from players with ≤13 games in a split have no baseline and are dropped. This costs ~25% of rows.
- **Test statistic: difference in mean `y_adj` (treated − control), with SEs two-way cluster-robust by player and by game** (Cameron–Gelbach–Miller 2011, floored at the larger one-way variance).
  - *Why player:* rows of a player are correlated.
  - *Why game:* the two rows of a game have exactly opposite residuals. For balanced comparisons like White vs Black, one-way clustering understates the SE by √2, which a unit test checks.
  - The p-value is normal, respecting `direction`. The CI is ±1.96 SE.
- **Calibration null strengthened to an "outcome shuffle":** each player's game outcomes (score, residual, termination) are re-ordered in time, and prev_result/streak/prev_termination are **recomputed** from the shuffled order. This null contains the artefact that fooled version 1, so passing it means something.
- **UNDERPOWERED** if n_treated < 500 (rows with a baseline) or treated players < 50.
- **`outcome: "score"` maps to `residual`.** *Why:* the spec allows score only with rating diff controlled, which is the residual.
- **Assumption, stated honestly:** under H0 a player's games are exchangeable in time. Genuine rating dynamics (Glicko lag/overshoot after streaks) are *real* effects under this framing and will be called REAL if they replicate. They are effects of the rating system, not necessarily psychology. The dashboard rationale says so.
- **Finding to flag:** "big favourites outperform Elo" is strongly *negative* (−0.08 on partial data). Lichess favourites score well below the 400-logistic, so the pre-registered direction `greater` makes it REJECTED, not REAL. We never flip a direction after seeing data.

## Pipeline (judge)
- **Logging comes first:** `Judge.submit` writes every raw proposal to the append-only JSONL ledger before validation or testing, including invalid and duplicate ones. A unit test checks the ordering.
- **Duplicate = same testable spec** (a hash of subset, condition, comparison, outcome and direction). The wording of the statement is ignored. *Why:* rewording a failed hypothesis must not buy a fresh test.
- **BH (q=0.05) runs over every non-underpowered explore p-value in the ledger, recomputed after every batch.** Verdicts are re-derived each time. *Why:* the spec counts all hypotheses ever tested. As the ledger grows the threshold can tighten, and an earlier survivor can be demoted to LUCKY@correction. That demotion is logged, never hidden.
- **min_effect is applied after BH on the explore effect, and again at validate and vault.** Failing at explore is recorded as LUCKY@min_effect. *Why:* by the spec's definition it "looked significant" (raw p<0.05), and a significant but tiny effect is not REAL.
- **Two-sided hypotheses:** the direction is set by the explore sign, then fixed for validate and vault.
- **UNDERPOWERED at validate or vault is its own verdict, not LUCKY.** *Why:* "no data" is not evidence of luck.
- **Skeptic checks run on explore+validate only.** The vault is never touched.
  - *Time stability:* the effect has the expected sign in both halves (split at the median start time).
  - *Subgroup stability:* the check fails only if a rating band with ≥500 treated rows shows an effect of the **opposite** sign whose 95% CI excludes 0. *Why:* requiring every small band to share the sign would fail on noise alone.
  - *Player concentration:* drop the top 1% most active players, then require the same sign and one-sided p<0.05. *Why:* "doesn't vanish" has to be operational.
  - *Leakage:* fails if any feature is flagged `leaky` in the schema (currently only `termination`). Sequence features are leak-free by construction, which unit tests check.
- **Vault:** a `Vault` object owns the vault rows. `read(id)` works once per hypothesis, every access is logged with granted/denied, and a second read raises `VaultAccessError`. The vault is read only for hypotheses that already passed validate and skeptic, and its result is never used to choose, tune or rank anything.
- **Treated/control masks are cached by spec.** In calibration, only masks of hypotheses that use no outcome-derived feature are shared across seeds; the rest are recomputed for every shuffled dataset.

## Calibration
- **Null:** the outcome shuffle described under *Test*, applied within each player across the whole analysed period, with sequence features recomputed. *Why:* it keeps each player's level and results distribution and destroys any real link between history/context and outcome, but keeps every mechanical artefact of how conditions are built.
- **Bank:** the hand-written bank plus every LLM hypothesis cached in `results/cache/llm_bank.json`. Calibration makes no API calls.
- **Planted:** effect δ is subtracted from the residual of rows matching HAND[0]'s condition (tilt) in all splits, for δ ∈ {0.005, 0.01, 0.02, 0.04}. Detection = REAL verdict.
- **Power at δ = min_effect = 0.005 is low by design.** The estimate must clear min_effect at three independent stages, so power is ≈ 0.5³ at the boundary. Power at 2× and above is the meaningful number.

- **Results (hand bank of 18, 50 null seeds + 4×50 planted seeds; `results/calibration.json`):**
  - False REAL rate per hypothesis = **0.0%**, and 0 of 50 null runs had any REAL.
  - Naive (p<0.05, no refinement) averages 1.5 false discoveries per run.
  - Power: **10% / 100% / 100% / 100%** at δ = 0.005 / 0.01 / 0.02 / 0.04.
- **Full bank (57 hypotheses: hand + template + Claude), 50 null seeds:** 0 false REAL in 2,800 tests, and 0/50 runs with any REAL. Naive p<0.05 gives 5.4% per hypothesis, close to the nominal 5%, which also confirms the test's p-values are calibrated under this null. Power was reused from the 18-hypothesis run (the planted test doesn't depend on the other hypotheses except through BH's m). Output is in `results/calibration.json`.
- **"Other REAL" in planted runs is not a false positive.** At δ = 0.04, about 4 other hypotheses per run were called REAL. They select overlapping rows (e.g. "3+ losing streak" rows are almost all "after a loss within 5 min"), so they correctly detect the planted effect through their overlap.

## Proposer
- **The LLM proposer is `claude-sonnet-5-5`, using structured JSON output (`output_config.format` with a JSON schema) and the server-side refusal fallback (`fallbacks: "default"`).** The SDK usage was checked against current docs.
- **It sees only the schema, explore-only aggregates and the ledger's existing statements.** The exact prompt and raw response of every attempt are saved in `results/proposer/`.
- **First batch: Claude returned 19 hypotheses for a request of 20, all valid on the first attempt** (8.4k input / 3.2k output tokens). We did not top up to 20: an extra batch only adds ledger entries and tightens BH.
- **Offline template proposer (`src/offline_proposer.py`)** was built first, while no API key was available. That followed the spec's rule of picking the closest sensible alternative. It sees the same inputs and covers the same themes, and its 20 hypotheses are in the main ledger too, so every proposal counts toward BH.
- **Main ledger after both batches: 57 hypotheses** (18 hand, 20 offline, 19 LLM) → 14 REAL, 19 LUCKY, 23 REJECTED, 1 UNDERPOWERED. The BH threshold at m = 56 testable is ≈ 5e-5.

## Naive mode
- **It uses the same proposer, explore only, no correction, and up to 3 refinement rounds with the explore p-values shown.**
- **Two refiners:** the LLM, as specced, and a deterministic "auto" p-hacker (rating-band subsets, threshold ×0.5/×2/×3, direction flip). *Why:* the auto refiner needs no API calls, so the naive-vs-honest contrast can run on 50 null seeds.
- **Every variant the naive process tried enters the honest ledger.** *Why:* the honest correction must pay for every fork the naive researcher took.

- **Results:**

  | Data | Refiner | Naive "discoveries" | Survive honest testing (REAL) |
  |---|---|---|---|
  | Null (20 shuffles) | auto p-hacker | 131 (6.6 per run) | **0**: 110 killed by BH, 20 by min_effect, 1 at validate |
  | Null (1 shuffle) | Claude, 3 refinement rounds | 6 from 66 tries | **0**: all killed by BH, several with confident made-up explanations |
  | Real | auto p-hacker | 14 | 8 (hot hand dies in the vault; 5 trivially small) |
  | Real | Claude, 3 refinement rounds | 19 from 29 | 7 (12 below Claude's own pre-registered min_effect) |

- **On real data, a naive discovery is often a genuine but tiny effect.** With 9.6M explore rows almost anything is "significant". So the null-data rows are the clean false-discovery comparison, and min_effect is what separates real from "real but irrelevant".

## Live vault
- **Lichess API facts were checked against the official OpenAPI spec** (github.com/lichess-org/api), not memory:
  - one request at a time;
  - on 429, wait a minute;
  - user game export is throttled to 20 games/s anonymous and 30 games/s with OAuth.
- **Anonymous `GET /api/games/user/{u}` returned 404 when tested on 2026-11-07**, while other user endpoints worked. So the export path is used only with `LICHESS_TOKEN`. Without a token we poll `GET /api/user/{u}/current-game`, which works anonymously, round-robin over the pool.
- **`/api/stream/games-by-users` was rejected.** *Why:* it only includes games where *both* players are on the list.
- **Player pool:** the 60 most active vault-period players per rating band, filtered to existing, non-closed accounts via `POST /api/users`, and fixed in the commitment before any post-commit data exists.
- **History:** 3 h of pre-commit history is fetched so each player's previous game is known. Only games starting after `committed_at` are analysed.
- **Anytime-valid interval:** a two-sided normal-mixture confidence sequence (Robbins; Howard et al. 2021) on the centred within-player permutation sum: `|S−E0S| < sqrt((V+ρ)(log(1+V/ρ) + 2 log(1/α)))`. It is asymptotically valid (Waudby-Smith et al. 2021), and a unit test shows ≥94% time-uniform coverage across 200 peeks, against >20% failure for naive repeated 95% CIs.
- **ρ is fixed at commit time** from the validate variance scaled to a planned 5,000 live rows. *Why:* a data-dependent ρ would void the guarantee.
- **Live verdicts:** *confirmed* (CS excludes 0 in the predicted direction), *contradicted* (CS excludes 0 the other way), or *inconclusive*.
- **Rate limits in practice:** back-to-back token exports for 300 players triggered repeated 429s, which cost 12 of 25 minutes. Every request now goes through one pacer: a minimum 2 s gap, doubled after any 429 (cap 8 s) after the mandated 60 s wait, and relaxed slowly afterwards. The next 8-minute run had zero 429s.
- **Pool selection by recency:** the 400 most active October players per band are filtered to accounts seen online in the last 2 h (`seenAt` from `POST /api/users`). *Why:* a month-old "most active" list gave 8 post-commit games in 10 minutes; recency-filtering gave ~6× more.
- **Throughput (token mode):** ~2,500 post-commit player-rows per 10-minute round. A White-sized effect (0.034) needs roughly 1–2 h of play before the 95% confidence sequence can exclude 0, and tilt-sized effects (~0.01) need many hours. Live verdicts during a short demo will mostly be "inconclusive", and we say so rather than loosening the interval.
- **Known limitation of polling mode:** games played between polls are missed, so `prev_result` etc. can refer to the last *observed* game. Token mode has complete histories.
- **Pre-registered one-shot final test (added 2026-11-08, before the real commit).** The commitment fixes `final_test_at` (3 h after commit). At that time we use only games started in [commit, final_test_at), take a one-sided p-value in the validated direction for each committed hypothesis, and apply BH at q = 0.05 across the committed set. Pass = BH-significant and |effect| ≥ min_effect. *Why:* the anytime-valid CS pays for unlimited peeking with wide intervals. A single test at a time fixed in advance has much more power and is still valid, because the time was chosen before any live data existed. The CS stays for watching live.
- **Real commitment:** 2026-11-08 08:53:43 UTC, 16 hypotheses (every ledger_main hypothesis that passed validate + skeptic), 300 players. SHA-256 `1f3368e95f4eb8090198ed00cbd4613e08d0c67ceeb741be140e8b2ce9c32aee`. Final test at 11:53:43 UTC.

## Reporting
- **REAL hypotheses are merged into distinct findings** (`src/findings.py`, on explore only). Same-sign pairs merge when their treated sets nest: overlap coefficient, rescaled so that chance overlap scores 0. Opposite-sign pairs merge only when one's treated rows are essentially the other's control rows (Jaccard ≥ 0.5, e.g. "White +" / "Black −"). Union-find at 0.5. Result: **14 REAL → 7 distinct findings**. *Why:* counting a mirror or nested variant as a new discovery oversells the result. A raw overlap coefficient merged everything, since any small group is ~50% White by chance, hence the rescaling.
- **"No effect (confirmed)" label:** a non-REAL hypothesis whose whole explore 95% CI sits inside ±min_effect. *Why:* separates "couldn't tell" from "any effect that matters is ruled out" (an equivalence test; the 95% CI is stricter than TOST's 90%). It is a display label only and never changes a verdict. 18 of 24 REJECTED and 7 LUCKY get it.
- **Effects are shown in rating points:** near equal ratings, d(expected)/d(Elo) = ln10/1600, so 1 score point ≈ 695 Elo and tilt's −0.012 ≈ −8 Elo. *Why:* "score points" mean nothing to a listener.
- **Winner's-curse chart:** for every hypothesis that reached validate, the explore effect is plotted against the effect re-measured on validate and vault, oriented in the predicted direction. *Why:* it shows directly that the ideas chosen as best on one dataset are overestimated on it.
- **Ask-a-question test mode** (default on) runs on `results/sandbox/ledger_ask.jsonl`, a copy of the main ledger. Questions still count toward that ledger's correction. *Why:* judges' questions during the demo must not change the main ledger or the committed live set.
- **What happens to old verdicts when the ledger grows.** BH is recomputed over every hypothesis tested so far, and any verdict that changes is appended to the ledger as a new verdict row, so nothing is overwritten silently. In practice the REALs have explore p-values around 1e-10 or smaller, far below any threshold the ledger could produce, so new questions can't revoke them. Borderline ones near the threshold can flip in either direction, and the ledger shows when. A hypothesis can only be tested in the vault once, so a late flip can never buy a second vault look. *Alternative considered:* online FDR (LOND / alpha-investing), which freezes verdicts by spending the error budget in order. We kept offline BH because it is simpler to explain and more powerful for a fixed batch; with more time, online FDR is the principled choice for an open-ended Ask box.

## Estimated local time (added 2026-11-08)
- **`local_hour` feature.** Lichess headers carry UTC only. For each player, the 6-hour UTC window with the fewest *explore* games is taken as their sleep, mapped to 01:00–06:59 local. The feature is NaN for players with fewer than 30 explore games (48% of rows). *Why:* "after midnight UTC" mixes New York evenings with Tokyo mornings, so a UTC hour can't test a late-night hypothesis.
- **Leak-safe:** it uses only game *start times*, only from explore, never outcomes or validate/vault data. It is a fixed per-player attribute. Within explore it does use the player's later start times, which is timing information, not outcome information. The shuffle null keeps all start times, so calibration is unaffected.
- **Sanity check:** the implied UTC offset is most often +1 (32% of rows), which fits Lichess's European-heavy user base.
- **First use:** "Do players perform worse after midnight, their local time?" (local 0–4 vs 10–21). Explore −0.0056 (≈ −4 Elo), p = 2e-9, n = 335k, then validate −0.0012, p = 0.15. **LUCKY at validate.** The UTC version was a confirmed null (whole CI inside ±min_effect). Run in the sandbox ledger.

## Power on a second hypothesis
- `python -m src.calibrate --power-only 14 --plant-seeds 20` plants the effect on "after losing on time, players underperform". Output is `results/power_14.json`, shown as a second curve in the Calibration tab.
- **Result:** 10% / 100% / 100% / 100% at 0.005 / 0.01 / 0.02 / 0.04, the same as tilt. We meant this as a *rare* condition, but it has 1.04M explore rows, so it does not test small samples. It shows the power result isn't specific to one hypothesis. Power for genuinely rare subgroups (tens of thousands of rows) is untested and will be lower. The UNDERPOWERED rule (treated n ≥ 500 and ≥ 50 players) is only a floor.
- **Spill-over at large effects:** at δ = 0.04, 3.25 *other* hypotheses per run were called REAL. These are not false discoveries. The planted rows also sit inside other hypotheses' treated groups (losing-streak and after-a-loss variants), so those hypotheses really do contain the planted effect. At δ ≤ 0.02 there was no spill-over.

## Dashboard
- Dark theme set in `.streamlit/config.toml`. A new Overview tab states each claim with its evidence, the findings in Elo terms, the limitations and the method sources. Each tab starts with a plain-English explainer. The Live tab verifies the hash on screen and counts down to the final test. The Ask tab shows the spec in words and the stage at which the hypothesis died. *Why:* judges should be able to follow the funnel without us in the room.
