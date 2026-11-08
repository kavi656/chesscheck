# ChessCheck: 2-minute demo script

**Before the demo (≥1 h ahead):**
1. Run `python -m src.live commit` and post the printed SHA-256 publicly (Slack / a tweet / a gist). The commitment also fixes the time of the one-shot **final test** (default 3 h later; `--final-hours N`). Make sure it lands before the demo.
2. Start `python -m src.live run --minutes 90` in a terminal.
3. Open the dashboard: `streamlit run src/app.py`. Start on the **Overview** tab; the four cards at the top are the whole story.

---

### 0:00: The problem (15 s) · *tab: Overview*
> "With 11 million games, *everything* is significant. Finding patterns is easy; the hard part is knowing which ones to trust. We built an AI researcher that proves its own false-discovery rate."

### 0:15: Naive vs honest (30 s) · *tab: Naive vs honest*
> "Here's Claude acting as a naive researcher on data where we **shuffled away every real effect**. It proposes hypotheses, sees the p-values, and refines them three times."

- Null data: **Claude found 6 "discoveries" in 66 tries; our deterministic p-hacker found 131 across 20 shuffles.**
- **Honest pipeline: 0 survive.** All die at the multiple-testing correction, several with confident made-up explanations ("because Elo expectations…").

### 0:45: The funnel (25 s) · *tab: Funnel / graveyard*
> "Every hypothesis is logged **before** it's tested. It must survive Benjamini–Hochberg across the whole ledger, a minimum effect size, a fresh validation period, skeptic checks, and a one-shot vault."

- **57 proposed** (hand, template and Claude) → 33 → 19 → 16 → **14 REAL**, which is **7 distinct findings** once mirrors ("White +" / "Black −") and nested variants are merged. Say this before a judge does.
- Scroll to the **winner's curse** chart: ideas picked as best on explore come out smaller on fresh data.
- Point at the **hot hand**: p = 3e-43 in explore, then it shrank below the minimum effect **in the vault**, so it's LUCKY. Also the losing-streak hypothesis that faded at validation.
- Point at a REAL one: **tilt**. After a loss, restarting within 2 minutes costs ~1.2% of a point versus the player's own baseline (**≈ 8 rating points**), replicated in the vault.

### 1:10: Proof it's calibrated (25 s) · *tab: Calibration*
> "How do we know it works? We ran it 50 times on shuffled data."

- **False REAL rate: 0 in 2,800 null tests** (57 hypotheses × 50 shuffles; target ≤ 5%). The naive rule on the same data: 5.4%, i.e. 3 fake discoveries per run.
- **Power:** a planted effect of 0.01, twice the minimum effect, is found **100%** of the time.
- The war story, one line: "Our first test passed this check *and* was badly biased: it found tilt on pure noise at p = 10⁻²¹³ (the hot-hand-fallacy bias). We fixed the test **and** made the null harder so it would have caught it."

### 1:35: The vault is the future (25 s) · *tab: Live vault*
> "The vault was still our own data. So before this talk we committed the survivors and published this hash."

- `python -m src.live verify` gives **VERIFIED**.
- Show games played **since the commit**, with running estimates and **anytime-valid confidence sequences**, so we can peek live without inflating false positives.
- Show the **pre-registered final test**: one shot, at a time fixed in the commitment, with BH across the committed set.
- Optional finale: click the example **"Do players perform worse after midnight their local time?"** in **Ask a question**. It looked real on explore (p = 2e-9) and then died at validation, so it's LUCKY. In UTC it was a confirmed null. Then let a judge type their own question (test mode on, so it can't touch the committed set). A "no" comes back as either *not enough evidence* or *NO EFFECT (confirmed)*, meaning anything that would matter is ruled out. Claude converts it, it's logged and counted in the correction, and it gets a verdict in ~25 s.

---

## Numbers cheat-sheet
| | |
|---|---|
| Data | Lichess Oct 2026, 11.1M rated blitz games (BOTs excluded), 22.2M player-game rows |
| Splits | explore Oct 2–5 14:00 · validate → Oct 7 16:00 · vault → Oct 8 |
| Null calibration | **0 / 2,800** false REAL (57 hyps × 50 shuffles); naive p<0.05: 5.4% |
| Power | 10% / 100% / 100% / 100% at δ = 0.005 / 0.01 / 0.02 / 0.04 (min_effect = 0.005) |
| Naive on null | auto p-hacker 131 → 0 REAL; Claude 6 → 0 REAL |
| Main ledger | 57 hypotheses → 14 REAL (**7 distinct findings**), 19 LUCKY, 23 REJECTED, 1 UNDERPOWERED |
| Live commitment | 2026-11-08 08:53 UTC · 16 hypotheses · SHA-256 `1f3368e9…` · final test 11:53 UTC |
| Bias caught | plain within-player test: "tilt" +0.047 at p = 10⁻²¹³ on pure noise → fixed |
