## Inspiration
With millions of data points, *everything* looks significant. Test enough ideas at $p < 0.05$ and about 1 in 20 will "work" by pure luck. An AI that proposes ideas, peeks at the results and tweaks them will always find something. The brief asked for a researcher that can tell a real discovery from a lucky one, so we built one that has to **prove** how often it fools itself.

## What it does
**ChessCheck** is an AI researcher for 11 million Lichess blitz games. Claude proposes hypotheses (tilt, hot hand, time of day, colour, openings…) and a pre-registered pipeline labels each one **REAL**, **LUCKY** or **REJECTED**:

1. **Logged first.** Every idea goes into an append-only ledger *before* it is tested.
2. **Explore.** The first test runs on the first 50% of the days.
3. **Correction.** Benjamini–Hochberg at $q = 0.05$ across *every* idea ever tried, so the bar rises as more ideas are tested: $p_{(k)} \le \frac{k}{m}q$. The effect must also clear a minimum size fixed in advance.
4. **Validate.** The identical spec is re-tested on later days it has never seen.
5. **Skeptic checks.** Same sign in both halves of time and across rating bands, still there without the most active 1% of players, and no leaky features.
6. **Vault.** The final 20% of the data, readable **once** per hypothesis (enforced in code and logged).

Then it **bets on the future**: surviving hypotheses are written to a file whose SHA-256 is published, then tested on games played *after* the commit, with anytime-valid confidence sequences and a one-shot final test fixed in advance.

### Results
- **0 false discoveries in 2,800 tests** on shuffled data (50 shuffles × 56 hypotheses). A naive $p<0.05$ rule: 5.4%.
- **100% power** at a planted effect of 0.01 (≈ 7 Elo), on two different hypotheses.
- **Naive vs honest:** with the safeguards off, Claude and a p-hacker made **137 "discoveries" in pure noise**, and Claude wrote confident explanations for them. ChessCheck kept **0**.
- **58 hypotheses → 14 REAL → 7 distinct findings** (mirror claims like "White +" / "Black −" are merged). Examples: White advantage ≈ +24 Elo; losing and re-queueing within minutes ≈ −5 to −8 Elo. The hot hand looked hugely significant, then died in the vault.
- **16 predictions sealed** and checked live against new Lichess games.

## How we built it
- **Data:** streamed the Lichess open database (`.pgn.zst`) and parsed headers only with a byte-regex, giving 11.2M rated blitz games and 22.2M player-game rows. Bots and players with more than 200 games a day are excluded.
- **Outcome:** each game is scored against the Elo expectation: $r = s - \frac{1}{1 + 10^{-\Delta/400}}$.
- **Test:** each game is compared with the player's *own* baseline (computed from games at least 6 positions away), with two-way (player × game) cluster-robust standard errors.
- **Calibration:** a null that shuffles each player's results in time and **rebuilds** streaks and previous results, plus planted-effect runs for power.
- **Live vault:** Lichess API polling of a fixed 300-player pool, with a normal-mixture confidence sequence so we can peek continuously: $|S_t| < \sqrt{(V_t+\rho)\left(\log(1+V_t/\rho) + 2\log(1/\alpha)\right)}$.
- **Stack:** Python, pandas, NumPy, SciPy and statsmodels; Claude (`claude-sonnet-5-5`) with structured JSON output as the proposer and to translate plain-English questions; Streamlit and Plotly for the dashboard; 26 pytest tests.

## Challenges we ran into
- **Our own test was biased.** The first version found an after-a-loss effect at $p = 10^{-213}$ in **coin flips**. Each game was compared with the player's average, and that average included the very loss that selected the game: the hot-hand selection bias (Miller & Sanjurjo, 2018). Worse, our first calibration *would have passed*, because the shuffled data didn't reproduce the artefact. We fixed both the test and the null.
- **Lichess rate limits.** We built a single paced request queue with back-off after any HTTP 429.
- **Memory.** 22M rows needed categorical encoding and chunked processing.
- **No timezones in the data.** We estimate each player's local clock from their quietest 6 hours.

## Accomplishments that we're proud of
A researcher that **admits uncertainty**. It shows, with numbers, that it doesn't fool itself on noise, still finds real effects, catches its own p-hacking, and stakes its claims on games that hadn't been played yet.

## What we learned
- A self-check is only as good as the noise you test it on.
- "Statistically significant" and "worth caring about" are different. With millions of rows you need a minimum effect.
- An LLM is a great idea generator and a terrible judge of its own ideas: it will write a convincing story for pure noise.

## What's next for ChessCheck
- Engine-based accuracy, to measure play quality rather than results.
- Online FDR control, so an open-ended "ask a question" box never revises old verdicts.
- More months and other time controls, plus power analysis for rare subgroups.
