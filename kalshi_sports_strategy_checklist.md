# Kalshi Sports Strategy Test Checklist (trade-data only)

Adjusted for **trade prints only** — no historical order book (book data only exists live via websocket). Every strategy below is restated as something computable from a trade tape: timestamp, price, size, and (if available) a taker-side flag. Column names below are placeholders (`ts`, `market_ticker`, `price`, `size`, `taker_side`) — swap in your actual schema.

Where a tick rule is used to infer trade direction (no taker-side flag), that's the classic Lee-Ready style proxy: price-up tick \= buy-initiated, price-down tick \= sell-initiated, flat tick \= inherit previous sign.

Check one off, run it, write a paragraph of findings, move on.

---

### \[ \] 1\. Trade-flow imbalance predictability — *academic, trade-based substitute for book OFI*

**Source:** Adapted from Cont, Kukanov, Stoikov (2014). True book-level OFI isn't available historically, so this uses **signed trade volume** as the standard fallback (this is the same flow concept your live C++ client tracks off the book — here it's the trade-print analog). **Hypothesis:** Net signed trade volume over a short rolling window predicts forward returns, decaying with horizon.

**DuckDB hint:**

WITH classified AS (

  SELECT \*,

    CASE

      WHEN taker\_side \= 'yes' THEN size

      WHEN taker\_side \= 'no'  THEN \-size

      \-- fallback tick rule if no taker\_side field:

      \-- WHEN price \> LAG(price) OVER (PARTITION BY market\_ticker ORDER BY ts) THEN size

      \-- WHEN price \< LAG(price) OVER (PARTITION BY market\_ticker ORDER BY ts) THEN \-size

      ELSE 0

    END AS signed\_size

  FROM trades

)

SELECT market\_ticker, ts, price,

       SUM(signed\_size) OVER (

         PARTITION BY market\_ticker ORDER BY ts

         ROWS BETWEEN 29 PRECEDING AND CURRENT ROW

       ) AS flow\_30trade

FROM classified;

**Regression:** `fwd_ret_h ~ flow_30trade`, run separately for h ∈ {5s, 30s, 2min}. Newey-West (HAC) SEs, lag ≈ window length in trades. Split by close-score vs. blowout and by sport. **Bar for "real":** Sign-stable, significant coefficient at short horizons that decays smoothly with h — not just one significant bucket out of many tested.

---

### \[ \] 2\. Flow toxicity (VPIN) ahead of price jumps — *academic, trade-based by design*

**Source:** Easley, López de Prado, O'Hara (2012). VPIN was built for exactly this case — no book required, only volume-bucketed, classified trades. **Hypothesis:** Elevated VPIN precedes large price jumps, the same way you found pre-event sweeps in FOMC markets.

**DuckDB hint:**

WITH classified AS (

  SELECT \*,

    CASE WHEN taker\_side \= 'yes' THEN size ELSE \-size END AS signed\_size,

    SUM(size) OVER (PARTITION BY market\_ticker ORDER BY ts) AS cum\_vol

  FROM trades

),

bucketed AS (

  SELECT \*, CAST(cum\_vol / 1000 AS INT) AS bucket\_id  \-- bucket size \= 1000 contracts, tune this

  FROM classified

)

SELECT market\_ticker, bucket\_id,

       ABS(SUM(signed\_size)) / SUM(size) AS bucket\_imbalance

FROM bucketed

GROUP BY market\_ticker, bucket\_id;

Then in pandas: VPIN \= rolling mean of `bucket_imbalance` over N buckets.

**Regression/test:** logistic — `P(jump in next 60s) ~ VPIN_lag`, or simpler: two-sample t-test of mean VPIN in the 60s before a flagged jump vs. a random 60s window. **Bar for "real":** VPIN materially and consistently elevated before jumps, not just visually suggestive on a handful of cases.

---

### \[ \] 3\. "Reverse line movement" via trade flow — *sportsbook folklore, trade-based version*

**Source:** Sportsbook concept of price moving against apparent public money. Without book depth, define it on trade flow instead. **Hypothesis:** When price moves opposite to the sign of net trade flow (e.g., price falls despite net buy volume), that implies a large, mostly-hidden counterparty absorbing flow — and the subsequent drift is more informative than ordinary flow-aligned moves.

**DuckDB hint:**

WITH flow AS (

  SELECT market\_ticker, ts, price,

         SUM(CASE WHEN taker\_side='yes' THEN size ELSE \-size END)

           OVER (PARTITION BY market\_ticker ORDER BY ts ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS flow\_10trade,

         price \- LAG(price, 10\) OVER (PARTITION BY market\_ticker ORDER BY ts) AS price\_chg\_10trade

  FROM trades

)

SELECT \*,

  CASE WHEN SIGN(flow\_10trade) \!= SIGN(price\_chg\_10trade) AND flow\_10trade \!= 0 AND price\_chg\_10trade \!= 0

       THEN 1 ELSE 0 END AS rlm\_flag

FROM flow;

**Regression:** `fwd_ret ~ rlm_flag + rlm_flag*direction`, compare coefficient size/significance against the flow-aligned (non-flagged) subset. **Bar for "real":** RLM-flagged moves show cleaner/stronger continuation than size-matched flow-aligned moves.

---

### \[ \] 4\. Favorite-longshot bias — *academic, very well established*

**Source:** Ali (1977); Snowberg & Wolfers (2010) **Hypothesis:** Extreme longshot contracts (e.g., \<10¢) are systematically overpriced relative to realized win frequency.

**DuckDB hint:**

SELECT

  CAST(price \* 10 AS INT) AS price\_decile,  \-- 0-9, in cents/10

  AVG(outcome) AS realized\_rate,

  AVG(price) AS avg\_implied\_prob,

  COUNT(\*) AS n

FROM trades\_with\_outcome  \-- needs a resolved-outcome join

GROUP BY price\_decile

ORDER BY price\_decile;

**Regression:** `outcome ~ implied_prob` — check intercept ≈ 0 and slope ≈ 1 (perfect calibration); deviation at the low end is the bias. Cluster SEs by game. **Bar for "real":** Calibration curve bows below the 45° line at low implied probability, and the gap survives Kalshi fees.

---

### \[ \] 5\. Hot-hand overreaction in scoring runs — *academic / behavioral*

**Source:** Camerer (1989); Gilovich, Vallone & Tversky (1985) **Hypothesis:** Markets overreact to scoring runs, then mean-revert. **Data note:** this needs scoring/play-by-play events joined to trade timestamps — not derivable from trades alone, so confirm you have (or can pull) a play-by-play feed before running this one.

**DuckDB hint** (once joined to a `scoring_events` table):

SELECT t.market\_ticker, t.ts, t.price,

       LEAD(t.price, 1\) OVER (PARTITION BY t.market\_ticker ORDER BY t.ts) AS price\_after\_run

FROM trades t

JOIN scoring\_events s

  ON t.market\_ticker \= s.market\_ticker

 AND t.ts BETWEEN s.run\_start\_ts AND s.run\_end\_ts;

**Regression:** `fwd_ret_after_run ~ price_move_during_run` — negative coefficient \= overreaction/reversal, positive \= underreaction/continuation. **Bar for "real":** Consistent fee-beating reversal, not noise that's an artifact of price being bounded in \[0,1\].

---

### \[ \] 6\. Pre-event informational drift — *academic transplant*

**Source:** Lucca & Moench (2015), pre-FOMC drift — direct analog of your FOMC sweep finding. **Hypothesis:** Ahead of scheduled, discrete sports info events (lineup confirmations, pitching changes, injury reports), there's a repeatable price drift before the info goes public. **Data note:** also needs an external event-timestamp log, same caveat as \#5.

**Regression:** `price_drift_pre_event ~ 1` (test mean ≠ 0\) pooled across many event instances, then `price_drift ~ event_type + sport` to see which event types carry the effect. **Bar for "real":** Consistent direction/magnitude across many independent instances — not driven by a handful of outlier games (this is the key thing that would make it different from the single-instance FOMC case).

---

### \[ \] 7\. Effective spread regime detection — *hedge fund/market-making style, trade-based proxy*

**Source:** Roll (1984), "A Simple Implicit Measure of the Effective Bid-Ask Spread" — the classic way to estimate spread *without* book data, using the negative serial covariance of trade price changes. (Replaces the book-depth-dependent Avellaneda-Stoikov framing, which needs quotes you don't have historically.) **Hypothesis:** Implied effective spread varies by game state (blowout vs. close, garbage time vs. crunch time) in a way that identifies windows more favorable to passive spread capture.

**DuckDB hint:**

WITH chg AS (

  SELECT market\_ticker, ts,

         price \- LAG(price) OVER (PARTITION BY market\_ticker ORDER BY ts) AS dprice

  FROM trades

)

SELECT market\_ticker,

       COVAR\_POP(dprice, LAG(dprice) OVER (PARTITION BY market\_ticker ORDER BY ts)) AS price\_chg\_autocov

FROM chg

GROUP BY market\_ticker;

Roll spread \= `2 * sqrt(-autocov)` when autocov \< 0 (undefined/discard when ≥ 0 — common in quiet markets, that's informative too).

**Regression:** `roll_spread_estimate ~ score_differential_bin + time_remaining_bin + sport`, HAC/cluster by game. **Bar for "real":** A clear regime split where estimated spread is wide relative to realized short-horizon volatility — i.e., a believable passive-quoting edge net of adverse selection, not just "spread is wider when nothing is happening" (which is trivially true and not by itself tradeable).

---

### \[ \] 8\. Cross-market consistency / stat arb — *hedge fund style*

**Source:** Classic statistical arbitrage (DE Shaw/Renaissance-style); no-arbitrage pricing bounds. **Hypothesis:** Correlated markets on the same game (moneyline-equivalent vs. spread-equivalent, team total vs. game total) occasionally violate no-arbitrage bounds, and violations close on a measurable timescale.

**DuckDB hint:**

SELECT a.ts, a.market\_ticker AS market\_a, b.market\_ticker AS market\_b,

       a.price AS price\_a, b.price AS price\_b,

       (a.price \+ b.price) AS implied\_sum  \-- adjust formula to the actual no-arb relationship

FROM trades a

JOIN trades b

  ON a.game\_id \= b.game\_id

 AND ABS(EXTRACT(EPOCH FROM a.ts \- b.ts)) \< 2  \-- nearest-time match

WHERE a.market\_type \= 'moneyline' AND b.market\_type \= 'spread';

**Regression:** `violation_size ~ time_since_violation_start` (decay rate), or survival analysis on time-to-close. **Bar for "real":** Violations large enough to clear fees, and persisting long enough (seconds-to-minutes, not microseconds) to actually act on with a manual/semi-automated process.

---

### \[ \] 9\. "Fade the public" on marquee games — *retail/forum heuristic*

**Source:** Common r/sportsbook belief that crowd money is more wrong on high-attention games than low-attention ones. **Hypothesis:** Calibration (price vs. realized outcome) is worse for high-volume marquee games than low-volume games.

**DuckDB hint:**

WITH game\_volume AS (

  SELECT game\_id, SUM(size) AS total\_vol

  FROM trades GROUP BY game\_id

),

terciled AS (

  SELECT game\_id,

         NTILE(3) OVER (ORDER BY total\_vol) AS volume\_tercile

  FROM game\_volume

)

SELECT t.market\_ticker, t.price, t.outcome, g.volume\_tercile

FROM trades\_with\_outcome t

JOIN terciled g USING (game\_id);

**Regression:** `outcome ~ implied_prob * volume_tercile` — test whether the calibration slope/intercept differs significantly across terciles. **Bar for "real":** A genuine, monotonic relationship — set the bar high on this one since it's the least theoretically grounded of the list; easy to talk yourself into a noisy result.

---

**Suggested order:** \#4 and \#9 first — pure calibration checks, fastest to run, no model needed, instant pass/fail. Then \#1 and \#2 (you have the flow-classification logic mostly built already, just adapting from book to trade-print version). Save \#5 and \#6 for a session where you've confirmed you have play-by-play/event data joined in — don't start those without checking that data dependency first.  
