# Model card — ttf_forecaster v0.1

**Status: no production or research model has been selected.** This card describes the model *ladder*
and the conditions any model must meet. It will be filled with results only after a real-data
evaluation against `research_specification.md` §8.

| Field | Value |
|---|---|
| Version | 0.1 (2026-10-17) |
| Owner | LNG desk quant research |
| Intended use | Research: test whether fundamentals improve on the TTF forward curve for fixed-contract excess returns (T3, T5), spreads (T7), volatility (T8) and tail/direction labels (T9) |
| Not intended for | Automated trading; price-level forecasting presented as edge; any use of synthetic outputs as evidence |
| Primary benchmark | Zero excess return (the forward price at origin); also AR(p) on the target's own known lags, rolling mean, price-only twin |
| Candidate models | L0 benchmarks · L1 OLS-HAC, ridge/lasso/elastic net, ECM (cointegration-gated) · L2 LightGBM · L3 rule regimes, filtered Markov switching · L4 quantile, conformal, calibrated event probabilities · L6 constrained stacking (unused until single models pass) |
| Training data | Bitemporal store; grade A/B features only in clean runs; storage grade B (lag-approximated) |
| Validation | Expanding walk-forward, 6-month outer folds, purge on `label_known_time`, inner time-ordered tuning, locked holdout from 2025-10-01 |
| Metrics | OOS R² vs benchmark, HLN-DM (gate) + Clark–West (supplementary), fold win share, ex-crisis R², net Sharpe at 2× cost on non-overlapping trades, P&L concentration, calibration by regime |
| Success criteria | `research_specification.md` §8 (all seven) |
| Current evidence | **None on real data.** Synthetic run (`outputs/synthetic_e2e_report.md`): null world → no model passes; planted world → planted signal detected; price-only twin equally good (criterion 3 positive control still to be built, D-026) |
| Known limitations | `docs/limitations.md`; short PIT weather/LNG history (2021+, crisis-dominated); storage revisions not reproducible; costs are assumptions |
| Ethical / governance | Licensed data never committed; no proprietary text to external models without compliance approval; LLM never a numeric forecaster |
