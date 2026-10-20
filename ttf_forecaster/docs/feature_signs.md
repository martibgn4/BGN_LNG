# Pre-registered expected signs (spec §8 criterion 6)

Fixed on 2026-10-17, **before any real-data run**. These signs come from the gas-fundamentals review,
reconciled with the LNG and weather-power agents' reports. A real-data model whose main fundamental
effects contradict this table fails criterion 6 unless the contradiction is explained physically and
logged in `modelling_log.md` *before* the result is used. Do not edit this table after seeing results:
add a dated amendment below instead.

Columns:
- **Return** is the sign on the prompt TTF excess return (T3/T5).
- **M1–M2** is the sign on the front-minus-second-month spread (T7a); + means backwardation widening.
- **Signs apply to the surprise or update at the origin.** A known level should already be priced
  (expected ≈ 0).

| Feature | Return | M1–M2 | Mechanism | Notes |
|---|---|---|---|---|
| `storage__deviation_pct_pts`, `_z`, `_twh` | − | − | more gas than normal loosens prompt | expected largely priced; loads on curve spreads more than on returns |
| `storage__injection_vs_normal_gwhd` | − | − | flow surprise vs seasonal normal | **main incremental candidate** for H1 |
| `storage__days_of_cover` | − | − | buffer | nonlinear below ~30 days |
| `storage__policy_gap_pct_pts`, `_required_injection_gwhd` | + | − in summer | mandated injection demand | required injection must be capped or logged near the deadline |
| `storage__prob_eos_below_*` | + | + | shortage tail risk | seasonal NaN by design; never impute 0 |
| `storage__proj_eos_dev_vs_normal_pct_pts` | − | − | projected end-of-season looseness | H1 "vs projected path" variable |
| `flows__no_unavailable_next7d_gwhd` | + | + | announced loss of Norwegian supply | unplanned part is the signal (H4); no source yet |
| `flows__lng_sendout_gwhd` (surprise) | − | − | regas supply | endogenous: high TTF attracts cargoes, so the raw level can show + via momentum |
| `weather__hdd_rev_*` | + | + (winter) | colder revision means unpriced demand | the H2 news variable; heating season only |
| `weather__hdd_fc_*`, `_fc_anom_*` | ≈0 | ≈0 | levels are priced | controls only |
| `weather__hdd_ens_spread_*` | none | none | predicts volatility (T8) | flag any directional coefficient |
| `weather__obs_hdd_anom_*` | ≈0 / + | + | already in storage | collinear with storage deviation |
| `power__de_wind_fc_anom_gw` | − | − | less gas burn | |
| `power__de_residual_load_rev_gw` | + | + | more gas burn | revision is the signal |
| `power__fr_nuclear_avail_anom_gw`, `_rev_gw` | − | − | nuclear shortfall raises gas burn | |
| `power__hydro_reservoir_anom_pct_*` | − | weak | thermal call | slow-moving, likely priced |
| `lng__arb_nea_minus_nwe_usd_mmbtu` (+ `_chg7d`) | + | + | Asia pulls flexible cargoes | grade A (Spark) |
| `lng__slots_m1_available` (+ `_chg20d`) | − | − | regas room means weak EU pull | grade A |
| `lng__arrivals_eu_*_chg7d` | − | − | more expected supply | **grade C** until vintaged vessel data exists |
| `lng__arrivals_nwe_cargoes_14d_err` | − | − | supply surprise | needs PIT source |
| `lng__diversions_eu_away_14d` | + | + | cargoes leave Europe | grade C |

**Columns with no defensible sign.** The model may use them, but they cannot support an economic
claim:
- `lng__onwater*_volume_t`: incoming supply vs contango floating storage.
- `lng__transits_*`: low information.
- `flows__no_utilisation_pct`: demand pull vs no spare capacity.
- `lng__destination_undeclared_share`: a volatility input.

**Price-derived, not fundamental.** These columns contain the TTF price itself, so they count as price
information in the criterion-3 twin (H0):
- `power__clean_spark_eur_mwh`, `power__clean_dark_eur_mwh`, `power__spark_minus_dark_eur_mwh`
- `power__switch_gas_price_eur_mwh`, `power__gas_minus_switch_eur_mwh`
- every `cross_commodity__*` spread against TTF

**Excluded from models.** Calendar or deterministic transforms:
- `storage__normal_*`
- `storage__proj_eos_{cold,mild,low_lng,high_outage}_pct`

## Amendments

(none)
