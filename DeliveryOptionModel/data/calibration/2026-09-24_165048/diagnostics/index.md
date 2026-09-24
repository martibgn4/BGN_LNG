# Calibration diagnostics - 2026-09-24_165048

As of 2026-09-24. Built from this folder's files and the price history the calibration
used (`history/`, re-fetched from Bloomberg). Numbers behind every plot are in `tables/`.

## Findings (computed)

- **TTF vols:** 6 live mid, 41 settlement, 14 filled, 0 missing; filled from 2030-11 onwards.
- **HH vols:** 7 live mid, 0 settlement, 54 filled, 0 missing; filled from 2027-05 onwards.
- **Brent vols:** 13 live mid, 33 settlement, 15 filled, 0 missing; filled from 2030-10 onwards.
- **TTF_M/TTF_M1:** 0.878 to 0.999 along the strip (lowest Mar-28).
  - variance is dominated by 2022 (48% of it on average), when this correlation averaged 0.988.
  - weakest month pairs: Mar/Apr 0.915, Apr/May 0.982, Oct/Nov 0.984.
- **HH_M/HH_M1:** 0.916 to 0.998 along the strip (lowest Feb-27).
  - variance is dominated by 2022 (40% of it on average), when this correlation averaged 0.993.
  - weakest month pairs: Feb/Mar 0.931, Mar/Apr 0.962, Oct/Nov 0.981.
- **Brent_M/Brent_M1:** 0.994 to 1.000 along the strip (lowest May-27).
  - variance is dominated by 2022 (31% of it on average), when this correlation averaged 1.000.
  - weakest month pairs: May/Jun 0.997, Jun/Jul 0.998, Apr/May 0.998.
- **Stale prints:** at most 1.5% zero weekly returns (2031-11_2031-12).

## Plots

| file | what it shows | what to look for |
|---|---|---|
| `vol_term_structure.png` | ATM vol per delivery month and hub, marked by source; grey = realised vol history gives the same maturity | where market vols stop and fills start; implied far above realised = a rich market |
| `vol_profile.png` | realised vol by days to expiry | the Samuelson shape the horizon adjustment uses; a flat curve means little is stripped from M+1 |
| `corr_term_structure.png` | M/M+1 and cross-hub correlation along the strip | weekly vs daily gap (close asynchrony); level drift with window length |
| `corr_<series>.png` | per tenor pair: pooled correlation and its breakdown by year, variance share, days to expiry, month pair | which years carry the variance; season-change month pairs (Mar/Apr, Sep/Oct) pulling a pair down |
| `data_quality.png` | sample size, window length and stale prints per pair | thin samples, stale far-dated prints |

Colours follow the hub: TTF `#2a78d6`, HH `#eb6834`, Brent `#1baf7a`. Heatmaps: one blue ramp for magnitudes, red-grey-blue
for correlations that can be negative.
