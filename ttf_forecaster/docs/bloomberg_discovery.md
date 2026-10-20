# Bloomberg discovery and live pull, 2026-10-19

Scope: European gas fundamentals, power and renewables fundamentals, weather, and the Task 1 pull of
every `known` series. Authorised by the user on 2026-10-19 (live pulls and read-only discovery).
Terminal: logged-in Desktop API on localhost:8194, xbbg 0.10.3, blpapi.

Method. Discovery used `//blp/instruments` (instrument search) and `//blp/refdata` ReferenceDataRequest
on NAME, PX_LAST, LAST_UPDATE_DT and QUOTE_UNITS. No history was pulled in discovery. Where history
depth had to be judged, an 8-day window was used (2011-01-03..10). Pulls used a temporary copy of
`configs/` with `live_calls_authorised: true`. The committed `local_data_sources.yaml` was not changed.
Stored data is in the git-ignored `data/live/` store only. No licensed values are written into this
file: only identifiers, dates, row counts and magnitudes needed to judge a series.

"Live" below means `LAST_UPDATE_DT` on or after 2026-10-01. "Dead" means last update before 2026-10-01.
"Entitlement wall" means the field is empty (`PX_LAST` `-`), so no data is returned.

## 1. Discovery: series found

### 1a. Gas: Norway, Gassco and other pipelines

| Ticker | Description | Unit | Freq | History start (pull or probe) | Last update | Usable | Why |
|---|---|---|---|---|---|---|---|
| GAANTOTL Index | Norway Total Exit Daily Within-Day Nominations (Gassco) | MCM/d | daily | 2020-12-01 | 2026-10-18 | **Yes, added** `flow.no.exit_total_mcmd` | Live. Nominations, not final allocations. Starts Dec 2020 |
| GAANNGAG Index | Norway to Germany Daily Historical Flows / Within-Day Nominations | MCM/d | daily | 2012-10-01 | 2026-10-18 | **Yes, added** `flow.no.exit_to_de_mcmd` | Live. Longest Norway-to-Germany history found |
| GAANUKRG Index | Norway to UK Total Exit Daily Within-Day Nominations | MCM/d | daily | 2020-12-01 | 2026-10-18 | **Yes, added** `flow.no.exit_to_uk_mcmd` | Live |
| GAANNOAG Index | Norway to Others Daily Historical Flows | MCM/d | daily | 2018-04-26 | 2026-10-18 | **Yes, added** `flow.no.exit_to_others_mcmd` | Live. Residual route, 1.4 MCM/d |
| GUDFDODF Index | Dornum Gasunie DTS Daily Gas Flows (Norway to Germany) | GWh/d | daily | 2013-11-13 | 2026-10-18 | **Yes, added** `flow.de.dornum_entry_gwhd` | Live. Same corridor as GAANNGAG in GWh/d: cross-check |
| GBITKMCM Index | UK Interconnector Bacton Gas Flow UK>BE | MCM/d | daily | 2016-05-13 | 2026-10-19 | **Yes, added** `flow.gb.bacton_iuk_to_be_mcmd` | Live. UK-to-Belgium route (IUK) |
| NWGPTYGP Index | Norway Fuel Production, Troll field gas | million m3 | monthly | 2011-01-31 | 2026-07-31 | **Yes, added** `prod.no.troll_field_mcm_monthly` | Monthly, one field. 45-day lag assumed |
| EGTPNTRC Index | Russia > CE Europe Net Gas Flow | MCM/d | daily | 2016-01-01 | 2026-10-17 | **Yes, added** `flow.ru.to_ce_europe_mcmd` | Live. Russian net flow into Central Europe |
| EGTPNTRS Index | Russia > SE Europe Gas Flow | MCM/d | daily | 2020-01-01 | 2026-10-17 | **Yes, added** `flow.ru.to_se_europe_mcmd` | Live |
| GAANGERG Index | Norway to Germany Total Exit Daily Within-Day Nominations | MCM/d | daily | not pulled | 2026-10-18 | No | Same level as GAANNGAG (143.3) |
| GAANAGGR Index | Norway Total Exit Daily Historical Flows | MCM/d | daily | not pulled | 2026-10-18 | No | Same level as GAANTOTL (341.0): duplicate |
| GAANZEEI Index | Zeebrugge Exit Intraday Within-Day Nominations | MCM/d | intraday | not pulled | 2026-10-19 | No | Intraday nomination, not a daily flow |
| GAANSFAG Index | St Fergus Exit Daily Historical Flows | MCM/d | daily | not pulled | 2026-10-18 | No | UK terminal leg of Norwegian gas, low value |
| GUDMDODF Index | Dornum Gasunie Deutschland daily flow (Norway to Germany) | MCM/d | daily | not pulled | 2024-10-05 | No | Dead |
| GAAFGERG Index | Gassco Daily Average Gas Flow, Norway to Germany | MCM/d | daily | not pulled | 2020-11-30 | No | Dead |
| GAAFUKG Index | Gassco Daily Average Gas Flow, Norway to UK | MCM/d | daily | not pulled | 2020-11-30 | No | Dead |
| GAAFENTY Index | Gassco Entry SEGAL Daily Average Gas Flow | MCM/d | daily | not pulled | 2020-11-30 | No | Dead |
| GASSGERG Index / GASSUKG Index | Gassco Natural Gas Flows, Total to Germany / UK | MSCM | daily | not pulled | 2020-12-01 | No | Dead, `PX_LAST` 0 |
| .GASSCOEU G Index | Global CIX: Total Gassco flows into Europe | - | daily | not pulled | 2026-10-19 | No | Vendor-computed CIX. Derivation not documented, so not used |
| GZNOFLEU Index / GZNOFLUK Index | Norway NCS Natural Gas Flow, net to Europe / UK | million m3 | - | not pulled | - | No | Entitlement wall (`PX_LAST` empty) |
| PGUTNENO Index | Natural Gas Exports, Norway | GWh | monthly | not pulled | 2026-07-31 | No | `PX_LAST` 0: not a usable value |
| NOEQNGDZ Index | Norway Export of Natural Gas to Algeria | 1000 m3 | - | not pulled | 2015-12-31 | No | Dead |
| TMPLTOEU Index | Natural Gas Trade by Pipeline, total imports Europe | bcm | annual | not pulled | 2023-12-31 | No | Dead |
| EGTPNTRN Index | Russia > NW Europe Gas Flow | MCM/d | daily | not pulled | 2023-08-30 | No | Dead |
| NRSTFLOW Index | NordStream Gas Flows at Greifswald | MCM/d | daily | not pulled | 2023-08-30 | No | Dead |
| .RUSGASF G Index | Global CIX: Russian Gas Flows | - | daily | not pulled | 2026-10-19 | No | CIX, `PX_LAST` 0 |
| UAGFTTOT Index and five Ukraine transit legs | Ukraine gas transit (UkrTransGas) | MCM | - | not pulled | - | Not checked | Found by search only. Transit to EU, not verified |
| GFRUBYUA Index | Gas flow from Russia and Belarus to Ukraine | - | - | not pulled | - | Not checked | Found by search only |
| GSSPMRBI Index | Spain gas imports by entry point, Bilbao, monthly | GWh | monthly | not pulled | 2026-08-31 | No | Monthly. ENAGRGBB (daily) used instead |
| ESNGPPNL Index | Eurostat Natural Gas Primary Production, Netherlands | GWh | monthly | not pulled | 2026-07-31 | No | Monthly Eurostat. Low frequency, NLFMPROD used instead |

### 1b. Gas: production, demand, storage

| Ticker | Description | Unit | Freq | History start | Last update | Usable | Why |
|---|---|---|---|---|---|---|---|
| NLFMPROD Index | Netherlands Gas Production (GTS) | MCM/d | daily | 2012-01-01 | 2026-10-19 | **Yes, added** `prod.nl.gas_production_mcmd` | Live. GWh/d twin NLFIPROD is the same series |
| NLGFPRDF Index | Netherlands Production Daily Gas Flow | GWh/d | daily | not pulled | 2026-10-18 | No | Same series as NLFMPROD in other units |
| NLGPTOTL, NLGPGROP, NLGPOFTO, NLGPOSTO Index | NL monthly production, all, Groningen, offshore, onshore | 1000 m3 | monthly | not pulled | 2022-02-28 | No | Dead |
| OIGPGRON Index | Netherlands Groningen Gas Production | 1000 m3 | - | not pulled | - | No | Entitlement wall |
| DEGSDEWK Index | German SLP Gas Demand, weekly average (BNetzA) | GWh/d | weekly | 2018-01-07 | 2026-10-04 | **Yes, added** `demand.de.slp_weekly_gwhd` | Live. Standard-load-profile customers only, not total German demand |
| TGD DE Index | Total Gas Demand from Power Generation, Germany | GWh | - | not pulled | 2011-12-31 | No | Dead |
| TGD EU15 Index / TGD NL Index | Total gas demand from power generation, EU15 / NL | GWh | - | not pulled | 2016-12-31 | No | Dead |
| GZEUDN Index | Europe Natural Gas Demand End Users | million m3 | - | not pulled | - | No | Entitlement wall |
| NLGDINDH Index | Netherlands Industrial Demand, H-gas (GTS) | GWh/d | daily | not pulled | 2026-10-17 | Not added | Same series as demand.nl.industry_h_mcmd (NLGDMINH) in other units. Its history may be longer (the MCM/d series starts 2022-07): check on the Terminal |
| NLFIINDH Index | Netherlands Industrial Demand, intraday H-gas | GWh/d | intraday | not pulled | 2026-10-19 | No | Intraday |

### 1c. LNG send-out and terminals

| Ticker | Description | Unit | Freq | History start | Last update | Usable | Why |
|---|---|---|---|---|---|---|---|
| LNGAMOSO Index | Montoir LNG France Send-Out (GIE, Elengy) | GWh/d | daily | 2011-12-31 | 2026-10-17 | **Yes, added** `lng.fr.montoir_sendout_gwhd` | Live send-out by terminal |
| DUNKSEND Index | Dunkerque LNG Send Out | GWh/d | daily | 2016-06-26 | 2026-10-19 | **Yes, added** `lng.fr.dunkerque_sendout_gwhd` | Live send-out by terminal |
| LNGASWSO Index | Swinoujscie LNG Poland Send-Out (GIE, Gaz-System) | GWh/d | daily | 2016-05-31 | 2026-10-17 | **Yes, added** `lng.pl.swinoujscie_sendout_gwhd` | Live |
| LNGAATSO Index | Rovigo LNG Italy Send-Out (GIE, Adriatic LNG) | GWh/d | daily | 2011-12-31 | 2026-10-17 | **Yes, added** `lng.it.rovigo_sendout_gwhd` | Live |
| ENAGRGBB Index | Spain LNG Regasification, Bilbao (Enagas) | GWh/d | daily | 2014-08-01 | 2026-10-18 | **Yes, added** `lng.es.bilbao_regas_gwhd` | Live. Regasification output used as send-out |
| FLXGLTNP Index | Zeebrugge LNG Belgium Net Entry Physical Flow (Fluxys) | GWh/d | daily | 2011-01-01 | 2026-10-18 | **Yes, added** `lng.be.zeebrugge_net_entry_gwhd` | Live. This is ENTRY, a send-out proxy |
| LNGAIGSO Index | GIE Isle of Grain LNG Terminal Send-Out | GWh/d | daily | not pulled | 2022-02-12 | No | Dead. Grain send-out not available after Feb 2022 |
| EGFMHUDF Index | Huelva LNG Entry Flow (Enagas) | MCM/d | daily | not pulled | 2020-03-31 | No | Dead |
| EGFG0036 / EGFV0036 and other Huelva/Sagunto Enagas legs | Huelva, Sagunto LNG capacity and flow | various | daily | not pulled | - | Not checked | Found by search only |
| LNGAMOIN Index | Montoir LNG France Inventory (GIE) | 1000 m3 | daily | not pulled | 2026-10-17 | Not added | Live. Usable, not added (LNG tank inventory, not requested) |
| LNGASWIN, LNGASWAC, LNGASWCC Index | Swinoujscie inventory, available and contracted capacity | 1000 m3 / GWh/d | daily | not pulled | 2026-10-17 | Not added | Live. Partial answer to the send-out capacity gap (see section 3) |
| LNGAATIN, LNGAATAC, LNGAATCC Index | Rovigo inventory, available and contracted capacity | 1000 m3 / GWh/d | daily | not pulled | 2026-10-17 | Not added | Live. Same as above |
| FGFMMNDF Index | Montoir LNG Gas Entry Physical Flow (NaTran) | MCM/d | daily | not pulled | 2026-10-15 | No | Entry, duplicate of the Montoir send-out |
| FGNODLDE Index | Dunkerque LNG Within Day Entry Nomination | GWh/d | daily | not pulled | 2026-10-19 | No | Nomination, not send-out |
| FHRMLTEA Index and other Fluxys intraday Zeebrugge legs | Zeebrugge intraday nominations | MCM/d | intraday | not pulled | 2026-10-19 | No | Intraday |
| SNAMSLNG Index | Panigaglia LNG Italy Gas Entry Physical Flow (SNAM) | GWh/d | daily | not pulled | 2026-10-17 | No | Live but about 1.4 GWh/d, immaterial |
| CPA1M ... Index | Panigaglia available regas capacity forecast (GIE) | GWh | - | not pulled | - | Not checked | Capacity forecast, not send-out |
| LBEV26 etc. (Sabine Pass to Gate freight, Spark30S) | LNG freight futures | USD/MMBtu | - | not pulled | 2026 | Not relevant | Freight, not gas flow. Gate terminal send-out not found |
| .FRLNGSO G Index | Global CIX: Fos and Montoir LNG send-out | GWh | daily | not pulled | - | No | CIX, derivation not documented |

Searched with no result: "LNG sendout terminal", "Gate LNG send out", "Revithoussa", "Grain LNG Terminal
send out" (only the dead Grain series). Gate (Rotterdam) and Revithoussa (Greece) have no send-out
series on Bloomberg in this search.

### 1d. Power and renewables

| Ticker | Description | Unit | Freq | History start | Last update | Usable | Why |
|---|---|---|---|---|---|---|---|
| AMPNDFDA Index | Amprion Daily Average Day-Ahead Forecast Demand (Germany) | MW | daily | 2011-01-01 | 2026-10-17 | **Yes, added** `power.de.amprion_load_fc_mw` | Live. **One TSO zone only**, not national. See anomaly 4 |
| AMPNVLDA Index | Amprion Daily Average Actual Vertical Load (Germany) | MW | daily | 2011-01-01 | 2026-10-17 | **Yes, added** `power.de.amprion_load_act_mw` | Live. One zone only |
| PW50DAVG / PWTEDAVG / PWAMDAVG / PWENDAVG Index | EEX Transparency day-ahead planned wind, 50Hertz, TenneT, Amprion, TransnetBW | MW | daily | not pulled | 2026-10-15 | No | Duplicates the national wind forecast already known (EXDEWSAY / EXDEWOAY). TransnetBW leg is dead (PSENDAVG, 2021-02-08) |
| RTEGNUCT Index | France Daily Average Generation, Nuclear (RTE) | MW | daily | not pulled | 2026-10-18 | No | Generation, not availability. Duplicates GT1CFRNU (already known) |
| RTEGNUCD Index / RTEGNUCL Index | France nuclear generation, daily total MWh / intraday MW | MWh / MW | daily / intraday | not pulled | 2026-10-18 | No | Generation |
| FRPGFV3 Index | Flamanville 3 nuclear generation (intraday) | MW | intraday | not pulled | - | No | Generation, single unit |
| PZFNGUAC Index | France Nuclear Power Generation | TWh | annual | not pulled | 2040-12-31 | No | Annual, and the date is a forecast placeholder |
| WTIFDEDA Index | Wattsight Inflow Day Ahead Forecast Germany | GWh/d | daily | not pulled | 2026-10-19 | Not checked | Live. The meaning of "inflow" is not established: not added |
| PZDEGS Index | Germany Solar Power Generation | MW | - | not pulled | - | No | Entitlement wall |
| PZDEGPCP Index | Germany utility-scale solar PV installed capacity | GW | - | not pulled | - | No | Entitlement wall |
| PSENDAVG Index | TransnetBW day-ahead planned solar (EEX Transparency) | MW | daily | not pulled | 2021-02-08 | No | Dead (already known as a dead series in bloomberg_power.yaml) |
| BELDAVG Index | Elia Power Real-Time Load Forecast Averages (Belgium) | MW | - | not pulled | 2021-03-17 | No | Dead |
| EL11E003 Index | Argus Daily German OTC base load day ahead | EUR/MWh | daily | not pulled | - | No | Entitlement wall |
| LD1CDE ACT 24AV, GT1CDESL, GT1CDEWS | Germany load actual, solar and wind generation | MW | daily | 2015-01-01 | 2026-10-18 | (existing) | Already known and pulled in Task 1 |
| MGENFS01 to MGENFS24 Index | MAVIR (Hungary) hourly solar generation forecast | MW | hourly | not pulled | - | Not relevant | Hungary, outside the requested scope |
| WNDFW1DE / WNDFW2DE / WNDFTTDE Index | Wind forecast capacity Germany (onshore, offshore, total) | MW | - | not pulled | 2036-12-31 | No | Capacity schedule with 2036 end date, not a generation forecast |

No national German load FORECAST, no national German solar FORECAST and no French nuclear AVAILABILITY
series were found (section 3).

### 1e. Weather

| Ticker | Description | Unit | Freq | History start | Last update | Usable | Why |
|---|---|---|---|---|---|---|---|
| WENLEHAM Index | WSI Custom Weather Amsterdam/Schiphol (observed) | degC | daily | 2011-01-01 | 2026-10-19 | **Yes, added** `weather.nl.schiphol_t2m_obs_degc` | Live. Observed station value, full daily coverage since 2011 |
| WEUKEGLL Index | WSI Custom Weather London/Heathrow (observed) | degC | daily | 2011-01-01 | 2026-10-18 | **Yes, added** `weather.gb.heathrow_t2m_obs_degc` | Live |
| WEFRLFPO Index | WSI Custom Weather Paris/Orly (observed) | degC | daily | 2011-01-01 | 2026-10-18 | **Yes, added** `weather.fr.orly_t2m_obs_degc` | Live |
| WEDLEDDB Index | WSI Custom Weather Berlin Schoenfeld (observed) | degC | daily | 2011-01-01 | 2026-10-18 | **Yes, added** `weather.de.berlin_t2m_obs_degc` | Live |
| WEDLEDDM Index | WSI Custom Weather Munich/Riem (observed) | degC | daily | 2011-01-01 | 2026-10-19 | **Yes, added** `weather.de.munich_t2m_obs_degc` | Live |
| WER1DL00 Index | WSI CW Weather Regional Average, Country Germany | degC (units blank) | daily | 2011-01-01 | 2026-10-18 | **Yes, added** `weather.de.regional_t2m_obs_degc` | Live. Units not stated in QUOTE_UNITS; values -13 to +30 match Celsius |
| WEDNEKCH Index | WSI Custom Weather Copenhagen/Kastrup | degC | daily | not pulled | 2026-10-19 | Not added | Live. Found, not needed for the current hypotheses |
| WEFRLFPB Index | WSI Custom Weather Paris/Le Bourget | degC | daily | not pulled | 2026-10-19 | Not added | Live. Orly chosen as the main Paris station |
| L6A / L7A / L5A / L8, Q7, Q5 CAT contracts (CME) | Monthly Cumulative Average Temperature futures: Paris, Amsterdam, London, Berlin, Rome, Madrid | index points | monthly | not pulled | 2026-10-16 (Paris Oct26, PX 446) | Not added | These are prices of a temperature index, not temperature observations. The settlement-to-realised-index mapping is unverified. Separate decision |
| CEHGM V26 / CEHGBOM / HISTCNGH, NOAA HDD (US) | US heating degree days, ECMWF ensemble forecast, NOAA population-weighted HDD | HDD | weekly/daily | not pulled | 2026 | Not relevant | US only |
| .GERMTEMP G / .GRMTEMP G Index | Global CIX: Germany weighted / average temperature | degC | - | not pulled | - | No | Entitlement wall (`PX_LAST` empty) |
| PZDEDT Index | Germany Power Demand Temperature Forecast | MW | - | not pulled | - | No | Entitlement wall |

No European HDD/CDD series (population- or gas-weighted), no European temperature FORECAST with issue
dates, and no ensemble forecast was found (section 3). The `WEATHER` / `WMAP` Terminal functions were not
tested: discovery used the instrument search (`instrumentListRequest`) only.

### 1f. Other candidates reference-checked

| Ticker | Description | Unit | Last update | Usable | Why |
|---|---|---|---|---|---|
| EXTEDAVG / PS50DAVG / PSAMDAVG Index | Day-ahead planned solar, TenneT, 50Hertz, Amprion | MW | 2026-10-15 | (existing) | Already known |
| EXDEWSAY / EXDEWOAY Index | Germany wind onshore / offshore generation forecast (EEX) | MW | 2026-10-18 | (existing) | Already known |
| DEDAILY Index | NOMX DE Base Futures Day | EUR/MWh | - | No | PX_LAST and LAST_UPDATE_DT empty (already `requires_user_input`) |
| THE / PEG / PSV / PVB / VTP monthly root tickers | resolved in D-040, see section 4a | EUR/MWh | 2026-10-19 | Yes (after D-040) | Pulled 2015-2026 with the verified forms |

## 2. Pull results (Task 1 and the Task 2 additions)

Rows are stored in `data/live/` as normalised records. `first` and `last` are local dates (the stored
UTC midnight shifted by 12 hours). `NaN` is measured on the adapter output (the adapter returns only
values Bloomberg actually sent, so 0 is the measured count, not a guarantee). An "empty before" note
means the Terminal returned no data before that year: the first date is the start of the history. This
is not an error.

| Series | Rows | First | Last | NaN | Note |
|---|---:|---|---|---:|---|
| curve TTF (tzt) volume, open interest | 129279 | 2015-01-02 | 2026-10-19 | 0 |  |
| curve JKM settle | 55927 | 2015-01-02 | 2026-10-16 | 0 |  |
| curve HH settle | 69218 | 2015-01-02 | 2026-10-16 | 0 |  |
| curve Brent settle | 68400 | 2015-01-02 | 2026-10-16 | 0 |  |
| curve NBP settle | 70107 | 2015-01-02 | 2026-10-16 | 0 |  |
| curve THE settle | 0 |  |  |  | no data |
| curve PEG settle | 0 |  |  |  | no data |
| curve PSV settle | 0 |  |  |  | no data |
| curve PVB settle | 0 |  |  |  | no data |
| curve VTP settle | 0 |  |  |  | no data |
| curve TTF_USD settle | 51372 | 2015-02-10 | 2026-10-16 | 0 |  |
| contract_metadata HH | 159 |  |  | 0 | 14 rule mismatches |
| contract_metadata NBP | 159 |  |  | 0 | 10 rule mismatches |
| contract_metadata TTF_USD | 130 |  |  | 0 | 21 rule mismatches |
| vendor expiry JKM | 159 |  |  |  | no rule (requires_user_input); raw vendor dates only |
| vendor expiry Brent | 159 |  |  |  | no rule (requires_user_input); raw vendor dates only |
| fx.eurgbp.spot_gbp_per_eur | 4116 | 2011-01-03 | 2026-10-19 | 0 |  |
| price.ttf.da_eur_mwh | 3320 | 2017-09-05 | 2026-10-18 | 0 | empty before 2014 |
| price.ttf.da_pegas_eur_mwh | 4044 | 2012-12-13 | 2026-10-19 | 0 | empty before 2012 |
| price.ttf.bom_eur_mwh | 2491 | 2013-04-01 | 2026-10-19 | 0 | empty before 2015 |
| price.coal_api2.m1_usd_t | 4077 | 2011-01-03 | 2026-10-16 | 0 |  |
| price.eua.front_eur_t | 4072 | 2011-01-03 | 2026-10-16 | 0 |  |
| price.power_de.base_m1_eur_mwh | 2114 | 2018-08-09 | 2026-10-16 | 0 | empty before 2014 |
| price.power_de.da_eur_mwh | 3910 | 2011-02-01 | 2026-10-19 | 0 |  |
| price.power_nl.base_m1_eur_mwh | 4086 | 2011-02-01 | 2026-10-19 | 0 |  |
| price.power_nl.base_w1_eur_mwh | 20 | 2026-09-21 | 2026-10-16 | 0 | empty before 2014 |
| vol.ttf.m1_atm_implied_pct | 218 | 2025-10-31 | 2026-10-16 | 0 | empty before 2014 |
| storage.de.inventory_twh | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.de.injection_gwhd | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.de.withdrawal_gwhd | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.de.working_gas_volume_twh | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.nl.inventory_twh | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.nl.injection_gwhd | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.nl.withdrawal_gwhd | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.nl.working_gas_volume_twh | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.be.inventory_twh | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.be.injection_gwhd | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.be.withdrawal_gwhd | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.be.working_gas_volume_twh | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.fr.inventory_twh | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.fr.injection_gwhd | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.fr.withdrawal_gwhd | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.fr.working_gas_volume_twh | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.at.inventory_twh | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.at.injection_gwhd | 5767 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.at.withdrawal_gwhd | 5768 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.at.working_gas_volume_twh | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.dk.inventory_twh | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.dk.injection_gwhd | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.dk.withdrawal_gwhd | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.dk.working_gas_volume_twh | 5769 | 2011-01-01 | 2026-10-17 | 0 |  |
| storage.cz.inventory_twh | 5361 | 2012-02-13 | 2026-10-17 | 0 | empty before 2012 |
| storage.cz.injection_gwhd | 5361 | 2012-02-13 | 2026-10-17 | 0 | empty before 2012 |
| storage.cz.withdrawal_gwhd | 5361 | 2012-02-13 | 2026-10-17 | 0 | empty before 2012 |
| storage.cz.working_gas_volume_twh | 5361 | 2012-02-13 | 2026-10-17 | 0 | empty before 2012 |
| storage.gb.inventory_twh | 5283 | 2012-05-01 | 2026-10-17 | 0 | empty before 2012 |
| storage.gb.injection_gwhd | 5283 | 2012-05-01 | 2026-10-17 | 0 | empty before 2012 |
| storage.gb.withdrawal_gwhd | 5283 | 2012-05-01 | 2026-10-17 | 0 | empty before 2012 |
| storage.gb.working_gas_volume_twh | 5283 | 2012-05-01 | 2026-10-17 | 0 | empty before 2012 |
| lng.eu.sendout_gwhd | 5527 | 2011-08-31 | 2026-10-17 | 0 |  |
| lng.eu.terminal_inventory_1000m3 | 5527 | 2011-08-31 | 2026-10-17 | 0 |  |
| demand.fr.total_gwhd | 5032 | 2013-01-01 | 2026-10-18 | 0 | empty before 2013 |
| demand.fr.ldz_gwhd | 5770 | 2011-01-01 | 2026-10-18 | 0 |  |
| demand.fr.industry_hms_gwhd | 5770 | 2011-01-01 | 2026-10-18 | 0 |  |
| demand.fr.industry_nonhms_gwhd | 5770 | 2011-01-01 | 2026-10-18 | 0 |  |
| demand.fr.terega_pitd_gwhd | 4933 | 2013-03-07 | 2026-10-15 | 0 | empty before 2013 |
| demand.fr.terega_pic_gwhd | 4933 | 2013-03-07 | 2026-10-15 | 0 | empty before 2013 |
| demand.be.ldz_mcmd | 5131 | 2012-10-01 | 2026-10-18 | 0 | empty before 2012 |
| demand.be.industry_mcmd | 5131 | 2012-10-01 | 2026-10-18 | 0 | empty before 2012 |
| demand.be.power_mcmd | 5131 | 2012-10-01 | 2026-10-18 | 0 | empty before 2012 |
| demand.nl.industry_h_mcmd | 1565 | 2022-07-01 | 2026-10-17 | 0 | empty before 2014 |
| demand.nl.industry_l_mcmd | 1565 | 2022-07-01 | 2026-10-17 | 0 | empty before 2014 |
| demand.at.area_east_mcmd | 5039 | 2013-01-01 | 2026-10-18 | 0 | empty before 2013 |
| demand.at.area_tyrol_mcmd | 4766 | 2013-10-01 | 2026-10-18 | 0 | empty before 2013 |
| demand.at.area_vorarlberg_mcmd | 4766 | 2013-10-01 | 2026-10-18 | 0 | empty before 2013 |
| demand.gb.ldz_eastern_mcmd | 5767 | 2011-01-01 | 2026-10-17 | 0 |  |
| flow.no.exit_total_mcmd | 2148 | 2020-12-01 | 2026-10-18 | 0 | empty before 2014 |
| flow.no.exit_to_de_mcmd | 5131 | 2012-10-01 | 2026-10-18 | 0 | empty before 2012 |
| flow.no.exit_to_uk_mcmd | 2147 | 2020-12-01 | 2026-10-18 | 0 | empty before 2014 |
| flow.no.exit_to_others_mcmd | 3098 | 2018-04-26 | 2026-10-18 | 0 | empty before 2014 |
| flow.de.dornum_entry_gwhd | 4723 | 2013-11-13 | 2026-10-18 | 0 | empty before 2013 |
| flow.ru.to_ce_europe_mcmd | 3795 | 2016-01-01 | 2026-10-17 | 0 | empty before 2014 |
| flow.ru.to_se_europe_mcmd | 2466 | 2020-01-01 | 2026-10-17 | 0 | empty before 2014 |
| prod.nl.gas_production_mcmd | 5400 | 2012-01-01 | 2026-10-19 | 0 | empty before 2012 |
| prod.no.troll_field_mcm_monthly | 190 | 2011-01-31 | 2026-07-31 | 0 |  |
| demand.de.slp_weekly_gwhd | 456 | 2018-01-07 | 2026-10-04 | 0 | empty before 2014 |
| lng.global.afloat_20d_volume_t | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.global.afloat_20d_vessels_count | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.global.afloat_30d_volume_t | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.global.afloat_30d_vessels_count | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.usa.afloat_20d_vessels_count | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.qatar.afloat_20d_vessels_count | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.australia.afloat_20d_vessels_count | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.nigeria.afloat_20d_vessels_count | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.russia.afloat_20d_vessels_count | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.other.afloat_20d_vessels_count | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.panama.laden_transits_count | 2872 | 2018-12-04 | 2026-10-17 | 0 | empty before 2014 |
| lng.suez.laden_transits_count | 2872 | 2018-12-04 | 2026-10-17 | 0 | empty before 2014 |
| lng.cape.laden_transits_count | 2873 | 2018-12-04 | 2026-10-17 | 0 | empty before 2014 |
| lng.nsr.laden_transits_count | 2872 | 2018-12-04 | 2026-10-17 | 0 | empty before 2014 |
| lng.amer_to_nwe.flow_t_per_d | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.amer_to_nasia.flow_t_per_d | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.amer_to_sasia.flow_t_per_d | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.amer_to_med.flow_t_per_d | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.amer_to_amer.flow_t_per_d | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.us.exports_t_per_d | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.qatar.exports_t_per_d | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.global.exports_t_per_d | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.global.imports_t_per_d | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.nwe.imports_t_per_d | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.nasia.imports_t_per_d | 3578 | 2017-01-01 | 2026-10-18 | 0 | empty before 2014 |
| freight.blng1_aus_jpn.front_future_usd_per_d | 682 | 2024-02-01 | 2026-10-16 | 0 | empty before 2014 |
| freight.blng2_usg_cont.front_future_usd_per_d | 685 | 2024-01-29 | 2026-10-16 | 0 | empty before 2014 |
| freight.blng3_usg_jpn.front_future_usd_per_d | 682 | 2024-02-01 | 2026-10-16 | 0 | empty before 2014 |
| lng.fr.montoir_sendout_gwhd | 5405 | 2011-12-31 | 2026-10-17 | 0 |  |
| lng.fr.dunkerque_sendout_gwhd | 3752 | 2016-06-26 | 2026-10-19 | 0 | empty before 2014 |
| lng.pl.swinoujscie_sendout_gwhd | 3792 | 2016-05-31 | 2026-10-17 | 0 | empty before 2014 |
| lng.it.rovigo_sendout_gwhd | 5405 | 2011-12-31 | 2026-10-17 | 0 |  |
| lng.es.bilbao_regas_gwhd | 4458 | 2014-08-01 | 2026-10-18 | 0 | empty before 2014 |
| lng.be.zeebrugge_net_entry_gwhd | 5770 | 2011-01-01 | 2026-10-18 | 0 |  |
| power.de.wind_onshore_fc_mw | 3427 | 2017-03-25 | 2026-10-19 | 0 | empty before 2014 |
| power.de.wind_offshore_fc_mw | 3429 | 2017-03-25 | 2026-10-19 | 0 | empty before 2014 |
| power.de.wind_onshore_gen_mw | 4309 | 2015-01-01 | 2026-10-18 | 0 | empty before 2014 |
| power.de.wind_total_gen_mw | 5683 | 2011-01-01 | 2026-10-19 | 0 |  |
| power.de.solar_fc_tennet_mw | 5575 | 2011-01-01 | 2026-10-19 | 0 |  |
| power.de.solar_fc_50hertz_mw | 5563 | 2011-01-01 | 2026-10-19 | 0 |  |
| power.de.solar_fc_amprion_mw | 5562 | 2011-01-01 | 2026-10-19 | 0 |  |
| power.de.solar_gen_mw | 4309 | 2015-01-01 | 2026-10-18 | 0 | empty before 2014 |
| power.de.load_act_mw | 4309 | 2015-01-01 | 2026-10-18 | 0 | empty before 2014 |
| power.fr.nuclear_gen_mw | 4276 | 2014-12-16 | 2026-10-18 | 0 | empty before 2014 |
| power.no.hydro_reservoir_mwh | 392 | 2019-03-27 | 2026-09-30 | 0 | empty before 2014 |
| power.se.hydro_reservoir_mwh | 391 | 2019-03-27 | 2026-09-30 | 0 | empty before 2014 |
| power.ch.hydro_reservoir_mwh | 590 | 2015-02-25 | 2026-09-30 | 0 | empty before 2014 |
| power.at.hydro_reservoir_mwh | 605 | 2015-01-07 | 2026-09-30 | 0 | empty before 2014 |
| power.fr.hydro_reservoir_mwh | 612 | 2014-12-31 | 2026-09-30 | 0 | empty before 2014 |
| weather.nl.schiphol_t2m_obs_degc | 5771 | 2011-01-01 | 2026-10-19 | 0 |  |
| weather.gb.heathrow_t2m_obs_degc | 5770 | 2011-01-01 | 2026-10-18 | 0 |  |
| weather.fr.orly_t2m_obs_degc | 5770 | 2011-01-01 | 2026-10-18 | 0 |  |
| weather.de.berlin_t2m_obs_degc | 5770 | 2011-01-01 | 2026-10-18 | 0 |  |
| weather.de.munich_t2m_obs_degc | 5771 | 2011-01-01 | 2026-10-19 | 0 |  |
| weather.de.regional_t2m_obs_degc | 5770 | 2011-01-01 | 2026-10-18 | 0 |  |
| power.de.amprion_load_fc_mw | 5740 | 2011-01-01 | 2026-10-17 | 0 |  |
| power.de.amprion_load_act_mw | 5399 | 2011-01-01 | 2026-10-17 | 0 |  |
| flow.gb.bacton_iuk_to_be_mcmd | 3734 | 2016-05-13 | 2026-10-19 | 0 | empty before 2014 |

Derived series not pulled (not Bloomberg identifiers): the EU/national HDD, the Germany-only load,
and the residual Norway route (Gassco total less Germany and UK).

## 2b. D-040 pull: THE and the PNX hubs (monthly settle, 2015 to 2026)

Same authorised method as Task 1 (temporary config copy, `live_calls_authorised: true`). Stored in `data/live`.
The five curve partitions were rebuilt in this pass: the first two passes had written stale post-expiry rows
(see 4a), and those partitions contained only this session's pulls.

| Series | Rows | First | Last | Contracts | Note |
|---|---:|---|---|---:|---|
| curve THE settle | 71139 | 2015-01-02 | 2026-10-16 | 159 | 159 contracts; 1 rows after last trade dropped |
| curve PEG settle | 19850 | 2015-01-02 | 2026-10-16 | 158 | 158 contracts; 2134 rows after last trade dropped |
| curve PSV settle | 16958 | 2015-09-18 | 2026-10-16 | 151 | 151 contracts; 2157 rows after last trade dropped |
| curve PVB settle | 13748 | 2019-05-30 | 2026-10-16 | 95 | 95 contracts; 1685 rows after last trade dropped |
| curve VTP settle | 16476 | 2016-12-01 | 2026-10-16 | 122 | 122 contracts; 2288 rows after last trade dropped |
| contract_metadata THE | 159 |  |  | 159 | 9 rule mismatches |
| contract_metadata PEG | 158 |  |  | 158 | 19 rule mismatches |
| contract_metadata PSV | 151 |  |  | 151 | 4 rule mismatches |
| contract_metadata PVB | 95 |  |  | 95 | 2 rule mismatches |
| contract_metadata VTP | 122 |  |  | 122 | 7 rule mismatches |

Monthly rows per hub after the post-expiry drop: THE 71,139; PEG 19,850; PSV 16,958; PVB 13,748; VTP 16,476.
Contract metadata: THE 159, PEG 158, PSV 151, PVB 95, VTP 122 (VTP has 123 contracts in metadata; one
contract's only rows were after its vendor last-trade date, so it has no curve rows).

## 3. Gaps NOT found on Bloomberg: requests for the Terminal user

Each row is a request the user can act on. Use the Terminal search function (`SECF`/`FLDS`/`DES`) or
the named source. Please upload anything found as a CSV or Parquet with a date column, value column,
unit, and the ticker, to a file under `data/raw/extracts/` (git-ignored). The local-file adapter reads it.

| # | Data wanted | Geography, frequency, unit | Why it is needed | Where to look on the Terminal |
|---|---|---|---|---|
| 1 | **National German load FORECAST**, day-ahead (ENTSO-E DE-LU "day-ahead total load forecast") | DE, hourly and daily average, MW, from 2015 | Load-forecast error drives the power and gas-burn features. Only the Amprion zone is on Bloomberg | `DES` search "Germany load forecast", "DE-LU day-ahead total load forecast", "ENTSO-E load forecast Germany"; or the EEX Transparency load grid |
| 2 | **Full-coverage German solar FORECAST**, national, day-ahead | DE, daily average, MW, from 2015 | The three live TSO solar forecasts sum to less than national PV (TransnetBW dead since Feb 2021), so the solar feature is biased | `DES` "Germany solar forecast day-ahead national", "DE-LU solar day-ahead forecast", ENTSO-E solar forecast, or the EEX Transparency national solar line |
| 3 | **French nuclear AVAILABILITY** (available capacity, not generation). **PARTLY CLOSED (D-043):** BSRCH unavailable MW by fuel for FR is stored daily from 2026-10-13 (vintages from that date only). Real history is one rolling year back. Fleet capacity is still needed to turn unavailable MW into available MW | FR, daily, MW, from 2015 | Outages drive the French and Belgian power price and the LNG send-out. Only generation was on Bloomberg before D-043 | History before 2025-10: `DES`/`SECF` for an archive or the REMIT unavailability feed (EDF/RTE, "Disponibilite du parc nucleaire"). Nuclear fleet capacity: a Bloomberg series or a cited config value |
| 4 | **DFBAW weekly power, history** | NL, weekly contract settle, EUR/MWh, from 2015 | Only the active weekly contract is on the Terminal (20 rows since 2026-09-21). A continuous weekly history would make the weekly power feature testable | `DES` on DFBAW and its historical week contracts (e.g. "DFBAW W43-25"); or EEX weekly history |
| 5 | **Implied volatility, TTF front month, history before Oct 2025** | TTF options, ATM implied vol, %, from 2015 | FJS1 `HIST_CALL_IMP_VOL` only exists from 2025-10-31 (218 rows). The vol feature cannot be backtested before that | `OMON <GO>` / `OVDV` history on FJS generics; or ICE Endex options settlements |
| 6 | **Total German daily gas demand** (national, GWh/d, from 2015) | DE, daily, GWh/d | TGD DE is dead (2011). The only German series is standard-load-profile demand | Trading Hub Europe (THE) published demand; `DES` "Germany gas demand daily THE"; Bnetza |
| 7 | **Grain LNG send-out after Feb 2022**; Huelva and Sagunto send-out after 2020 | UK, ES, daily, GWh/d | LNGAIGSO (Grain) and EGFMHUDF (Huelva) are dead. Send-out by terminal is incomplete | `DES` "Grain LNG send out daily National Grid", "Huelva LNG send-out Enagas" |
| 8 | **UK daily gas production** | UK, daily, MCM/d or GWh/d | Search "UK gas production daily" returned nothing | National Gas, NSTA, ONS production data on `DES` |
| 9 | **TurkStream daily flow** (Strandzha, Kipi) | Entry Bulgaria / Greece, daily, MCM/d | Search "TurkStream flow" returned nothing. EGTP CE/SE series cover Russia to Europe only in aggregate | `DES` "Strandzha 2 gas flow", "Kipi entry flow", ENTSOG/Bulgartransgaz |
| 10 | **Algerian pipeline flows to Spain and Italy** (Medgaz, Transmed) | Entry ES and IT, daily, GWh/d | Search returned only trade monthly series | `DES` "Medgaz flow", "Transmed Italy entry", Enagas, Snam |
| 11 | **Gassco daily flows before Dec 2020, all routes** | NO, daily, MCM/d | Gassco totals (GAANTOTL, GAANUKRG) start Dec 2020. GAANNGAG (to Germany) starts 2012 | Confirm on the Terminal whether an earlier Gassco total exists; otherwise the total history is 2020+ |

## 4. Anomalies and dead ends (for the modelling team)

1. **JKM and Brent expiry has no rule** (`business_days_before_delivery: {M: null}`, requires_user_input).
   The 159 vendor `LAST_TRADEABLE_DT` dates per hub are saved raw to `data/live/vendor_expiry/`, not as
   contract metadata. Not used for targets.
2. **Rule mismatches (D-035 applied).** TTF 8 of 159 (already authoritative). HH 14 of 159, NBP 10 of 159,
   TTF_USD 21 of 130. For HH and NBP the mismatches are one-session holiday effects. For TTF_USD the
   mismatches are larger: e.g. 2015-03, rule 26 Feb, vendor 16 Mar. The rule does not describe this root.
   TTF_USD is not used for targets (target_definitions.md section 0).
3. **Amprion forecast vs actual (`power.de.amprion_load_fc_mw` / `_act_mw`).** The forecast maximum is
   31,863 MW against an actual maximum of 19,279 MW for the same zone. Five forecast rows are flagged
   outlier-suspect. This needs a check of unit or period before use. Not used in any feature.
4. **Weekly power (DFBAW) and FJS1 implied vol have short history.** See section 3, items 6 and 7.
5. **Shorter histories than the config implies.** `demand.nl.industry_h_mcmd` (NLGDMINH) starts 2022-07.
   `flow.no.exit_total_mcmd` starts 2020-12. `flow.ru.to_se_europe_mcmd` starts 2020-01.
6. **Weekly DEGSDEWK (SLP demand)** is weekly and last updated 2026-10-04; treat as weekly, not daily.
7. **Weather units.** WER1DL00 has blank units; values match Celsius but the unit is unconfirmed.
   The WSI "Custom Weather" description does not say whether the daily value is mean, max or min. Confirm
   with DES before any degree-day build. The range (about -16 to +34 degC across stations) is plausible for daily means, but this is not verified.
8. **Revised data.** TTF volume and open interest for the most recent session (18 contract rows) were
   revised by Bloomberg between two pulls on 2026-10-19. The store keeps the first vintage (see log).
9. **Outlier-suspect rows** are flagged, not dropped, by the ingestion validation (counts in the pull log).
10. **Zero-valued series.** PGUTNENO and .RUSGASF G have PX_LAST 0 and are not usable.
11. **CIX series** (.GASSCOEU G, .FRLNGSO G, .GERMTEMP G, .RUSGASF G) are vendor-computed indices with no
    documented derivation. Not used.

## 4a. Resolved (D-040): THE, PEG, PSV, PVB, VTP monthly history

The earlier finding (section 4 of the first pass: "no monthly history") was a ticker-naming gap, not missing
data. Verified 2026-10-19 with bdh on short windows (orchestrator follow-up, re-checked here):

| Hub | Form | Ticker examples | Result |
|---|---|---|---|
| THE (ICE Endex) | TZT rule, root LFT, 2-digit year | `LFTX16 Comdty`, `LFTX21 Comdty`, `LFTX26 Comdty` | 5, 5 and 6 rows (2016-10 settle 15.82; 2021-10 96.84; 2026-10 74.98) |
| THE | same | `LFTX6 Comdty` (1-digit) | 6 rows for 2026, 0 for 2016: not used (LFTX26 works) |
| PEG, PSV, PVB, VTP (PNX/EEX) | listed: root + month + 1-digit year + M | `NCGX6M Comdty` | 6 rows for 2026, 0 for 2016 |
| same | expired: root + month + 1-digit year + M + 2-digit year | `PNGX6M16` (2016 15.88), `PSRX1M21` (2021, 96.68), `NCGX6M16` (15.65) | 5 rows each for the historical window |

Findings that shaped the code (`contracts/futures.py`, `data/bloomberg.py`):
1. **One security, two identifiers.** While a contract is listed, the listed and expired forms name the same
   security and return identical settles (e.g. `PNGZ5M` and `PNGZ5M25`, `PNGF6M` and `PNGF6M26`). Both forms are
   requested. If both return data and agree on every common day, the expired form is kept and counted. If they
   disagree, the pull raises.
2. **Static reference data needs the expired form.** The listed 1-digit form, requested as static reference data,
   can name a later security once the contract has expired. For PEG.M.2015-12 the two forms disagreed on
   `LAST_TRADEABLE_DT`. The expired form (which carries the full year) is used whenever it has a value.
3. **Post-expiry rows are not trades.** Expired PNX securities return a flat, carried-forward value after their
   vendor last-trade date (e.g. PEG.M.2017-12: 13 rows at one value, 20.78, after 2017-11-28). These rows were
   dropped before ingest, with counts: THE 1, PEG 2,134, PSV 2,157, PVB 1,685, VTP 2,288. The raw pulls are in
   the raw layer. TTF, TTF_USD, HH, NBP, JKM and Brent have no rows after their last trade date.
4. **Tiny and repeated settles remain (not tuned).** PNX hubs carry settles below 0.5 EUR/MWh: PEG 2,205 rows
   across 61 contracts, PSV 3,425 across 62, PVB 1,665 across 42, VTP 1,682 across 45. Example: PEG.M.2018-04
   prints 0.001 on many days in December 2017. THE has none below 0.5. Repeated-value rows (no change from the
   previous day) are 527 for THE and 1,635 to 3,431 for the PNX hubs. These are plausible for thin, expiring
   contracts but need a liquidity rule before any use. Nothing has been removed for this reason.
5. **Rule mismatches (D-035 applied).** THE 9 of 159, PEG 19 of 158, PSV 4 of 151, PVB 2 of 95, VTP 7 of 122.
   `vendor_authoritative: true` is set for all five hubs in `configs/instruments.yaml`. The mismatch lists are in
   `docs/modelling_log.md` D-040 and the pull log.

## 5. What changed in the repo

- `configs/sources/bloomberg_fundamentals.yaml`: 17 new `known` series (flows, production, demand, weather
  stations). Each has provenance, unit, pit_grade B and a conservative lag.
- `configs/sources/bloomberg_lng.yaml`: 6 new `known` terminal series (send-out or regas, plus one entry flow).
- `configs/sources/bloomberg_power.yaml`: 2 new `known` Amprion zone series.
- `configs/instruments.yaml`: `vendor_authoritative: true` for HH, NBP and TTF_USD (D-039), and for THE, PEG, PSV, PVB and VTP (D-040).
- `configs/sources/bloomberg.yaml`: THE moved to root LFT with the TZT rule (D-040). PEG, PSV, PVB, VTP use `ticker_form: pnx` with monthly roots only.
- `contracts/futures.py` and `data/bloomberg.py`: `ticker_form` support, candidate tickers, and reconciliation of the two forms (D-040).
- `tests/unit/market/`: tests for the LFT rule, the PNX candidates and parse, and the resolution rules (D-040).
- `tests/unit/fundamentals/test_data_fundamentals.py`: the provenance assertion accepts "Bloomberg discovery"
  as well as "GasImbalanceModel" (intent unchanged: known series must cite their provenance).
- Not changed: `configs/local_data_sources.yaml` (git-ignored, remains `live_calls_authorised: false`), no
  model, feature or target code, no commits.

## 6. BloombergNEF European Power Market Monitor: registration and pull (2026-10-19)

Source: the user-supplied BNEF workbook (`BloombergNEF_EuropeanPower_MarketMonitor.xlsm`, git-ignored). Its ticker list
(642 rows, 523 unique tickers) and long daily table (488 tickers, 2021-02-28 to 2026-10-19, 1.23M values) were used.
The workbook itself was not opened. The identifiers are registered in `configs/sources/bloomberg_bnef.yaml`, one entry
per series, every entry reference-checked (NAME, PX_LAST, LAST_UPDATE_DT) on 2026-10-19.

**Registered and pulled: 278 series, all verified live (LAST_UPDATE_DT on or after 2025-01-01), all returned data.**
Stored rows: 1,245,058. Pull window 2000 to 2026 (inception where the series is older), year chunks, temporary authorised config.

### 6a. Summary by metric

| Metric | Series | Rows | Earliest history | Median start year | Last update |
|---|---:|---:|---|---:|---|
| Capacity (cumulative solar, wind forecast capacity) | 16 | 416 | 2000-12-31 | 2000 | 2025-12-31 |
| Carbon price support | 1 | 13 | 2013-12-31 | 2013 | 2025-12-31 |
| Gas consumption / gas for power | 5 | 26,257 | 2008-11-08 | 2012 | 2026-10-18 |
| Gas day-ahead hub prices | 9 | 34,041 | 2008-11-26 | 2012 | 2026-10-19 |
| Generation by fuel | 184 | 779,385 | 2010-03-01 | 2015 | 2026-10-18 |
| Load (day-ahead average actual) | 22 | 92,897 | 2010-03-01 | 2015 | 2026-10-18 |
| Power day-ahead auction prices | 41 | 312,049 | 2000-01-01 | 2004 | 2026-10-19 |

### 6b. Cross-check against `ticker_data_long.parquet` (overlap days, per ticker)

Rule: a day matches if |Bloomberg - BNEF| <= max(0.5% of the larger value, 0.01). 246 of 278 series overlap the BNEF table.
242 match on at least 99% of overlap days (most at 100%). Four disagree and are flagged in the registry note:

| Series | Ticker | Overlap days | Match % | Max abs diff | BBG mean | BNEF mean | Finding |
|---|---|---:|---:|---:|---:|---:|---|
| price.power_gb.da_apx_gbp_mwh | PXPXBASE Index | 2060 | 0.0 | 111.4 | 114.2 | 133.5 | Not the same series or averaging. Not used |
| price.power_pl.da_pln_mwh | PXPEIDTB Index | 2049 | 0.0 | 1543.2 | 519.2 | 116.3 | BNEF values are EUR-scale under a PLN label. Bloomberg kept (PLN) |
| price.gas_gb.da_nbp_gbp_therm | NBPGDAHD Index | 1467 | 0.0 | 581.1 | 122.9 | 1.44 | Unit or definition differs (GBp vs GBP/therm). Not used |
| carbon.gb.price_support_gbp_t | CARBPRIC Index | 5 | 0.0 | 3.8 | 18.0 | 21.0 | Five overlap days only. Insufficient to judge |

32 registered series are in the workbook ticker list but not in the long-table extract (capacity series, several
gas flows, peak and off-peak prices, LD1CNL, N2EXAM, MIBGTPDA). They have no BNEF check: Bloomberg data only.

### 6c. Not registered (listed for later)

- **Interconnector flows** (257 rows in the list): deferred per request. Available as the BNEF `Interconnector Flows` metric.
- **Spark and dark spreads, carbon and coal calculated fields** (BNEF `Calculated field` rows): no Bloomberg ticker. They are derived in BNEF's own workbook. Build them from the registered price series if needed.
- **Pumped-storage consumption** (RTEGPMPT): not requested.
- **Other fuels** (oil, waste, other thermal, peat, geothermal, oil shale, other renewable): listed in the workbook, not requested.
- **Dead tickers** (last update before 2025, excluded): GT1CBECO (Belgium coal, PX_LAST 0, last 2019-08-06), GT1CDENU (Germany nuclear, PX_LAST 0, last 2024-01-29).
- **Duplicate or alternate tickers**: where a country and fuel had two series, the `GT1C`/`GT2C` (ACT 24AV) series was registered and the RTE or other variant was left out (for example France gas CCGT/OCGT). Subtypes of gas are not registered.
- **Already registered, reused** (not duplicated): power.de.gen_solar_mw (GT1CDESL), power.de.gen_wind_on_mw (GT1CDEWS), power.de.load_act_mw (LD1CDE).

### 6d. Decision: German day-ahead power (D-041)

`price.power_de.da_eur_mwh` now points at **LPXBHRBS** (EPEX Germany day-ahead baseload index, BNEF `Commodity prices`). History
2000-06-15 to 2026-10-19, 9,623 rows. The previous target **ELGBDAHD** (OTC forward day-ahead assessment) starts 2011-02-01
(probe on 2001, 2008 and 2011-01 windows: LPXBHRBS returned rows; ELGBDAHD returned none before 2011-02). ELGBDAHD is kept
under the new id `price.power_de.da_otc_elgb_eur_mwh` with its stored history moved to that id. DEDAILY stays `requires_user_input`.

### 6e. Gaps closed by this registration

- **Gas-for-power by country**: ES (ENAGONGP, outflows for power), IT (SNAMGPGE, gas power generation), GB (NGIOPTOT, NTS power station flows), BE (FXYPHPPP, burn H-zone). BE is a burn figure, not confirmed power-only.
- **German load actual**: already registered (LD1CDE). BNEF adds load actuals for 22 other countries and zones.
- **National solar generation**: German series already registered (GT1CDESL). BNEF adds the same measure for other countries (for example power.fi.gen_solar_mw, power.lv.gen_solar_mw).
- **German day-ahead power**: closed (LPXBHRBS, see 6d).
- **Not closed**: national German load FORECAST, national solar FORECAST, French nuclear AVAILABILITY, European HDD/temperature forecasts. The BNEF workbook has no forecast metrics.

### 6f. Per-series table

| Id | Ticker | Rows | First | Last | Overlap match % |
|---|---|---:|---|---|---:|
| carbon.gb.price_support_gbp_t | CARBPRIC Index | 13 | 2013-12-31 | 2025-12-31 | 0.0 |
| gas.be.burn_hzone_gwhd | FXYPHPPP Index | 5,070 | 2012-12-01 | 2026-10-18 |  |
| gas.es.outflows_national_gwhd | ENAGONAD Index | 4,460 | 2014-08-01 | 2026-10-18 |  |
| gas.es.power_burn_gwhd | ENAGONGP Index | 4,462 | 2014-08-01 | 2026-10-18 |  |
| gas.gb.power_station_mcmd | NGIOPTOT Index | 6,554 | 2008-11-08 | 2026-10-18 |  |
| gas.it.power_burn_gwhd | SNAMGPGE Index | 5,711 | 2011-03-01 | 2026-10-18 |  |
| power.at.gen_biomass_mw | GT1CATBM ACT 24AV Index | 4,305 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.at.gen_coal_mw | GT1CATCO ACT 24AV Index | 4,307 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.at.gen_gas_mw | GT1CATGS ACT 24AV Index | 4,307 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.at.gen_hydro_res_mw | GT1CATWR ACT 24AV Index | 4,307 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.at.gen_hydro_ror_mw | GT1CATRR ACT 24AV Index | 4,308 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.at.gen_pumped_storage_mw | GT1CATPS ACT 24AV Index | 4,308 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.at.gen_solar_mw | GT1CATSL ACT 24AV Index | 4,304 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.at.gen_wind_off_mw | GT1CATWS ACT 24AV Index | 4,304 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.at.load_act_mw | LD1CAT ACT 24AV INDEX | 4,308 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.be.gen_biomass_mw | GT1CBEBM ACT 24AV Index | 4,307 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.be.gen_gas_mw | GT1CBEGS ACT 24AV Index | 4,307 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.be.gen_hydro_ror_mw | GT1CBERR ACT 24AV Index | 4,307 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.be.gen_nuclear_mw | GT1CBENU ACT 24AV Index | 4,307 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.be.gen_pumped_storage_mw | GT1CBEPS ACT 24AV Index | 4,303 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.be.gen_solar_mw | GT1CBESL ACT 24AV Index | 4,339 | 2014-12-01 | 2026-10-18 | 100.0 |
| power.be.gen_wind_off_mw | GT1CBEWO ACT 24AV Index | 4,336 | 2014-11-30 | 2026-10-18 | 100.0 |
| power.be.load_act_mw | LD1CBE ACT 24AV Index | 4,308 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.be.solar_cap_mw | SOLFTTBE Index | 26 | 2000-12-31 | 2025-12-31 |  |
| power.be.wind_off_cap_mw | WNDFW2BE Index | 26 | 2000-12-31 | 2025-12-31 |  |
| power.be.wind_on_cap_mw | WNDFW1BE Index | 26 | 2000-12-31 | 2025-12-31 |  |
| power.ch.gen_hydro_res_mw | GT1CCHWR ACT 24AV Index | 4,122 | 2015-07-01 | 2026-10-18 | 100.0 |
| power.ch.gen_hydro_ror_mw | GT1CCHRR ACT 24AV Index | 4,003 | 2015-10-13 | 2026-10-18 | 100.0 |
| power.ch.gen_nuclear_mw | GT1CCHNU ACT 24AV Index | 4,284 | 2015-01-17 | 2026-10-18 | 100.0 |
| power.ch.gen_pumped_storage_mw | GT1CCHPS ACT 24AV Index | 4,122 | 2015-07-01 | 2026-10-18 | 100.0 |
| power.ch.gen_solar_mw | GT1CCHSL ACT 24AV Index | 4,288 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.ch.gen_wind_on_mw | GT1CCHWS ACT 24AV Index | 4,291 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.ch.load_act_mw | LD1CCH ACT 24AV Index | 4,301 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.cz.gen_biomass_mw | GT1CCZBM ACT 24AV Index | 4,306 | 2014-12-27 | 2026-10-18 | 100.0 |
| power.cz.gen_coal_mw | GT1CCZCO ACT 24AV Index | 4,305 | 2014-12-27 | 2026-10-18 | 100.0 |
| power.cz.gen_gas_mw | GT1CCZGS ACT 24AV Index | 4,306 | 2014-12-27 | 2026-10-18 | 100.0 |
| power.cz.gen_hydro_res_mw | GT1CCZWR ACT 24AV Index | 4,304 | 2014-12-27 | 2026-10-18 | 100.0 |
| power.cz.gen_hydro_ror_mw | GT1CCZRR ACT 24AV Index | 4,302 | 2014-12-27 | 2026-10-18 | 100.0 |
| power.cz.gen_lignite_mw | GT1CCZLI ACT 24AV Index | 4,306 | 2014-12-27 | 2026-10-18 | 100.0 |
| power.cz.gen_nuclear_mw | GT1CCZNU ACT 24AV Index | 4,305 | 2014-12-27 | 2026-10-18 | 100.0 |
| power.cz.gen_pumped_storage_mw | GT1CCZPS ACT 24AV Index | 4,307 | 2014-12-27 | 2026-10-18 | 100.0 |
| power.cz.gen_solar_mw | GT1CCZSL ACT 24AV Index | 4,291 | 2014-12-27 | 2026-10-18 | 100.0 |
| power.cz.gen_wind_on_mw | GT1CCZWS ACT 24AV Index | 4,305 | 2014-12-27 | 2026-10-18 | 100.0 |
| power.cz.load_act_mw | LD1CCZ ACT 24AV INDEX | 4,308 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.de.gen_biomass_mw | GT1CDEBM ACT 24AV Index | 4,307 | 2015-01-01 | 2026-10-16 | 100.0 |
| power.de.gen_coal_mw | GT1CDECO ACT 24AV Index | 4,308 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.de.gen_gas_mw | GT1CDEGS ACT 24AV Index | 4,309 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.de.gen_hydro_res_mw | GT1CDEWR ACT 24AV Index | 4,309 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.de.gen_hydro_ror_mw | GT1CDERR ACT 24AV Index | 4,309 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.de.gen_lignite_mw | GT1CDELI ACT 24AV Index | 4,309 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.de.gen_pumped_storage_mw | GT1CDEPS ACT 24AV Index | 4,309 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.de.gen_wind_off_mw | GT1CDEWO ACT 24AV Index | 4,309 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.de.solar_cap_mw | SOLFTTDE Index | 26 | 2000-12-31 | 2025-12-31 |  |
| power.de.wind_off_cap_mw | WNDFW2DE Index | 26 | 2000-12-31 | 2025-12-31 |  |
| power.de.wind_on_cap_mw | WNDFW1DE Index | 26 | 2000-12-31 | 2025-12-31 |  |
| power.dk.gen_biomass_mw | GT1CDKBM ACT 24AV Index | 4,291 | 2015-01-02 | 2026-10-18 | 100.0 |
| power.dk.gen_coal_mw | GT1CDKCO ACT 24AV Index | 4,062 | 2015-01-02 | 2026-04-16 | 100.0 |
| power.dk.gen_gas_mw | GT1CDKGS ACT 24AV Index | 4,293 | 2015-01-02 | 2026-10-18 | 100.0 |
| power.dk.gen_solar_mw | GT1CDKSL ACT 24AV Index | 4,190 | 2015-01-02 | 2026-10-18 | 100.0 |
| power.dk.gen_wind_off_mw | GT1CDKWO ACT 24AV Index | 4,245 | 2015-01-02 | 2026-10-18 | 100.0 |
| power.dk.gen_wind_on_mw | GT1CDKWS ACT 24AV Index | 4,296 | 2015-01-02 | 2026-10-18 | 100.0 |
| power.dk.load_act_mw | LD1CDK ACT 24AV Index | 3,919 | 2016-01-01 | 2026-10-04 | 100.0 |
| power.ee.gen_biomass_mw | GT2CEEBM ACT 24AV Index | 4,054 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.ee.gen_gas_mw | GT2CEEGS ACT 24AV Index | 4,055 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.ee.gen_hydro_ror_mw | GT2CEERR ACT 24AV Index | 4,055 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.ee.gen_solar_mw | GT2CEESL ACT 24AV Index | 4,079 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.ee.gen_wind_on_mw | GT2CEEWS ACT 24AV Index | 4,048 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.ee.load_act_mw | LD2CEE ACT 24AV Index | 3,806 | 2016-01-01 | 2026-10-18 | 100.0 |
| power.es.gen_biomass_mw | GT1CESBM ACT 24AV Index | 4,282 | 2014-12-20 | 2026-10-18 | 100.0 |
| power.es.gen_coal_mw | GT1CESCO ACT 24AV Index | 4,286 | 2014-12-20 | 2026-10-18 | 100.0 |
| power.es.gen_gas_mw | GT1CESGS ACT 24AV Index | 4,284 | 2014-12-20 | 2026-10-18 | 100.0 |
| power.es.gen_hydro_res_mw | GT1CESWR ACT 24AV Index | 4,283 | 2014-12-20 | 2026-10-18 | 100.0 |
| power.es.gen_hydro_ror_mw | GT1CESRR ACT 24AV Index | 4,283 | 2014-12-20 | 2026-10-18 | 100.0 |
| power.es.gen_lignite_mw | GT1CESLI ACT 24AV Index | 4,283 | 2014-12-20 | 2026-10-18 | 100.0 |
| power.es.gen_nuclear_mw | GT1CESNU ACT 24AV Index | 4,285 | 2014-12-20 | 2026-10-18 | 100.0 |
| power.es.gen_pumped_storage_mw | GT1CESPS ACT 24AV Index | 2,094 | 2021-01-01 | 2026-10-18 | 100.0 |
| power.es.gen_solar_mw | GT1CESSL ACT 24AV Index | 4,283 | 2014-12-20 | 2026-10-18 | 100.0 |
| power.es.gen_wind_off_mw | GT1CESWO ACT 24AV Index | 4,272 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.es.gen_wind_on_mw | GT1CESWS ACT 24AV Index | 4,284 | 2014-12-20 | 2026-10-18 | 100.0 |
| power.es.load_act_mw | LD1CES ACT 24AV Index | 4,296 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.es.solar_cap_mw | SOLFTTES Index | 26 | 2000-12-31 | 2025-12-31 |  |
| power.es.wind_on_cap_mw | WNDFW1ES Index | 26 | 2000-12-31 | 2025-12-31 |  |
| power.fi.gen_biomass_mw | GT2CFIBM ACT 24AV Index | 4,261 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.fi.gen_coal_mw | GT2CFICO ACT 24AV Index | 4,264 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.fi.gen_gas_mw | GT2CFIGS ACT 24AV Index | 4,261 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.fi.gen_hydro_ror_mw | GT2CFIRR ACT 24AV Index | 4,260 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.fi.gen_nuclear_mw | GT2CFINU ACT 24AV Index | 4,262 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.fi.gen_solar_mw | GT2CFISL ACT 24AV Index | 1,158 | 2023-05-26 | 2026-10-18 | 100.0 |
| power.fi.gen_wind_on_mw | GT2CFIWS ACT 24AV Index | 4,257 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.fi.load_act_mw | LD2CFI ACT 24AV Index | 3,910 | 2016-01-01 | 2026-10-18 | 100.0 |
| power.fr.gen_biomass_mw | RTEGTBGT Index | 5,032 | 2013-01-01 | 2026-10-18 | 100.0 |
| power.fr.gen_coal_mw | RTEGCOAT Index | 5,392 | 2012-01-01 | 2026-10-18 | 100.0 |
| power.fr.gen_gas_mw | RTEGGAST Index | 5,398 | 2012-01-01 | 2026-10-18 | 100.0 |
| power.fr.gen_hydro_res_mw | RTEGHLKT Index | 5,032 | 2013-01-01 | 2026-10-18 | 100.0 |
| power.fr.gen_hydro_ror_mw | RTEGHRRT Index | 5,033 | 2013-01-01 | 2026-10-18 | 100.0 |
| power.fr.gen_nuclear_mw | RTEGNUCT Index | 5,398 | 2012-01-01 | 2026-10-18 | 100.0 |
| power.fr.gen_pumped_storage_mw | RTEGHPHT Index | 5,032 | 2013-01-01 | 2026-10-18 | 100.0 |
| power.fr.gen_solar_mw | RTEGSLRT Index | 5,397 | 2012-01-01 | 2026-10-18 | 100.0 |
| power.fr.gen_wind_on_mw | RTEGWNDT Index | 5,392 | 2012-01-01 | 2026-10-18 | 100.0 |
| power.fr.solar_cap_mw | SOLFTTFR Index | 26 | 2000-12-31 | 2025-12-31 |  |
| power.fr.wind_on_cap_mw | WNDFW1FR Index | 26 | 2000-12-31 | 2025-12-31 |  |
| power.gb.gen_biomass_mw | GBDGBIOM Index | 3,220 | 2017-12-07 | 2026-10-18 | 100.0 |
| power.gb.gen_coal_mw | GBDGCOAL Index | 5,013 | 2012-09-28 | 2026-10-18 | 100.0 |
| power.gb.gen_gas_mw | GBDGCCGT Index | 5,045 | 2012-09-28 | 2026-10-18 | 100.0 |
| power.gb.gen_hydro_res_mw | GBDGHYDR Index | 5,048 | 2012-09-28 | 2026-10-18 | 100.0 |
| power.gb.gen_nuclear_mw | GBDGNUCL Index | 5,048 | 2012-09-28 | 2026-10-18 | 100.0 |
| power.gb.gen_pumped_storage_mw | GBDGPUMP Index | 5,034 | 2012-09-28 | 2026-10-18 | 100.0 |
| power.gb.gen_solar_mw | GBDGSOLA Index | 2,653 | 2019-07-01 | 2026-10-18 | 100.0 |
| power.gb.gen_wind_on_mw | GBDGWIND Index | 5,048 | 2012-09-28 | 2026-10-18 | 100.0 |
| power.gb.solar_cap_mw | SOLFTTGB Index | 26 | 2000-12-31 | 2025-12-31 |  |
| power.gb.wind_off_cap_mw | WNDFW2GB Index | 26 | 2000-12-31 | 2025-12-31 |  |
| power.gb.wind_on_cap_mw | WNDFW1GB Index | 26 | 2000-12-31 | 2025-12-31 |  |
| power.gr.gen_gas_mw | GT2CGRGS ACT 24AV Index | 4,199 | 2014-12-30 | 2026-10-08 | 100.0 |
| power.gr.gen_lignite_mw | GT2CGRLI ACT 24AV Index | 3,990 | 2015-07-27 | 2026-10-08 | 100.0 |
| power.gr.gen_solar_mw | GT2CGRSL ACT 24AV Index | 4,243 | 2014-12-29 | 2026-10-18 | 100.0 |
| power.gr.gen_wind_on_mw | GT2CGRWS ACT 24AV Index | 4,246 | 2014-12-29 | 2026-10-18 | 100.0 |
| power.gr.load_act_mw | LD2CGR ACT 24AV Index | 3,880 | 2016-01-01 | 2026-10-18 | 100.0 |
| power.hu.gen_biomass_mw | GT1CHUBM ACT 24AV Index | 4,303 | 2014-12-23 | 2026-10-18 | 100.0 |
| power.hu.gen_gas_mw | GT1CHUGS ACT 24AV Index | 4,303 | 2014-12-23 | 2026-10-18 | 100.0 |
| power.hu.gen_hydro_res_mw | GT1CHUWR ACT 24AV Index | 4,298 | 2014-12-23 | 2026-10-18 | 100.0 |
| power.hu.gen_hydro_ror_mw | GT1CHURR ACT 24AV Index | 4,293 | 2014-12-23 | 2026-10-18 | 100.0 |
| power.hu.gen_lignite_mw | GT1CHULI ACT 24AV Index | 4,304 | 2014-12-23 | 2026-10-18 | 100.0 |
| power.hu.gen_nuclear_mw | GT1CHUNU ACT 24AV Index | 4,303 | 2014-12-23 | 2026-10-18 | 100.0 |
| power.hu.gen_solar_mw | GT1CHUSL ACT 24AV Index | 2,582 | 2019-09-23 | 2026-10-18 | 100.0 |
| power.hu.gen_wind_on_mw | GT1CHUWS ACT 24AV Index | 4,304 | 2014-12-23 | 2026-10-18 | 100.0 |
| power.hu.load_act_mw | LD1CHU ACT 24AV Index | 4,305 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.it.gen_biomass_mw | GT1CITBM ACT 24AV Index | 3,954 | 2015-01-04 | 2026-10-18 | 100.0 |
| power.it.gen_coal_mw | GT1CITCG ACT 24AV Index | 3,889 | 2015-10-25 | 2026-09-24 | 100.0 |
| power.it.gen_gas_mw | GT1CITGS ACT 24AV Index | 3,941 | 2015-10-25 | 2026-10-18 | 100.0 |
| power.it.gen_hydro_res_mw | GT1CITWR ACT 24AV Index | 3,927 | 2015-10-25 | 2026-10-18 | 100.0 |
| power.it.gen_hydro_ror_mw | GT1CITRR ACT 24AV Index | 3,940 | 2015-10-25 | 2026-10-18 | 100.0 |
| power.it.gen_pumped_storage_mw | GT1CITPS ACT 24AV Index | 3,595 | 2015-10-25 | 2025-12-31 | 100.0 |
| power.it.gen_solar_mw | GT1CITSL ACT 24AV Index | 4,028 | 2015-10-02 | 2026-10-18 | 100.0 |
| power.it.gen_wind_on_mw | GT1CITWS ACT 24AV Index | 3,937 | 2015-10-25 | 2026-10-18 | 100.0 |
| power.it.load_act_mw | LD1CIT ACT 24AV Index | 4,309 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.it.solar_commercial_cap_mw | SOLFS2IT Index | 26 | 2000-12-31 | 2025-12-31 |  |
| power.it.solar_utility_cap_mw | SOLFS3IT Index | 26 | 2000-12-31 | 2025-12-31 |  |
| power.it.wind_on_cap_mw | WNDFW1IT Index | 26 | 2000-12-31 | 2025-12-31 |  |
| power.lt.gen_biomass_mw | GT2CLTBM ACT 24AV Index | 4,107 | 2015-01-05 | 2026-10-18 | 100.0 |
| power.lt.gen_gas_mw | GT2CLTGS ACT 24AV Index | 4,108 | 2015-01-05 | 2026-10-18 | 100.0 |
| power.lt.gen_hydro_ror_mw | GT2CLTRR ACT 24AV Index | 4,109 | 2015-01-05 | 2026-10-18 | 100.0 |
| power.lt.gen_pumped_storage_mw | GT2CLTPS ACT 24AV Index | 4,108 | 2015-01-05 | 2026-10-18 | 100.0 |
| power.lt.gen_solar_mw | GT2CLTSL ACT 24AV Index | 4,108 | 2015-01-05 | 2026-10-18 | 100.0 |
| power.lt.gen_wind_on_mw | GT2CLTWS ACT 24AV Index | 4,108 | 2015-01-05 | 2026-10-18 | 100.0 |
| power.lt.load_act_mw | LD2CLT ACT 24AV Index | 3,927 | 2016-01-01 | 2026-10-18 | 100.0 |
| power.lv.gen_biomass_mw | GT2CLVBM ACT 24AV Index | 4,239 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.lv.gen_gas_mw | GT2CLVGS ACT 24AV Index | 4,240 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.lv.gen_hydro_ror_mw | GT2CLVRR ACT 24AV Index | 4,160 | 2015-03-26 | 2026-10-18 | 100.0 |
| power.lv.gen_solar_mw | GT2CLVSL ACT 24AV Index | 1,009 | 2024-01-01 | 2026-10-18 | 100.0 |
| power.lv.gen_wind_on_mw | GT2CLVWS ACT 24AV Index | 4,241 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.lv.load_act_mw | LD2CLV ACT 24AV Index | 3,936 | 2016-01-01 | 2026-10-18 | 100.0 |
| power.nl.gen_biomass_mw | GT1CNLBM ACT 24AV Index | 4,304 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.nl.gen_coal_mw | GT1CNLCO ACT 24AV Index | 4,302 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.nl.gen_gas_mw | GT1CNLGS ACT 24AV Index | 4,307 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.nl.gen_nuclear_mw | GT1CNLNU ACT 24AV Index | 4,305 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.nl.gen_solar_mw | GT1CNLSL ACT 24AV Index | 4,305 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.nl.gen_wind_off_mw | GT1CNLWO ACT 24AV Index | 4,302 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.nl.gen_wind_on_mw | GT1CNLWS ACT 24AV Index | 4,302 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.nl.load_act_mw | LD1CNL ACT 24AV Index | 4,307 | 2015-01-01 | 2026-10-18 |  |
| power.no.gen_gas_mw | GT1CNOGS ACT 24AV Index | 4,237 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.no.gen_hydro_res_mw | GT1CNOWR ACT 24AV Index | 4,255 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.no.gen_hydro_ror_mw | GT1CNORR ACT 24AV Index | 4,257 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.no.gen_pumped_storage_mw | GT1CNOPS ACT 24AV Index | 2,005 | 2020-07-02 | 2026-09-27 | 100.0 |
| power.no.gen_wind_on_mw | GT1CNOWS ACT 24AV Index | 4,254 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.no.load_act_mw | LD1CNO ACT 24AV Index | 3,908 | 2016-01-01 | 2026-10-18 | 100.0 |
| power.nordpool.gen_biomass_mw | GT2CNPBM ACT 24AV Index | 3,662 | 2015-01-05 | 2026-10-18 | 100.0 |
| power.nordpool.gen_coal_mw | GT2CNPCO ACT 24AV Index | 3,964 | 2015-01-03 | 2026-04-16 | 100.0 |
| power.nordpool.gen_gas_mw | GT2CNPGS ACT 24AV Index | 3,637 | 2015-01-05 | 2026-10-18 | 100.0 |
| power.nordpool.gen_hydro_res_mw | GT2CNPWR ACT 24AV Index | 4,209 | 2015-01-05 | 2026-10-18 | 100.0 |
| power.nordpool.gen_hydro_ror_mw | GT2CNPRR ACT 24AV Index | 3,573 | 2015-03-26 | 2026-10-18 | 100.0 |
| power.nordpool.gen_nuclear_mw | GT2CNPNU ACT 24AV Index | 4,154 | 2015-01-05 | 2026-10-18 | 100.0 |
| power.nordpool.gen_pumped_storage_mw | GT2CNPPS ACT 24AV Index | 3,804 | 2015-01-05 | 2026-09-27 | 100.0 |
| power.nordpool.gen_solar_mw | GT2CNPSL ACT 24AV Index | 3,708 | 2015-01-05 | 2026-10-18 | 100.0 |
| power.nordpool.gen_wind_off_mw | GT2CNPWO ACT 24AV Index | 4,229 | 2015-01-02 | 2026-10-18 | 100.0 |
| power.nordpool.gen_wind_on_mw | GT2CNPWS ACT 24AV Index | 3,599 | 2015-01-05 | 2026-10-18 | 100.0 |
| power.nordpool.load_act_mw | NO1CNPTT Index | 4,222 | 2013-06-22 | 2025-02-04 | 100.0 |
| power.pl.gen_biomass_mw | GT1CPLBM ACT 24AV Index | 4,306 | 2015-01-02 | 2026-10-18 | 100.0 |
| power.pl.gen_coal_mw | GT1CPLCO ACT 24AV Index | 4,306 | 2015-01-02 | 2026-10-18 | 100.0 |
| power.pl.gen_gas_mw | GT1CPLGS ACT 24AV Index | 4,306 | 2015-01-02 | 2026-10-18 | 100.0 |
| power.pl.gen_hydro_res_mw | GT1CPLWR ACT 24AV Index | 4,304 | 2015-01-02 | 2026-10-18 | 100.0 |
| power.pl.gen_hydro_ror_mw | GT1CPLRR ACT 24AV Index | 4,305 | 2015-01-02 | 2026-10-18 | 100.0 |
| power.pl.gen_lignite_mw | GT1CPLLI ACT 24AV Index | 4,306 | 2015-01-02 | 2026-10-18 | 100.0 |
| power.pl.gen_pumped_storage_mw | GT1CPLPS ACT 24AV Index | 4,305 | 2015-01-02 | 2026-10-18 | 100.0 |
| power.pl.gen_solar_mw | GT1CPLSL ACT 24AV Index | 2,379 | 2020-04-10 | 2026-10-18 | 100.0 |
| power.pl.gen_wind_on_mw | GT1CPLWS ACT 24AV Index | 4,307 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.pl.load_act_mw | LD1CPL ACT 24AV INDEX | 4,308 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.pt.gen_biomass_mw | GT0CPTBM ACT 24AV Index | 4,331 | 2014-12-09 | 2026-10-18 | 100.0 |
| power.pt.gen_coal_mw | GT0CPTCO ACT 24AV Index | 4,331 | 2014-12-09 | 2026-10-18 | 100.0 |
| power.pt.gen_gas_mw | GT0CPTGS ACT 24AV Index | 4,331 | 2014-12-09 | 2026-10-18 | 100.0 |
| power.pt.gen_hydro_res_mw | GT0CPTWR ACT 24AV Index | 4,331 | 2014-12-09 | 2026-10-18 | 100.0 |
| power.pt.gen_hydro_ror_mw | GT0CPTRR ACT 24AV Index | 4,331 | 2014-12-09 | 2026-10-18 | 100.0 |
| power.pt.gen_pumped_storage_mw | GT0CPTPS ACT 24AV Index | 4,330 | 2014-12-09 | 2026-10-18 | 100.0 |
| power.pt.gen_solar_mw | GT0CPTSL ACT 24AV Index | 4,330 | 2014-12-09 | 2026-10-18 | 100.0 |
| power.pt.gen_wind_off_mw | GT0CPTWO ACT 24AV Index | 2,330 | 2020-06-01 | 2026-10-18 | 100.0 |
| power.pt.gen_wind_on_mw | GT0CPTWS ACT 24AV Index | 4,331 | 2014-12-09 | 2026-10-18 | 100.0 |
| power.pt.load_act_mw | LD0CPT ACT 24AV INDEX | 4,309 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.ro.gen_biomass_mw | RPFCBFAV Index | 4,770 | 2013-08-01 | 2026-10-17 | 100.0 |
| power.ro.gen_coal_mw | RPFCAPCL Index | 5,882 | 2010-03-01 | 2026-10-17 | 100.0 |
| power.ro.gen_gas_mw | RPFCAPGS Index | 5,882 | 2010-03-01 | 2026-10-17 | 100.0 |
| power.ro.gen_hydro_mw | RPFCAPHY Index | 5,882 | 2010-03-01 | 2026-10-17 | 100.0 |
| power.ro.gen_nuclear_mw | RPFCAPNC Index | 5,882 | 2010-03-01 | 2026-10-17 | 100.0 |
| power.ro.gen_solar_mw | RPFCAPSL Index | 4,933 | 2013-03-01 | 2026-10-17 | 100.0 |
| power.ro.gen_wind_mw | RPFCAPWD Index | 5,874 | 2010-03-01 | 2026-10-17 | 100.0 |
| power.ro.load_act_mw | RPFCCOAV Index | 5,882 | 2010-03-01 | 2026-10-17 | 100.0 |
| power.se.gen_hydro_res_mw | GT1CSEWR ACT 24AV Index | 4,280 | 2015-01-05 | 2026-10-18 | 100.0 |
| power.se.gen_nuclear_mw | GT1CSENU ACT 24AV Index | 4,274 | 2015-01-05 | 2026-10-18 | 100.0 |
| power.se.gen_wind_on_mw | GT1CSEWS ACT 24AV Index | 4,283 | 2015-01-05 | 2026-10-18 | 100.0 |
| power.se.load_act_mw | LD1CSE ACT 24AV Index | 3,943 | 2016-01-01 | 2026-10-18 | 100.0 |
| power.si.gen_biomass_mw | GT1CSIBM ACT 24AV Index | 3,540 | 2017-01-01 | 2026-10-18 | 100.0 |
| power.si.gen_gas_mw | GT1CSIGS ACT 24AV Index | 4,272 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.si.gen_hydro_ror_mw | GT1CSIRR ACT 24AV Index | 4,270 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.si.gen_lignite_mw | GT1CSILI ACT 24AV Index | 4,270 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.si.gen_nuclear_mw | GT1CSINU ACT 24AV Index | 4,268 | 2015-01-01 | 2026-10-17 | 100.0 |
| power.si.gen_pumped_storage_mw | GT1CSIPS ACT 24AV Index | 4,268 | 2015-01-01 | 2026-10-17 | 100.0 |
| power.si.gen_solar_mw | GT1CSISL ACT 24AV Index | 4,283 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.si.gen_wind_on_mw | GT1CSIWS ACT 24AV Index | 4,268 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.si.load_act_mw | LD1CSI ACT 24AV Index | 4,237 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.sk.gen_biomass_mw | GT1CSKBM ACT 24AV Index | 4,161 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.sk.gen_coal_mw | GT1CSKCO ACT 24AV Index | 4,161 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.sk.gen_gas_mw | GT1CSKGS ACT 24AV Index | 4,162 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.sk.gen_hydro_res_mw | GT1CSKWR ACT 24AV Index | 4,161 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.sk.gen_hydro_ror_mw | GT1CSKRR ACT 24AV Index | 4,160 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.sk.gen_lignite_mw | GT1CSKLI ACT 24AV Index | 4,159 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.sk.gen_nuclear_mw | GT1CSKNU ACT 24AV Index | 4,161 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.sk.gen_pumped_storage_mw | GT1CSKPS ACT 24AV Index | 4,158 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.sk.gen_solar_mw | GT1CSKSL ACT 24AV Index | 4,167 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.sk.gen_wind_on_mw | GT1CSKWS ACT 24AV Index | 4,167 | 2015-01-01 | 2026-10-18 | 100.0 |
| power.sk.load_act_mw | LD1CSK ACT 24AV Index | 4,268 | 2015-01-01 | 2026-10-18 | 100.0 |
| price.gas_at.da_vtp_eur_mwh | PEGAETDA Index | 2,660 | 2016-11-21 | 2026-10-19 | 100.0 |
| price.gas_cz.da_vtp_eur_mwh | EGSIVTED Index | 3,041 | 2017-12-08 | 2026-10-18 | 100.0 |
| price.gas_de.da_thc_eur_mwh | EEDANCG1 Index | 4,057 | 2012-12-13 | 2026-10-19 | 100.0 |
| price.gas_es.da_pvb_ref_eur_mwh | MIBGRPDA Index | 3,934 | 2015-12-16 | 2026-10-18 | 100.0 |
| price.gas_es.da_pvb_traded_eur_mwh | MIBGTPDA Index | 3,934 | 2015-12-16 | 2026-10-18 |  |
| price.gas_fr.da_peg_eur_mwh | PNXIGNDA Index | 4,889 | 2008-11-26 | 2026-10-19 | 100.0 |
| price.gas_gb.da_nbp_gbp_therm | NBPGDAHD Index | 3,900 | 2011-02-01 | 2026-10-19 | 0.0 |
| price.gas_it.da_psv_eur_mwh | PSVGDAHD Index | 3,560 | 2011-10-07 | 2026-10-19 | 100.0 |
| price.gas_nl.da_ttf_fwd_eur_mwh | TTFGDAHD Index | 4,066 | 2011-02-01 | 2026-10-19 | 100.0 |
| price.power_at.da_eur_mwh | ATPWHRBS Index | 2,942 | 2018-09-30 | 2026-10-19 | 100.0 |
| price.power_be.da_eur_mwh | DABEBASE Index | 7,267 | 2006-11-21 | 2026-10-19 | 100.0 |
| price.power_ch.da_eur_mwh | LSWBHRBS Index | 7,246 | 2006-12-11 | 2026-10-19 | 100.0 |
| price.power_cz.da_eur_mwh | OTIXBSCZ Index | 5,965 | 2010-02-07 | 2026-10-19 | 100.0 |
| price.power_de.da_eur_mwh | LPXBHRBS Index | 9,623 | 2000-06-15 | 2026-10-19 | 100.0 |
| price.power_de.da_offpeak1_eur_mwh | LPXBHOP1 Index | 9,621 | 2000-06-15 | 2026-10-19 |  |
| price.power_de.da_peak_eur_mwh | LPXBHRPK Index | 9,622 | 2000-06-15 | 2026-10-19 |  |
| price.power_dk1.da_eur_mwh | ENDSODAV Index | 9,787 | 2000-01-01 | 2026-10-19 | 100.0 |
| price.power_dk2.da_eur_mwh | CPHE24AV Index | 9,516 | 2000-09-28 | 2026-10-19 | 100.0 |
| price.power_ee.da_eur_mwh | ESTPPXAV Index | 5,859 | 2010-10-03 | 2026-10-19 | 100.0 |
| price.power_es.da_eur_mwh | OMLPDAHD Index | 9,789 | 2000-01-01 | 2026-10-19 | 100.0 |
| price.power_fi.da_eur_mwh | ENNSHEAV Index | 9,789 | 2000-01-01 | 2026-10-19 | 100.0 |
| price.power_fr.da_eur_mwh | PWNXFRAV Index | 9,094 | 2001-11-26 | 2026-10-19 | 100.0 |
| price.power_fr.da_offpeak1_eur_mwh | PWNXFOP1 Index | 9,094 | 2001-11-26 | 2026-10-19 |  |
| price.power_fr.da_peak_eur_mwh | PWNXFRPK Index | 9,094 | 2001-11-26 | 2026-10-19 |  |
| price.power_gb.da_apx_gbp_mwh | PXPXBASE Index | 5,929 | 2008-12-02 | 2026-10-19 | 0.0 |
| price.power_gb.da_apx_offpeak_gbp_mwh | PXPXOFPK Index | 5,980 | 2008-12-02 | 2026-10-19 |  |
| price.power_gb.da_apx_peak_gbp_mwh | PXPXPEAK Index | 5,915 | 2008-12-02 | 2026-10-19 |  |
| price.power_gb.da_gbp_mwh | N2EXAM Index | 6,125 | 2010-01-12 | 2026-10-19 |  |
| price.power_hu.da_eur_mwh | HUDABASE Index | 5,551 | 2011-08-09 | 2026-10-19 | 100.0 |
| price.power_it.da_eur_mwh | GMELITBS Index | 7,971 | 2004-12-23 | 2026-10-19 | 100.0 |
| price.power_lt.da_eur_mwh | OLITEURP Index | 5,236 | 2012-06-17 | 2026-10-19 | 100.0 |
| price.power_lv.da_eur_mwh | LVNPEUAG Index | 4,886 | 2013-06-02 | 2026-10-19 | 100.0 |
| price.power_nl.da_eur_mwh | AELCTDAY Index | 9,719 | 2000-01-03 | 2026-10-19 | 100.0 |
| price.power_nl.da_offpeak_eur_mwh | AELCTMOF Index | 8,493 | 2000-01-03 | 2026-10-19 |  |
| price.power_nl.da_peak_eur_mwh | AELCTMON Index | 8,493 | 2000-01-03 | 2026-10-19 |  |
| price.power_no1.da_eur_mwh | ENOSOSAV Index | 9,788 | 2000-01-01 | 2026-10-19 | 100.0 |
| price.power_no2.da_eur_mwh | ENOSKRAV Index | 9,788 | 2000-01-01 | 2026-10-19 | 100.0 |
| price.power_no3.da_eur_mwh | ENOSTRAV Index | 9,787 | 2000-01-02 | 2026-10-19 | 100.0 |
| price.power_no4.da_eur_mwh | ENOSTMAV Index | 9,785 | 2000-01-02 | 2026-10-19 | 100.0 |
| price.power_no5.da_eur_mwh | ENOSBGAV Index | 9,787 | 2000-01-02 | 2026-10-19 | 100.0 |
| price.power_nordpool.da_eur_mwh | ENWSSPAV Index | 9,789 | 2000-01-01 | 2026-10-19 | 100.0 |
| price.power_pl.da_pln_mwh | PXPEIDTB Index | 3,935 | 2015-12-31 | 2026-10-19 | 0.0 |
| price.power_pt.da_eur_mwh | OMPTDAHD Index | 7,046 | 2007-06-29 | 2026-10-19 | 100.0 |
| price.power_ro.da_eur_mwh | ROPXDAMB Index | 4,582 | 2014-03-25 | 2026-10-19 | 100.0 |
| price.power_se1.da_eur_mwh | ENWSS1AV Index | 5,766 | 2011-01-02 | 2026-10-19 | 100.0 |
| price.power_se2.da_eur_mwh | ENWSS2AL Index | 5,766 | 2011-01-02 | 2026-10-19 | 100.0 |
| price.power_se3.da_eur_mwh | ENWSS3AL Index | 9,787 | 2000-01-02 | 2026-10-19 | 100.0 |
| price.power_se4.da_eur_mwh | ENWSS4AL Index | 5,766 | 2011-01-02 | 2026-10-19 | 100.0 |
| price.power_si.da_eur_mwh | BSPEBASE Index | 5,755 | 2008-11-19 | 2026-02-26 | 100.0 |
| price.power_sk.da_eur_mwh | OTIXBSSK Index | 6,316 | 2009-07-01 | 2026-10-19 | 100.0 |

## 7. Weather through BQL weather(), via the user's Excel (D-042, 2026-10-19)

Source: `weather()` through the user's Excel BQL add-in (`=BQL.QUERY(...)` on scratch books, closed unsaved; see
`src/ttf_forecaster/data/bql_excel.py`). The Python API login is not entitled to BQL (`//blp/bqlsvc` refuses), so the
Excel route is the one used. Registry: `configs/sources/bloomberg_weather.yaml` (204 series, `ticker: null`, so the ticker
adapter refuses them). Code: `data/weather_bql.py`, `pipelines/bql_weather.py`.

**Lead mapping (proved, pinned in tests).** `as_of_date = D`, `offset = range(1,16)`: row `i` of a location's block is the
valid day `D + i`. A single offset `k` returns the same value as row `k-1` of the range query, and the dates audit confirms
the valid days. Each pull audits the dates of its first as-of (valid = as-of + 0..14) before storing.

**Coverage stored (git-ignored `data/live`).**
- Forecasts: every daily as-of from 2025-01-01 to 2026-10-19 (650 issue days), ECMWF and GFS, 0z run. Locations DE, NL, FR,
  GB, IT, BE, AT, PL, plus NWE, CEN, SEU, EEU. Fields HDD_18C, CDD_18C, TEMPERATURE (`t2m_mean`), WIND_SPEED and
  SOLAR_RADIATION. Models: ECMWF mean, deterministic, min, max, median; GFS mean and deterministic. 4,168,500 rows after
  removing duplicate raw deliveries. Batches: 41 for the resumed run, 0 failures, 9,844 s for the resumed 2025-09-08..2026-10-19
  stretch. Earlier stretches ran before it.
- Actuals: daily, 2011-01-01 to the last published day (2026). **Re-pulled after the date fix (D-045).** The first pull labelled
  each value with the requested day, one day early. BQL labels the value with the NEXT day, so the stored event time is now
  the observed day (taken from a dates-only query). Each year is checked against the WSI regional series before it is stored.
- Normals: `NORMALS_10Y` and `NORMALS_30Y`, 17,520 rows per source. **Grade C** (hindsight): they never enter a clean backtest.

**Units (ASSUMED from the BQL convention, to check on the Terminal):** HDD and CDD in degC-day, temperature in degC, wind in
m/s, solar radiation in W/m2. Solar radiation is not returned for actuals or normals.

**Known data facts.**
- **Lead 0 is empty for HDD, CDD and solar**: every lead-0 row of those fields is null. About 1 forecast row in 15 is null
  (exactly the lead-0 rows). Temperature and wind are present at lead 0.
- GFS returns 16 rows per block, not 15; the parser accepts 15 or 16 and checks the ID per location.
- A date field and a value field requested in one `get` do not align, so the dates are audited on their own.

**Point-in-time assumptions.** Forecast issue = as-of day 00:00 UTC; publication = 10:00 UTC on that day (**ASSUMED**,
logged as D-042 with sensitivity). Actuals grade B, published D+1 at 12:00 UTC. Forecasts grade A. Normals grade C.

**Actuals date fix (D-045).** The original actuals were stored one day early for DE (labels per location now; GB is labelled on the request day). Quarantined, not deleted:
`data/live/quarantine/bql_weather_obs_wrong_day_2026-10-19/` (git-ignored). Rows re-pulled with the observed day.

**Sanity checks (2026-10-19, stored raw data, 650 issue days).**
- DE HDD18, ECMWF mean vs actual, RMSE by lead: 1.73 at 24 h, 2.47 at 240 h, 2.61 at 336 h. It grows with lead, as it should.
- DE temperature, ECMWF mean, RMSE by lead: 2.12 at lead 0, 2.12 at 24 h, 2.90 at 240 h, 3.15 at 336 h. Lead 0 has the smallest
  error, consistent with row 0 being the as-of day.
- DE HDD actual mean by month: January 15.5, July 0.5. Seasonality is right.

**Backtest consequence (short).** Vintaged forecasts cover about 1.8 heating seasons (2025-01-01 to 2026-10-19). That is enough
for a short H2 backtest only. Daily accumulation continues through `scripts/run_daily_snapshot.py` (section 8), which runs
this extractor for today's as-of.

## 8. BSRCH power-plant outages, ASSET_OUTAGE:POWER_PLANT (D-043, 2026-10-19)

Source: the user's Excel form `=@BSRCH("ASSET_OUTAGE:POWER_PLANT", "LOCATION=..", "START_DATE=..", "END_DATE=..",
"AS_OF_DATE=..", "STACK=FALSE")`, called from Python through `xbbg.blp.bsrch` (live path, `known` series only, temporary
authorised config). Registry: `configs/sources/bloomberg_outages.yaml`, 32 series. Code: `data/outages.py`,
`pipelines/outages.py`, `pipelines/daily_snapshot.py`, `scripts/run_daily_snapshot.py`.

Measured on 2026-10-19:

| Question | Answer |
|---|---|
| Locations with data | FR, DE, NORDPOOL. BE, NL, GB, NO, SE, ES, IT return 0 rows |
| Columns (STACK=FALSE) | one row per applicable day, unavailable MW per fuel. FR: Biomass, Coal, Hydro, Natural Gas, Nuclear, Oil, Other, Storage, Wind. DE adds Lignite, Other Gas, Solar, Waste. NORDPOOL adds Solar, Waste |
| Plan horizon | one as-of gives the plan to 2028-12-31 (805 to 811 rows) |
| Window limits | a window longer than about three years returns 0 rows (2025-10-13..2028-12-31 gave none; 2026-01-01..2028-12-31 gave 1,096). A snapshot is therefore two requests: realised (as-of minus 365 days to as-of minus 1) and forward (as-of to 2028-12-31) |
| Realised history | rolling one year before the as-of date. An earlier start is cut to the as-of minus 365 days |
| As-of retention | AS_OF_DATE returns data from **2026-10-13**. 2026-10-12 and earlier return nothing |
| Vintages | real. The same future day differs between as-of 2026-10-13 and 2026-10-19 (FR Nuclear, 2026-10-19: 15,675 vs 17,125 MW). Realised days can also be revised (FR Hydro, 2025-10-19) |
| Gas-asset domains | ASSET_OUTAGE:GAS, PIPELINE, GAS_FIELD, LNG, LNG_TERMINAL, GAS_STORAGE return 0 rows. An unknown domain also returns 0 rows with no error, so domains cannot be discovered from the API. None is invented. **Gas outages are not on this path** |
| Fleet capacity | no FR nuclear capacity series is in the registry (only solar and wind capacity from BNEF). **Superseded by D-044**: the fleet constant is ASSUMED (63,000 MW). `power.fr.nuclear_avail_*` now uses it |

What is stored (per location and as-of `A`, request time `R`):

* **Forward plan**, applicable day `D >= A`: FORECAST_SCHEMA rows. `issue_time` = `A` 00:00 UTC (the vintage), `publication_time` = `R`
  (the conservative stamp: Bloomberg gives no publication time), grade A, `lead_hours` = `D - A`.
* **Realised days**, `D < A`: OBSERVATION_SCHEMA rows, grade B. `publication_time` = max(`D+1` 12:00 UTC, `R`). `revision` = the
  as-of date's offset from 2026-10-13, so a later snapshot that revises a day is a new revision, not a conflict.
  Realised values seen in the 2026-10-13 and 2026-10-19 snapshots differ for some days, so the rule matters.
* Series id `power.<geo>.outage_<fuel>_unavail_mw`. The same id carries forecast and observation rows.
* Raw batches are named `bsrch_outage_<geo>_asof<A>_req<R date>_{fc,obs}`, so a re-run on the same day is skipped.

Stored on 2026-10-19 (git-ignored `data/live`):

* as-of 2026-10-13: FR 7,299 forecast and 3,285 observation rows; DE 9,732 and 4,380; NORDPOOL 8,921 and 4,015.
* as-of 2026-10-19 (today): FR 7,245 and 3,285; DE 9,660 and 4,380; NORDPOOL 8,855 and 4,015.
* The flagged "outlier-suspect" rows (FR 583 ... NORDPOOL 644) are flagged, not dropped.

Limits to keep in mind:

* Only vintages from 2026-10-13 exist. A backtest before 2026-10-19 cannot see the snapshots, because they are stamped with the
  request time. Daily accumulation builds the vintage history from now on.
* The store rejects a conflicting re-delivery of a stored key. A realised day whose value changes between snapshots is a new
  revision, so it does not conflict.

### Daily snapshot and scheduling (the user decides)

`scripts/run_daily_snapshot.py` takes today's outage snapshot (FR, DE, NORDPOOL) and today's BQL weather forecasts. The BQL
part runs only while Excel with the Bloomberg add-in is open; otherwise it is skipped with a message. Re-running on the same
day adds nothing. The script writes a temporary authorised config copy, as the other live pulls do. **No scheduled task is
created.** To schedule it, run this once in a Command Prompt (the user's decision, not done here). The task runs
`scripts/run_daily_snapshot.bat`, which appends all output to `data\live\logs\daily_snapshot.log`:

```bat
schtasks /Create /SC DAILY /ST 18:00 /TN "TTF daily snapshot" /TR "\"C:\Marti\python_tests_2\BGN_LNG\ttf_forecaster\scripts\run_daily_snapshot.bat\""
```

Options: `--skip-bql` (outages only), `--skip-fleet`, `--extra-as-of YYYY-MM-DD` (a past snapshot, from 2026-10-13).

Terminal preflight (D-053): before any pull the script opens a desktop-API session and `//blp/refdata` (no data request).
If that fails it stops with exit code 2 and logs `Bloomberg Terminal not connected: log in to the Terminal and re-run.
Nothing was pulled today`. Recovery for a missed day: outages via `--extra-as-of` (stamped with the late request time),
BQL weather by re-pulling that as-of; the LNG fleet snapshot cannot be recovered.

### Open items

* Nuclear availability is wired in D-044 (`derived:` with signed terms and a constant). The fleet constant (63,000 MW) is ASSUMED.
  Verify it, or replace it with a Bloomberg FR nuclear capacity series if one exists.
* Gas-asset outages: no BSRCH domain found (section 8).

## 9. Outage sheet, implied Norwegian outage, LNG cargo imports (2026-10-20)

### 9a. Gas/LNG outage spreadsheet (D-046)
File `data/raw/extracts/outages/LNG_Gas_Outages_20Sep26.xlsx` (a Bloomberg export; the file name is wrong: downloaded 2026-10-20 09:42,
rows to 2026-10-19). Loader: `data/outage_sheet.py`, `pipelines/outage_sheet.py`, config `configs/sources/outage_sheet.yaml`.
- **Snapshot time = file modification time** (08:31:40 UTC for this file). The filename date is not used.
- 125 events, 125 unique event ids: 119 planned, 1 unplanned, 5 unclassified. Regions: N. Sea 112, Atlantic 12, Pacific 1 (Wheatstone, Chevron).
- Units: 110 rows in mcm/d; 2 rows in **mtpa** (Karsto -12.6, Wheatstone 8.9: kept in the side table, not converted, no LNG conversion
  configured); 13 rows with no quantity ('--'); 1 row with no end date ('n.a.', Nord Stream, kept, not in the aggregates).
- Sign: the sheet's Prod./Reduction is negative for a loss; stored reduction = -(sheet value), positive = capacity lost.
- Gas days run Start to End inclusive. Aggregates run from the snapshot day to the last end date, zero-filled.
- Aggregate series (FORECAST rows, grade A, issue = publication = snapshot time): `outage.no.gassco_planned_mcmd`,
  `outage.no.gassco_unplanned_mcmd` (regions N. Sea and Atlantic, all operators), and the GWh/d twins `..._gwhd` at the project GCV
  (10.55 kWh/m3 from fundamentals `balance.gas_quality`, flagged DEFAULT_UNVERIFIED there). LNG terminals:
  `outage.no.lng_terminal_mcmd` (Hammerfest LNG, mapped to NO).
- Snapshot-day values: planned 82.3 mcm/d on 2026-10-23 (Oseberg 5.7 plus others); unplanned 5.2 on 2026-10-20 (Troll).
- **Deviation from the brief**: aggregate rows use issue = snapshot time, not each event's Last Update, because an aggregate is a snapshot
  total. The per-event Last Update, raw text and event id are in the side table `data/live/side/outage_events.parquet` (125 rows, keyed by
  file SHA-256), so revisions can be tracked.
- Idempotent: a file whose SHA-256 is already in the side table is skipped. The side table is written last, so a failed load is retried.

### 9b. Implied realised Norwegian outage (D-047)
- Flow: `flow.no.exit_total_intraday_mcmd` = Bloomberg **GAANTOTI Index** (Norway Total Exit Intraday Within-Day Nominations), verified
  2026-10-20, pulled 2149 rows (2020-11-30 23:00 UTC = 2020-12-01 Amsterdam, to 2026-10-18). Registered in `configs/sources/bloomberg_flows.yaml`
  with publication D+1 07:00 Europe/Amsterdam, grade B.
- Rule (`data/implied_outage.py`, parameters `configs/sources/flow_outage.yaml`): capacity proxy(D) = 95th percentile of the flows of the
  previous 30 days (needs 20, strictly past: shift(1)); shortfall = proxy - flow; implied outage = max(0, shortfall); event = shortfall > 10 mcm/d.
  Outputs (OBSERVATION, grade B, published D+1 07:00 Amsterdam): `flow.no.implied_outage_mcmd`, `flow.no.implied_outage_event`,
  and the GWh/d twins `flow.no.implied_outage_gwhd`, `flow.no.exit_total_intraday_gwhd`. 2129 days computed.
- **Validation against the spreadsheet FAILED.** Comparison on 1 Sep to 19 Oct 2026, implied outage vs the sheet's total active reduction:
  correlation -0.86 over the whole window, -0.29 over 28 Sep to 19 Oct. The method reports the outage on the wrong days (see below).
  Nyhamna (-19.8 mcm/d from 28 Sep to 26 Nov): mean implied outage over 28 Sep to 19 Oct is 3.4 mcm/d against the sheet's 25.8 mcm/d.
  The flow did not fall when Nyhamna was out: mean flow was 332.6 mcm/d for 1 to 19 Oct against 272.4 in September, and the 1 to 19 Oct
  implied outage averages 1.1 mcm/d.
  The failure is a regime effect: the 30-day 95th percentile was set by late-summer flows (mean 319.5 mcm/d over 15 Jul to 31 Aug),
  so the September flows read as outages of 40 to 110 mcm/d, and the event flag is 1 on every day from 1 to 29 Sep.
  **Not tuned to rescue the result** (CLAUDE.md: negative results are results). The implied series is stored but NOT validated; it is not a
  reliable outage signal on this data.
- **Feature mapping (partial, decision for the user):** `flows_features.norway_flow_series` is mapped to `flow.no.exit_total_intraday_gwhd`
  (the measured flow, GAANTOTI x GCV), registered as derived in `configs/sources/fundamentals.yaml`. `norway_unavailable_series` is **left on the
  placeholder** `flows.no.outage_unavailable_gwhd`, because the implied outage did not validate. Mapping it is a decision for the user.

### 9c. LNG cargo imports by country (D-048)
- Source: BSRCH `TPD:DEX` saved searches, one BIKEY per country, configs in `configs/sources/lng_bsrch.yaml`. **Only NL is registered**
  (BIKEY 41UT24O9ARLJ60ST27JSQI3N6: 1,377 import cargoes, arrivals 2016-12-11 to 2026-10-19). The other 13 user keys were not supplied to the
  build and return 0 rows according to the user: they are not listed.
- The NL export has **14 columns, not the 30** the brief expected (Berth_Name, Visit_Type, Port_Country, Port_Region, Arrival, Departure,
  Vessel_Name, Imo, Port_Name, Bmap/Imo links, Qtymtonnes, Qtymcm). Quantities: `Qtymtonnes` in TONNES (negative for an import, so the absolute
  value is used), `Qtymcm` in mcm.
- Side table `data/live/side/lng_cargoes.parquet`: one row per (country, IMO, arrival). Observation series per country and arrival day:
  `lng.<cc>.imports_cargoes_count` (count), `lng.<cc>.imports_tonnes` (t), `lng.<cc>.imports_mcm` (MCM). Days with no arrival inside the span are
  0. Grade B, published at D+1 12:00 UTC (assumption: the cargo is known the day after arrival). NL: 3,600 days, 1,377 cargoes, 118,409 mcm.
- Idempotent: the side table is keyed (country, IMO, arrival), and a second pull adds no cargo.

### 9d. Validation summary and open points
- Implied outage vs sheet: failed (9b). The Nyhamna outage is not visible in Norwegian nominations on this data.
- Open: the 13 other LNG keys, the 30-column export, the mtpa rows, the LNG conversion, and the choice of the flow-based outage for the
  feature (9b).

## 10. LNG fleet snapshot, BSRCH COMDTY:SHIP (D-049, 2026-10-20)

Source: `xbbg.blp.bsrch("COMDTY:SHIP", {VESSEL_TYPE_4: "LNG Tanker", OPERATIONAL_STATUS: "In Service/Commission", OUTPUT: ...})`,
user-supplied and verified from Python on 2026-10-20: **883 rows, 17 columns** with display names. A CURRENT-STATE table: one snapshot per
day from now on, no history. Config `configs/sources/lng_fleet.yaml` (status known), code `data/fleet.py` (mapping to VESSEL_SNAPSHOT_SCHEMA),
`pipelines/fleet.py` (store, arrivals and destination observations), called by `scripts/run_daily_snapshot.py` (Python API, no Excel).

- snapshot_time = publication_time = request time. vessel_id = IMO. laden = Cargo status == 'laden' (315 of 883 today; 54 blank, 514 ballast).
- Destination = Destination (normalized). 'FOR ORDERS' (239 rows), 'Unknown' and blank are not destinations (null).
- Region = the destination's country (normalized): Netherlands, Belgium, France, Germany, United Kingdom -> NWE. Spain, Italy, Portugal,
  Poland, Lithuania, Croatia, Finland, Denmark -> EU. Other countries (incl. Norway, Russia, Turkey, Gibraltar) are unmapped (null).
- ETA earlier than the snapshot -> unknown (null), never arrived: 286 vessels today (57 laden).
- Staleness: Last Seen older than 3 days is kept in the side table but left out of forecasts (47 vessels today, 6 laden).
- Volume: Cargo quantity for laden vessels, unit **t_LNG**. ASSUMPTION: the magnitude (about 60,000 to 90,000 for every laden vessel) is a cargo
  in tonnes; no DES field description was checked. Confirm before use.
- Origin: null. The output has no departure-port field (DEPART_TIME only); nothing is guessed.
- Today's snapshot (2026-10-20 08:54 UTC): 836 fresh vessels, 309 laden. Forecast rows (grade A, issue = snapshot time), NWE / EU:
  cargoes 7 d: 7 / 8; 14 d: 12 / 11; 30 d: 13 / 15; 45 d: 13 / 15. NWE volume 7 d: 411,980 t. Destination rows: 3.
- Point-in-time: a vessel's latest visible snapshot is used. Later snapshots do not change earlier forecasts (tested). No forecast row exists
  before the first snapshot: destination-based features are NaN before it and are never back-filled. Grade A from the first snapshot date.
- Side tables: `data/live/side/lng_vessel_snapshots.parquet` (VESSEL_SNAPSHOT_SCHEMA, all vintages) and `lng_fleet_raw.parquet` (every vessel,
  with the stale and ETA-unknown flags).
- Idempotent per UTC day: a second run on the same day makes no request and writes nothing (scripts/run_daily_snapshot.py --skip-fleet to skip).
- Region labels are a stated choice: "EU" here means EU countries outside NWE, not the whole EU.
## 11. LNG cargo imports: 15 countries, regions, truncation, AHOY cross-check (D-050, 2026-10-20)

Supersedes 9c: the user supplied new keys (2026-10-20), all 15 verified with TPD:DEX the same day. Config `configs/sources/lng_bsrch.yaml`
(`countries`, `regions`), code `data/lng_imports.py` and `pipelines/lng_imports.py`. Pulled 2026-10-20: 13,019 cargoes in total (side table).

| Country | Keys | Rows | First arrival | Last arrival | Note |
|---|---|---|---|---|---|
| NL | 1 | 1377 | 2016-12-11 | 2026-10-19 | |
| PT | 1 | 530 | 2016-12-17 | 2026-10-19 | |
| ES | 1 | 2000 | 2018-07-16 | 2026-10-19 | **capped** (BSRCH cap = 2,000): history_truncated_before 2018-07-16 |
| FI | 1 | 250 | 2017-11-19 | 2026-10-19 | |
| BE | 1 | 926 | 2017-01-02 | 2026-10-18 | |
| GB | 1 | 1364 | 2016-12-21 | 2026-10-18 | |
| FR | 1 | 2000 | 2020-02-19 | 2026-10-17 | **capped**: history_truncated_before 2020-02-19 |
| SE | 1 | 183 | 2019-03-09 | 2026-10-16 | |
| IT | 1 | 1611 | 2016-12-13 | 2026-10-16 | |
| DE | 1 | 329 | 2022-12-20 | 2026-10-15 | |
| GI | 1 | 119 | 2019-01-10 | 2026-10-13 | |
| HR | 1 | 137 | 2020-12-15 | 2026-10-11 | |
| PL | 1 | 455 | 2017-01-21 | 2026-10-19 | |
| LT | 1 | 380 | 2016-12-30 | 2026-10-02 | |
| TR | 1 | 1358 | 2016-12-08 | 2026-10-15 | |

- **Truncation.** For ES and FR the capped search returns the most recent 2,000 cargoes, so the earliest years are missing. Days before
  `history_truncated_before` are **not stored** (absent, never zero). A feature that reads them gets NaN, not zero imports. The user may add
  date-split keys later: several keys per country are supported, merged and de-duplicated by (country, IMO, arrival); the test covers it.
- **Series.** Per country: `lng.<cc>.imports_cargoes_count`, `_tonnes` (t; the export's `Qtymtonnes`, absolute value), `_mcm`. Per region:
  `lng.region.<nwe|iberia|med|nordic_baltic|turkey>.imports_*`. A regional day is stored only when every member has it, so a region that
  contains ES or FR starts on the truncation date (NWE starts 2020-02-19 because of FR).
- **Regions** (config, as the user specified): NWE = NL, BE, FR, GB, DE; IBERIA = ES, PT; MED = IT, HR, GI; NORDIC_BALTIC = FI, SE, PL, LT;
  TURKEY separate (TR). Calendar: 2016-12-08 to 2026-10-19 (one shared calendar). Grade B, published D+1 12:00 UTC (as before).
- **Cross-check with the AHOY aggregates in the store** (`cross_check_ahoy`, the same-day comparison):
  - NWE regional total vs `lng.nwe.imports_t_per_d` (AHOYIMEU), over 2,433 overlapping days: totals agree to about 13 % (ours 355.2 Mt against
    AHOY 407.1 Mt, ratio 0.87), but the **same-day daily correlation is only 0.08**.
  - All 15 countries vs `lng.global.imports_t_per_d` (AHOYIMTL, global): ratio 0.26 (ours 682.9 Mt against 2,614.7 Mt). That is expected,
    since AHOYIMTL is global and includes Asia; correlation 0.09.
  - **Timing finding:** shifting AHOYIMEU by one day gives correlation **0.94** over 2,434 days (AHOY on day D matches our arrivals on D+1).
    So AHOY's dating is one day earlier than the port arrival date, or the two use different day conventions. Our observations are NOT
    shifted (the arrival date is the port date from the export); the convention difference is recorded here for the user to decide. Until that is
    decided, do not compare the two series on the same day.
  - 7-day means: correlation 0.94 at zero lag (smoothing removes the timing offset).
- **Side table** is the source of truth; the daily series are derived from it.
