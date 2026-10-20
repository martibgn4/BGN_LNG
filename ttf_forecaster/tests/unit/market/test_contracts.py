"""Calendar, contract, expiry, ticker and roll tests. SYNTHETIC data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.contracts.calendars import CalendarError, TradingCalendar
from ttf_forecaster.data.base import IdentifierUnknown
from ttf_forecaster.contracts.futures import (
    BloombergCurveIds,
    ContractParseError,
    ExpiryMismatchError,
    ExpiryRule,
    FuturesContract,
    bloomberg_ticker,
    bloomberg_ticker_candidates,
    contract_metadata,
    cross_check_expiry,
    hours_weights,
    last_trade_date,
    last_trade_map,
    parse_bloomberg_ticker,
    parse_contract_id,
)
from ttf_forecaster.contracts.rolls import (
    MonthlyChain,
    RollRule,
    continuous_level,
    continuous_returns,
    front_contract,
    held_contract,
    safe_log,
    settlement_panel,
)
from ttf_forecaster.data.base import CONTRACT_METADATA_SCHEMA, NonPositivePriceError, validate_frame


# ------------------------------------------------------------------ calendar
class TestCalendar:
    def test_weekend_arithmetic(self) -> None:
        cal = TradingCalendar()
        assert cal.add_sessions("2025-10-31", 1) == pd.Timestamp("2025-11-03")  # Fri -> Mon
        assert cal.add_sessions("2025-11-01", -1) == pd.Timestamp("2025-10-31")  # Sat - 1 = Fri
        assert cal.add_sessions("2025-11-01", 1) == pd.Timestamp("2025-11-03")  # Sat + 1 = Mon
        assert cal.sessions_between("2025-10-31", "2025-11-07") == 5

    def test_configured_holiday_used_outside_observed(self) -> None:
        cal = TradingCalendar(future_holidays=["2030-12-25"])
        assert not cal.is_session("2030-12-25")
        assert cal.next_session("2030-12-24") == pd.Timestamp("2030-12-26")

    def test_empirical_sessions_override_rule(self) -> None:
        obs = pd.bdate_range("2024-01-02", "2024-01-31").drop(pd.Timestamp("2024-01-15"))
        cal = TradingCalendar(obs, future_holidays=[])
        assert not cal.is_session("2024-01-15")         # no settlement observed -> no session
        assert cal.is_session("2024-02-01")             # after observed range: rule (weekday)
        assert cal.add_sessions("2024-01-12", 1) == pd.Timestamp("2024-01-16")

    def test_as_known_at_truncates(self) -> None:
        obs = pd.bdate_range("2024-01-02", "2024-01-31").drop(pd.Timestamp("2024-01-22"))
        cal = TradingCalendar(obs).as_known_at("2024-01-10")
        assert cal.is_session("2024-01-22")             # gap not yet knowable on Jan 10

    def test_tz_aware_input_converted(self) -> None:
        cal = TradingCalendar()
        # 23:30 UTC on Mon 2026-03-30 is Tue 01:30 in Amsterdam
        assert cal.position(pd.Timestamp("2026-03-30 23:30", tz="UTC")) == cal.position("2026-03-31")

    def test_out_of_range(self) -> None:
        with pytest.raises(CalendarError):
            TradingCalendar().add_sessions("2045-12-29", 5)


# ------------------------------------------------------------------ contracts
class TestContracts:
    @pytest.mark.parametrize(("month", "hours"), [(3, 743), (10, 745), (1, 744), (4, 720)])
    def test_dst_hours(self, month: int, hours: int) -> None:
        assert FuturesContract.month("TTF", 2026, month).delivery_hours() == hours

    def test_feb_leap_and_quarter_hours(self) -> None:
        assert FuturesContract.month("TTF", 2024, 2).delivery_hours() == 696
        q1 = FuturesContract.quarter("TTF", 2026, 1)
        assert q1.delivery_hours() == 744 + 672 + 743

    def test_ids_round_trip(self) -> None:
        for c in (FuturesContract.month("TTF", 2026, 11), FuturesContract.quarter("TTF", 2027, 1),
                  FuturesContract.season("TTF", 2026, "WIN"),
                  FuturesContract.calendar_year("TTF", 2027)):
            assert parse_contract_id(c.contract_id) == c
        win = parse_contract_id("TTF.S.2026-WIN")
        assert [m.label for m in win.constituent_months()][0] == "2026-10"
        assert win.constituent_months()[-1].label == "2027-03"
        with pytest.raises(ContractParseError):
            parse_contract_id("TTF.M.2026-13")

    def test_strip_weights_hours(self) -> None:
        q1 = FuturesContract.quarter("TTF", 2026, 1)
        w = hours_weights(q1.constituent_months())
        np.testing.assert_allclose(w, np.array([744, 672, 743]) / 2159)
        assert w.sum() == pytest.approx(1.0)


class TestExpiry:
    def test_two_business_days_before(self) -> None:
        cal, rule = TradingCalendar(), ExpiryRule.from_config()
        # Nov 2025 starts on a Saturday: Mon Nov 3 is first session; 2 back = Thu Oct 30
        assert last_trade_date(FuturesContract.month("TTF", 2025, 11), cal, rule) == pd.Timestamp(
            "2025-10-30")
        # Oct 2025 starts Wednesday -> Mon Sep 29
        assert last_trade_date(FuturesContract.month("TTF", 2025, 10), cal, rule) == pd.Timestamp(
            "2025-09-29")

    def test_holiday_shifts_expiry(self) -> None:
        cal = TradingCalendar(future_holidays=["2030-11-29"])  # Friday before Sun Dec 1 2030
        got = last_trade_date(FuturesContract.month("TTF", 2030, 12), cal, ExpiryRule.default())
        assert got == pd.Timestamp("2030-11-27")

    def test_vendor_mismatch_is_hard_error(self) -> None:
        cross_check_expiry("TTF.M.2025-11", "2025-10-30", "2025-10-30")
        cross_check_expiry("TTF.M.2025-11", "2025-10-30", None)
        with pytest.raises(ExpiryMismatchError):
            cross_check_expiry("TTF.M.2025-11", "2025-10-30", "2025-10-31")
        with pytest.raises(ExpiryMismatchError):
            contract_metadata([FuturesContract.month("TTF", 2025, 11)], TradingCalendar(),
                              ExpiryRule.default(),
                              vendor_last_trade={"TTF.M.2025-11": pd.Timestamp("2025-10-29")})

    def test_metadata_schema_and_source(self) -> None:
        cs = [FuturesContract.month("TTF", 2025, m) for m in (10, 11)]
        meta = contract_metadata(cs, TradingCalendar(), ExpiryRule.default(),
                                 vendor_last_trade={"TTF.M.2025-11": "2025-10-30"})
        validate_frame(meta, CONTRACT_METADATA_SCHEMA)
        assert list(meta["last_trade_date_source"]) == ["rule", "vendor"]
        assert last_trade_map(meta)["TTF.M.2025-10"] == pd.Timestamp("2025-09-29")


class TestTickers:
    def test_ttf_rule(self) -> None:
        ids = BloombergCurveIds.from_config("TTF")
        assert bloomberg_ticker(FuturesContract.month("TTF", 2026, 11), ids) == "TZTX26 Comdty"
        assert bloomberg_ticker(FuturesContract.quarter("TTF", 2027, 1), ids) == "QZTF27 Comdty"
        assert bloomberg_ticker(FuturesContract.quarter("TTF", 2027, 4), ids) == "QZTV27 Comdty"
        assert bloomberg_ticker(FuturesContract.season("TTF", 2027, "SUM"), ids) == "QQTJ27 Comdty"
        assert bloomberg_ticker(FuturesContract.season("TTF", 2026, "WIN"), ids) == "QQTV26 Comdty"
        assert bloomberg_ticker(FuturesContract.calendar_year("TTF", 2028), ids) == "QTTF28 Comdty"

    def test_the_uses_ice_endex_lft_root(self) -> None:
        # D-039: THE is ICE Endex root LFT with the TZT 2-digit rule (LFTX16, LFTX26 verified live)
        ids = BloombergCurveIds.from_config("THE")
        assert ids.form == "tzt" and ids.roots == {"M": "LFT"}
        assert bloomberg_ticker(FuturesContract.month("THE", 2016, 10), ids) == "LFTV16 Comdty"
        assert bloomberg_ticker(FuturesContract.month("THE", 2026, 11), ids) == "LFTX26 Comdty"

    def test_pnx_hub_candidates(self) -> None:
        # D-039: PNX form. Listed contract: root+month+1-digit year+M. Expired: same + 2-digit year.
        ids = BloombergCurveIds.from_config("PEG")
        assert ids.form == "pnx"
        listed, expired = bloomberg_ticker_candidates(FuturesContract.month("PEG", 2026, 11), ids)
        assert (listed, expired) == ("PNGX6M Comdty", "PNGX6M26 Comdty")
        assert bloomberg_ticker_candidates(FuturesContract.month("PEG", 2016, 11), ids) == (
            "PNGX6M Comdty", "PNGX6M16 Comdty")
        with pytest.raises(IdentifierUnknown):
            bloomberg_ticker(FuturesContract.month("PEG", 2026, 11), ids)  # single form refused
        with pytest.raises(IdentifierUnknown):
            bloomberg_ticker_candidates(FuturesContract.quarter("PEG", 2027, 1), ids)

    def test_pnx_parse_expired_only(self) -> None:
        ids = BloombergCurveIds.from_config("PSV")
        c = FuturesContract.month("PSV", 2021, 11)
        assert parse_bloomberg_ticker("PSRX1M21 Comdty", ids) == c  # PSRX1M21 = Nov21 expired
        assert parse_bloomberg_ticker("PSRX6M16 Comdty", ids) == FuturesContract.month("PSV", 2016, 11)
        with pytest.raises(ContractParseError):
            parse_bloomberg_ticker("PSRX6M Comdty", ids)  # listed form: year ambiguous
        with pytest.raises(ContractParseError):
            parse_bloomberg_ticker("PSRX5M16 Comdty", ids)  # digits disagree

    def test_parse_round_trip(self) -> None:
        for hub in ("TTF", "THE"):
            ids = BloombergCurveIds.from_config(hub)
            cs = [FuturesContract.month(hub, 2026, 3)]
            if hub == "TTF":  # THE is monthly-only (LFT roots for Q/S/Y are unverified)
                cs += [FuturesContract.quarter(hub, 2027, 3), FuturesContract.season(hub, 2027, "WIN"),
                       FuturesContract.calendar_year(hub, 2029)]
            for c in cs:
                assert parse_bloomberg_ticker(bloomberg_ticker(c, ids), ids) == c


# ------------------------------------------------------------------ rolls
@pytest.fixture(scope="module")
def chain_and_panel(syn_curve: pd.DataFrame) -> tuple[MonthlyChain, pd.DataFrame]:
    panel = settlement_panel(syn_curve)
    cal = TradingCalendar.from_curve(syn_curve)
    meta = contract_metadata([parse_contract_id(c) for c in panel.columns], cal,
                             ExpiryRule.default())
    return MonthlyChain(last_trade_map(meta), cal), panel


class TestRolls:
    def test_front_rolls_R_sessions_before_expiry(
        self, chain_and_panel: tuple[MonthlyChain, pd.DataFrame]
    ) -> None:
        chain, _ = chain_and_panel
        cal = chain.calendar
        ltd = chain.last_trade[chain.index_of["TTF.M.2023-03"]]   # 2023-02-27 (Mon)
        assert ltd == pd.Timestamp("2023-02-27")
        # 6 sessions before ltd: still front; 5 sessions before: rolled
        assert front_contract(cal.add_sessions(ltd, -6), chain, 5) == "TTF.M.2023-03"
        assert front_contract(cal.add_sessions(ltd, -5), chain, 5) == "TTF.M.2023-04"
        assert front_contract(cal.add_sessions(ltd, -1), chain, 0) == "TTF.M.2023-03"
        assert front_contract(ltd, chain, 0) == "TTF.M.2023-04"

    def test_held_contract_trades_at_exit(
        self, chain_and_panel: tuple[MonthlyChain, pd.DataFrame]
    ) -> None:
        chain, _ = chain_and_panel
        cal = chain.calendar
        for t in cal.sessions_in("2022-03-01", "2023-06-30"):
            for h in (1, 5, 20):
                c = held_contract(t, h, chain, RollRule(5, 0, True))
                ltd = chain.last_trade[chain.index_of[c]]
                assert ltd >= cal.add_sessions(t, h)
                assert cal.sessions_between(t, ltd) > 5

    def test_literal_rule_can_hold_expiry_week(
        self, chain_and_panel: tuple[MonthlyChain, pd.DataFrame]
    ) -> None:
        chain, _ = chain_and_panel
        t = chain.calendar.add_sessions("2023-02-27", -2)
        assert held_contract(t, 1, chain, RollRule(5, 0, False)) == "TTF.M.2023-03"
        assert held_contract(t, 1, chain, RollRule(5, 0, True)) == "TTF.M.2023-04"

    def test_returns_are_within_contract(
        self, chain_and_panel: tuple[MonthlyChain, pd.DataFrame]
    ) -> None:
        chain, panel = chain_and_panel
        ret = continuous_returns(panel, chain, 5)
        lvl = continuous_level(panel, chain, 5)
        days = ret.index
        for i in range(1, len(days)):
            c = ret["contract_id"].iloc[i]
            expected = np.log(panel.at[days[i], c]) - np.log(panel.at[days[i - 1], c])
            assert ret["value"].iloc[i] == pytest.approx(expected)
            assert c == lvl["contract_id"].iloc[i - 1]
        rolls = lvl["contract_id"].ne(lvl["contract_id"].shift())
        roll_days = lvl.index[rolls][1:]
        assert len(roll_days) > 10
        # on a roll day the return is still measured on the contract held over the day (the old
        # front), never as new-contract level minus old-contract level
        for d in roll_days:
            prev = days[days.get_loc(d) - 1]
            assert ret.at[d, "contract_id"] == lvl.at[prev, "contract_id"]
            assert ret.at[d, "contract_id"] != lvl.at[d, "contract_id"]

    def test_missing_settlement_gives_nan_return(
        self, chain_and_panel: tuple[MonthlyChain, pd.DataFrame]
    ) -> None:
        chain, panel = chain_and_panel
        p = panel.copy()
        day = pd.Timestamp("2022-06-15")
        c = front_contract(chain.calendar.previous_session(day), chain, 5)
        p.loc[day, c] = np.nan
        ret = continuous_returns(p, chain, 5)
        assert np.isnan(ret.at[day, "value"])
        assert np.isnan(ret.at[chain.calendar.next_session(day), "value"])

    def test_non_positive_log_raises(self) -> None:
        with pytest.raises(NonPositivePriceError):
            safe_log(np.array([10.0, -1.0]))
        assert np.isnan(safe_log(float("nan")))
