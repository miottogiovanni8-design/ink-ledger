from trading_desk.engine.prefilter import filter_candidates, is_candidate
from trading_desk.engine.schemas import IndicatorSnapshot


def make_snapshot(**overrides) -> IndicatorSnapshot:
    defaults = dict(
        symbol="AAPL",
        asset_class="equity",
        price=100.0,
        rsi_14=50.0,
        macd=0.0,
        macd_signal=0.0,
        macd_prev=0.0,
        macd_signal_prev=0.0,
        bollinger_upper=110.0,
        bollinger_lower=90.0,
        atr_14=2.0,
        has_fresh_headline=False,
    )
    defaults.update(overrides)
    return IndicatorSnapshot(**defaults)


def test_neutral_snapshot_is_not_a_candidate():
    snapshot = make_snapshot()
    is_cand, reasons = is_candidate(snapshot)
    assert is_cand is False
    assert reasons == []


def test_oversold_rsi_is_a_candidate():
    snapshot = make_snapshot(rsi_14=25.0)
    is_cand, reasons = is_candidate(snapshot)
    assert is_cand is True
    assert any("oversold" in r for r in reasons)


def test_overbought_rsi_is_a_candidate():
    snapshot = make_snapshot(rsi_14=75.0)
    is_cand, reasons = is_candidate(snapshot)
    assert is_cand is True
    assert any("overbought" in r for r in reasons)


def test_macd_bullish_crossover_is_a_candidate():
    snapshot = make_snapshot(macd_prev=-0.1, macd_signal_prev=0.0, macd=0.1, macd_signal=0.0)
    is_cand, reasons = is_candidate(snapshot)
    assert is_cand is True
    assert any("bullish crossover" in r for r in reasons)


def test_macd_bearish_crossover_is_a_candidate():
    snapshot = make_snapshot(macd_prev=0.1, macd_signal_prev=0.0, macd=-0.1, macd_signal=0.0)
    is_cand, reasons = is_candidate(snapshot)
    assert is_cand is True
    assert any("bearish crossover" in r for r in reasons)


def test_bollinger_breakout_upper_is_a_candidate():
    snapshot = make_snapshot(price=111.0, bollinger_upper=110.0)
    is_cand, reasons = is_candidate(snapshot)
    assert is_cand is True
    assert any("upper Bollinger" in r for r in reasons)


def test_bollinger_breakout_lower_is_a_candidate():
    snapshot = make_snapshot(price=89.0, bollinger_lower=90.0)
    is_cand, reasons = is_candidate(snapshot)
    assert is_cand is True
    assert any("lower Bollinger" in r for r in reasons)


def test_fresh_headline_is_a_candidate():
    snapshot = make_snapshot(has_fresh_headline=True)
    is_cand, reasons = is_candidate(snapshot)
    assert is_cand is True
    assert any("headline" in r for r in reasons)


def test_multiple_triggers_all_reported():
    snapshot = make_snapshot(rsi_14=25.0, has_fresh_headline=True)
    is_cand, reasons = is_candidate(snapshot)
    assert is_cand is True
    assert len(reasons) == 2


def test_filter_candidates_returns_only_flagged_symbols():
    snapshots = [
        make_snapshot(symbol="NEUTRAL"),
        make_snapshot(symbol="OVERSOLD", rsi_14=20.0),
    ]
    candidates = filter_candidates(snapshots)
    assert len(candidates) == 1
    assert candidates[0][0].symbol == "OVERSOLD"
