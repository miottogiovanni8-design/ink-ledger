"""Rule-based, LLM-free candidate screening.

Runs against every tracked symbol every cycle; only symbols that trip a
trigger here become candidates for a (paid) Claude reasoning call. This is
the cost/quality tradeoff that keeps call volume proportional to genuinely
interesting market moves rather than the size of the watchlist.
"""

from typing import List, Tuple

from trading_desk.engine.schemas import IndicatorSnapshot

RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0


def is_candidate(snapshot: IndicatorSnapshot) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    if snapshot.rsi_14 <= RSI_OVERSOLD:
        reasons.append(f"RSI {snapshot.rsi_14:.1f} oversold (<= {RSI_OVERSOLD:.0f})")
    elif snapshot.rsi_14 >= RSI_OVERBOUGHT:
        reasons.append(f"RSI {snapshot.rsi_14:.1f} overbought (>= {RSI_OVERBOUGHT:.0f})")

    bullish_cross = snapshot.macd_prev <= snapshot.macd_signal_prev and snapshot.macd > snapshot.macd_signal
    bearish_cross = snapshot.macd_prev >= snapshot.macd_signal_prev and snapshot.macd < snapshot.macd_signal
    if bullish_cross:
        reasons.append("MACD bullish crossover")
    elif bearish_cross:
        reasons.append("MACD bearish crossover")

    if snapshot.price >= snapshot.bollinger_upper:
        reasons.append("price broke above upper Bollinger band")
    elif snapshot.price <= snapshot.bollinger_lower:
        reasons.append("price broke below lower Bollinger band")

    if snapshot.has_fresh_headline:
        reasons.append("fresh headline since last cycle")

    return (len(reasons) > 0, reasons)


def filter_candidates(snapshots: List[IndicatorSnapshot]) -> List[Tuple[IndicatorSnapshot, List[str]]]:
    candidates = []
    for snapshot in snapshots:
        is_cand, reasons = is_candidate(snapshot)
        if is_cand:
            candidates.append((snapshot, reasons))
    return candidates
