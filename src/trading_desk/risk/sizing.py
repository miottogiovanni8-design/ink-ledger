"""Fixed-fractional position sizing and bracket-order price levels."""

from typing import Literal

Direction = Literal["long", "short"]


def position_size_eur(
    daily_budget_eur: float,
    risk_pct_per_trade: float,
    stop_loss_distance_pct: float,
    max_position_pct_of_budget: float,
) -> float:
    """Size a position so that hitting the stop loses exactly `risk_pct_per_trade`
    of the daily budget, capped at `max_position_pct_of_budget` of the budget."""
    if stop_loss_distance_pct <= 0:
        raise ValueError("stop_loss_distance_pct must be positive")

    raw_size = (daily_budget_eur * risk_pct_per_trade) / stop_loss_distance_pct
    cap = daily_budget_eur * max_position_pct_of_budget
    return min(raw_size, cap)


def stop_loss_price(entry_price: float, atr: float, direction: Direction, atr_multiplier: float) -> float:
    distance = atr * atr_multiplier
    if direction == "long":
        return entry_price - distance
    return entry_price + distance


def take_profit_price(entry_price: float, stop_loss: float, direction: Direction, rr_ratio: float) -> float:
    stop_distance = abs(entry_price - stop_loss)
    reward_distance = stop_distance * rr_ratio
    if direction == "long":
        return entry_price + reward_distance
    return entry_price - reward_distance


def stop_loss_distance_pct(entry_price: float, stop_loss: float) -> float:
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    return abs(entry_price - stop_loss) / entry_price


def clamp_decision_to_risk_limits(
    decision,
    entry_price: float,
    atr: float,
    daily_budget_eur: float,
    remaining_budget_eur: float,
    risk_pct_per_trade: float,
    max_position_pct_of_budget: float,
    atr_multiplier: float,
    rr_ratio: float,
):
    """The LLM proposes a size and stop/take-profit; this is where the
    deterministic layer disposes. Never execute the model's numbers as-is —
    always re-derive stop/take-profit if missing and cap size at the
    fixed-fractional risk limit and whatever daily budget remains."""
    if decision.direction == "hold":
        return decision.model_copy(
            update={"position_size_usd": 0.0, "stop_loss_price": None, "take_profit_price": None}
        )

    stop_loss = decision.stop_loss_price
    if not stop_loss or stop_loss <= 0:
        stop_loss = stop_loss_price(entry_price, atr, decision.direction, atr_multiplier)

    take_profit = decision.take_profit_price
    if not take_profit or take_profit <= 0:
        take_profit = take_profit_price(entry_price, stop_loss, decision.direction, rr_ratio)

    distance_pct = stop_loss_distance_pct(entry_price, stop_loss)
    risk_based_cap = position_size_eur(
        daily_budget_eur, risk_pct_per_trade, distance_pct, max_position_pct_of_budget
    )
    size = min(decision.position_size_usd or risk_based_cap, risk_based_cap, max(remaining_budget_eur, 0.0))

    return decision.model_copy(
        update={
            "position_size_usd": max(size, 0.0),
            "stop_loss_price": stop_loss,
            "take_profit_price": take_profit,
        }
    )
