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
