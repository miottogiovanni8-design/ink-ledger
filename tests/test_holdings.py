import pytest

from trading_desk.reporting.holdings import build_holdings_detail, compute_cost_basis


class TestComputeCostBasis:
    def test_no_transactions_is_none(self):
        assert compute_cost_basis([]) is None

    def test_single_buy(self):
        basis = compute_cost_basis([{"side": "buy", "notional_usd": 1000.0, "price": 100.0}])
        assert basis == pytest.approx(100.0)

    def test_two_buys_at_different_prices_weighted_average(self):
        txs = [
            {"side": "buy", "notional_usd": 1000.0, "price": 100.0},  # 10 shares
            {"side": "buy", "notional_usd": 1000.0, "price": 200.0},  # 5 shares
        ]
        basis = compute_cost_basis(txs)
        # 15 shares, 2000 total cost -> 133.33 avg
        assert basis == pytest.approx(2000.0 / 15.0)

    def test_partial_sell_does_not_change_average_cost(self):
        txs = [
            {"side": "buy", "notional_usd": 1000.0, "price": 100.0},  # 10 shares @ 100
            {"side": "sell", "notional_usd": 500.0, "price": 120.0},  # sell at a gain, doesn't affect remaining cost basis
        ]
        basis = compute_cost_basis(txs)
        assert basis == pytest.approx(100.0)

    def test_full_sell_closes_position(self):
        txs = [
            {"side": "buy", "notional_usd": 1000.0, "price": 100.0},  # 10 shares
            {"side": "sell", "notional_usd": 1100.0, "price": 110.0},  # sell all 10 shares at the new price
        ]
        assert compute_cost_basis(txs) is None

    def test_sell_without_prior_position_is_ignored(self):
        txs = [{"side": "sell", "notional_usd": 500.0, "price": 100.0}]
        assert compute_cost_basis(txs) is None

    def test_zero_price_transaction_is_skipped(self):
        txs = [{"side": "buy", "notional_usd": 1000.0, "price": 0.0}]
        assert compute_cost_basis(txs) is None


class TestBuildHoldingsDetail:
    def test_computes_pct_since_purchase(self):
        weights = {"AAPL": 0.15}
        transactions = {"AAPL": [{"side": "buy", "notional_usd": 1000.0, "price": 100.0}]}
        holdings = build_holdings_detail(
            weights, total_equity=10_000, latest_prices={"AAPL": 110.0},
            name_map={"AAPL": "Apple Inc."}, transactions_by_symbol=transactions,
        )
        assert len(holdings) == 1
        h = holdings[0]
        assert h["name"] == "Apple Inc."
        assert h["market_value_usd"] == pytest.approx(1500.0)
        assert h["cost_basis"] == pytest.approx(100.0)
        assert h["pct_since_purchase"] == pytest.approx(0.10)

    def test_missing_transactions_gives_none_pct(self):
        holdings = build_holdings_detail(
            {"XLK": 0.1}, total_equity=10_000, latest_prices={"XLK": 240.0},
            name_map={}, transactions_by_symbol={},
        )
        assert holdings[0]["name"] == "XLK"
        assert holdings[0]["cost_basis"] is None
        assert holdings[0]["pct_since_purchase"] is None
