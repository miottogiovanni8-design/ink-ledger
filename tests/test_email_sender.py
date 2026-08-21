from trading_desk.reporting.email_sender import render_recap_html, send_recap_email


def make_snapshot():
    return {
        "equity_curve": [{"t": "2026-08-14T00:00:00Z", "equity": 1000.0}, {"t": "2026-08-21T00:00:00Z", "equity": 1025.0}],
        "stats": {
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.1,
            "max_drawdown": 0.08,
            "win_rate": 0.6,
            "profit_factor": 1.8,
            "closed_trades_count": 5,
            "total_realized_pnl_eur": 25.0,
        },
        "trade_journal": [
            {
                "symbol": "AAPL",
                "direction": "long",
                "confidence": 0.72,
                "rationale": "RSI oversold with a positive earnings headline.",
                "skipped_by_risk_layer": False,
            },
            {
                "symbol": "TSLA",
                "direction": "hold",
                "confidence": 0.4,
                "rationale": "Mixed signals.",
                "skipped_by_risk_layer": False,
            },
            {
                "symbol": "MSFT",
                "direction": "short",
                "confidence": 0.55,
                "rationale": "Circuit breaker tripped.",
                "skipped_by_risk_layer": True,
            },
        ],
        "positions": [],
    }


class FakeEmailsClient:
    def __init__(self):
        self.sent = []

    def send(self, params):
        self.sent.append(params)
        return {"id": "fake-email-id"}


def test_render_recap_html_includes_stats_and_filters_journal():
    html = render_recap_html("Aug 14 - Aug 21, 2026", make_snapshot())

    assert "1025.00" in html
    assert "AAPL" in html
    assert "TSLA" not in html  # holds are excluded from top trades
    assert "MSFT" not in html  # skipped-by-risk-layer entries are excluded


def test_send_recap_email_wires_params_correctly():
    client = FakeEmailsClient()
    send_recap_email(client, "desk@example.com", "user@example.com", "Weekly Recap", "<p>hi</p>")

    assert len(client.sent) == 1
    sent = client.sent[0]
    assert sent["from"] == "desk@example.com"
    assert sent["to"] == ["user@example.com"]
    assert sent["subject"] == "Weekly Recap"
    assert sent["html"] == "<p>hi</p>"
