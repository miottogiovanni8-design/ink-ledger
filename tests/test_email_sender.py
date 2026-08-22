from trading_desk.reporting.email_sender import render_recap_html, send_recap_email


def make_snapshot():
    return {
        "equity_curve": [{"t": "2026-08-14T00:00:00Z", "equity": 1000.0}, {"t": "2026-08-21T00:00:00Z", "equity": 1025.0}],
        "performance_stats": {"sharpe_ratio": 1.5, "sortino_ratio": 2.1, "max_drawdown": 0.08, "total_return": 0.025},
        "active_risk_profile": "balanced",
        "scenarios": {
            "balanced": {"weights": {"AAPL": 0.15}, "expected_return": 0.09, "volatility": 0.14, "sharpe": 0.64, "var_95": 0.02, "cvar_95": 0.03},
        },
        "investment_committee_notes": [
            {"symbol": "AAPL", "expected_return_annualized": 0.09, "confidence": 0.55, "rationale": "Strong iPhone cycle plus services growth."},
            {"symbol": "XLE", "expected_return_annualized": -0.02, "confidence": 0.3, "rationale": "Weak crude demand outlook."},
        ],
    }


class FakeEmailsClient:
    def __init__(self):
        self.sent = []

    def send(self, params):
        self.sent.append(params)
        return {"id": "fake-email-id"}


def test_render_recap_html_includes_equity_and_notes():
    html = render_recap_html("Aug 14 - Aug 21, 2026", make_snapshot())
    assert "1025.00" in html
    assert "AAPL" in html
    assert "Strong iPhone cycle" in html
    assert "Bilanciato" in html  # risk profile localized in the subtitle


def test_render_recap_html_shows_fallback_when_equity_curve_empty():
    snapshot = make_snapshot()
    snapshot["equity_curve"] = []
    html = render_recap_html("Aug 14 - Aug 21, 2026", snapshot)
    assert "nessuno storico ancora" in html
    assert "&euro;0.00" not in html


def test_render_recap_html_uses_active_scenario_metrics():
    html = render_recap_html("Aug 14 - Aug 21, 2026", make_snapshot())
    assert "14.0%" in html  # volatility of the active (balanced) scenario
    assert "2.0%" in html  # VaR 95%


def test_render_recap_html_includes_narrative_when_given():
    html = render_recap_html("Aug 14 - Aug 21, 2026", make_snapshot(), narrative="Modest gains this week.")
    assert "Modest gains this week." in html


def test_render_recap_html_omits_narrative_block_when_empty():
    html = render_recap_html("Aug 14 - Aug 21, 2026", make_snapshot())
    assert "background: #f7f7f7" not in html


def test_send_recap_email_wires_params_correctly():
    client = FakeEmailsClient()
    send_recap_email(client, "desk@example.com", "user@example.com", "Weekly Recap", "<p>hi</p>")

    assert len(client.sent) == 1
    sent = client.sent[0]
    assert sent["from"] == "desk@example.com"
    assert sent["to"] == ["user@example.com"]
    assert sent["subject"] == "Weekly Recap"
    assert sent["html"] == "<p>hi</p>"
