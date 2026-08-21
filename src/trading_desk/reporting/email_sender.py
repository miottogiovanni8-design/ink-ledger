"""Weekly recap email via Resend. The trading engine's own script calls this
directly — it's an autonomous system reporting its own status, not something
routed through a chat interface."""

from typing import Any, Dict

from jinja2 import Template

EMAIL_TEMPLATE = Template(
    """
<div style="font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 640px; margin: 0 auto; color: #1a1a1a;">
  <h1 style="font-size: 20px;">AI Paper Trading Desk — Weekly Recap</h1>
  <p style="color: #555;">{{ period_label }}</p>

  <table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">Equity</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">&euro;{{ "%.2f"|format(stats.equity_eur) }}</td>
    </tr>
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">Realized P&amp;L (period)</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">&euro;{{ "%.2f"|format(stats.total_realized_pnl_eur) }}</td>
    </tr>
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">Sharpe ratio</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">{{ "%.2f"|format(stats.sharpe_ratio) }}</td>
    </tr>
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">Sortino ratio</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">{{ "%.2f"|format(stats.sortino_ratio) }}</td>
    </tr>
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">Max drawdown</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">{{ "%.1f"|format(stats.max_drawdown * 100) }}%</td>
    </tr>
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">Win rate</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">{{ "%.0f"|format(stats.win_rate * 100) }}%</td>
    </tr>
  </table>

  <h2 style="font-size: 16px;">Top trades this period</h2>
  <ul style="padding-left: 18px;">
    {% for trade in top_trades %}
    <li style="margin-bottom: 8px;">
      <strong>{{ trade.symbol }} ({{ trade.direction }})</strong> —
      {{ "%.2f"|format(trade.confidence * 100) }}% confidence<br/>
      <span style="color: #555;">{{ trade.rationale }}</span>
    </li>
    {% endfor %}
  </ul>

  <p style="color: #999; font-size: 12px; margin-top: 24px;">
    Paper trading only — no real funds involved. Full dashboard available on request.
  </p>
</div>
"""
)


def render_recap_html(period_label: str, snapshot: Dict[str, Any], top_trades_limit: int = 5) -> str:
    stats_view = dict(snapshot["stats"])
    stats_view.setdefault("equity_eur", snapshot["equity_curve"][-1]["equity"] if snapshot["equity_curve"] else 0.0)
    top_trades = [
        entry
        for entry in snapshot["trade_journal"]
        if not entry.get("skipped_by_risk_layer") and entry.get("direction") != "hold"
    ][:top_trades_limit]
    return EMAIL_TEMPLATE.render(period_label=period_label, stats=stats_view, top_trades=top_trades)


def send_recap_email(
    emails_client: Any,
    email_from: str,
    email_to: str,
    subject: str,
    html: str,
) -> Any:
    return emails_client.send(
        {
            "from": email_from,
            "to": [email_to],
            "subject": subject,
            "html": html,
        }
    )
