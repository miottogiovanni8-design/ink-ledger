"""Weekly recap email via Resend. The engine's own script calls this
directly — it's an autonomous system reporting its own status, not something
routed through a chat interface."""

from typing import Any, Dict

from jinja2 import Template

EMAIL_TEMPLATE = Template(
    """
<div style="font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 640px; margin: 0 auto; color: #1a1a1a;">
  <h1 style="font-size: 20px;">Ink Ledger — Weekly Recap</h1>
  <p style="color: #555;">{{ period_label }} &middot; risk profile: {{ risk_profile|capitalize }}</p>

  {% if narrative %}
  <p style="background: #f7f7f7; padding: 12px 16px; border-radius: 6px; line-height: 1.5;">{{ narrative }}</p>
  {% endif %}

  <table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">Equity</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">&euro;{{ "%.2f"|format(equity_eur) }}</td>
    </tr>
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">Total return (period)</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">{{ "%.1f"|format(perf.total_return * 100) }}%</td>
    </tr>
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">Sharpe ratio</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">{{ "%.2f"|format(perf.sharpe_ratio) }}</td>
    </tr>
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">Max drawdown</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">{{ "%.1f"|format(perf.max_drawdown * 100) }}%</td>
    </tr>
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">Expected volatility ({{ risk_profile }})</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">{{ "%.1f"|format(scenario.volatility * 100) }}%</td>
    </tr>
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">VaR 95% (1-day)</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">{{ "%.1f"|format(scenario.var_95 * 100) }}%</td>
    </tr>
  </table>

  <h2 style="font-size: 16px;">Investment committee notes</h2>
  <ul style="padding-left: 18px;">
    {% for note in top_notes %}
    <li style="margin-bottom: 8px;">
      <strong>{{ note.symbol }}</strong> — expected return {{ "%.1f"|format(note.expected_return_annualized * 100) }}%,
      {{ "%.0f"|format(note.confidence * 100) }}% confidence<br/>
      <span style="color: #555;">{{ note.rationale }}</span>
    </li>
    {% endfor %}
  </ul>

  <p style="color: #999; font-size: 12px; margin-top: 24px;">
    Paper investment account only — no real funds involved. Full dashboard available on request.
  </p>
</div>
"""
)


def render_recap_html(
    period_label: str,
    snapshot: Dict[str, Any],
    narrative: str = "",
    top_notes_limit: int = 5,
) -> str:
    risk_profile = snapshot.get("active_risk_profile", "balanced")
    scenario = snapshot.get("scenarios", {}).get(risk_profile, {"volatility": 0.0, "var_95": 0.0})
    equity_eur = snapshot["equity_curve"][-1]["equity"] if snapshot["equity_curve"] else 0.0
    top_notes = snapshot["investment_committee_notes"][:top_notes_limit]
    return EMAIL_TEMPLATE.render(
        period_label=period_label,
        risk_profile=risk_profile,
        scenario=scenario,
        equity_eur=equity_eur,
        perf=snapshot["performance_stats"],
        top_notes=top_notes,
        narrative=narrative,
    )


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
