"""Weekly recap email via Resend. The engine's own script calls this
directly — it's an autonomous system reporting its own status, not something
routed through a chat interface."""

from typing import Any, Dict

from jinja2 import Template

RISK_PROFILE_LABELS_IT = {"conservative": "Conservativo", "balanced": "Bilanciato", "aggressive": "Aggressivo"}

EMAIL_TEMPLATE = Template(
    """
<div style="font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 640px; margin: 0 auto; color: #1a1a1a;">
  <h1 style="font-size: 20px;">Ink Ledger &mdash; Riepilogo settimanale</h1>
  <p style="color: #555;">{{ period_label }} &middot; profilo di rischio: {{ risk_profile_label }}</p>

  {% if narrative %}
  <p style="background: #f7f7f7; padding: 12px 16px; border-radius: 6px; line-height: 1.5;">{{ narrative }}</p>
  {% endif %}

  <table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">Capitale</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">{% if equity_eur is not none %}&euro;{{ "%.2f"|format(equity_eur) }}{% else %}n/d &mdash; nessuno storico ancora{% endif %}</td>
    </tr>
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">Rendimento totale (periodo)</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">{{ "%.1f"|format(perf.total_return * 100) }}%</td>
    </tr>
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">Indice di Sharpe</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">{{ "%.2f"|format(perf.sharpe_ratio) }}</td>
    </tr>
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">Drawdown massimo</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">{{ "%.1f"|format(perf.max_drawdown * 100) }}%</td>
    </tr>
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">Volatilit&agrave; attesa ({{ risk_profile_label }})</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">{{ "%.1f"|format(scenario.volatility * 100) }}%</td>
    </tr>
    <tr>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee;">VaR 95% (1 giorno)</td>
      <td style="padding: 8px 0; border-bottom: 1px solid #eee; text-align: right;">{{ "%.1f"|format(scenario.var_95 * 100) }}%</td>
    </tr>
  </table>

  <h2 style="font-size: 16px;">Note del comitato investimenti</h2>
  <ul style="padding-left: 18px;">
    {% for note in top_notes %}
    <li style="margin-bottom: 8px;">
      <strong>{{ note.symbol }}</strong> &mdash; rendimento atteso {{ "%.1f"|format(note.expected_return_annualized * 100) }}%,
      confidenza {{ "%.0f"|format(note.confidence * 100) }}%<br/>
      <span style="color: #555;">{{ note.rationale }}</span>
    </li>
    {% endfor %}
  </ul>

  <p style="color: #999; font-size: 12px; margin-top: 24px;">
    Conto di investimento simulato &mdash; nessun fondo reale coinvolto.<br/>
    <a href="{{ dashboard_url }}" style="color: #666;">Dashboard completa</a> &middot;
    <a href="{{ repo_url }}" style="color: #666;">Codice sorgente su GitHub</a>
  </p>
</div>
"""
)

DASHBOARD_URL = "https://miottogiovanni8-design.github.io/ink-ledger/"
REPO_URL = "https://github.com/miottogiovanni8-design/ink-ledger"


def render_recap_html(
    period_label: str,
    snapshot: Dict[str, Any],
    narrative: str = "",
    top_notes_limit: int = 5,
) -> str:
    risk_profile = snapshot.get("active_risk_profile", "balanced")
    scenario = snapshot.get("scenarios", {}).get(risk_profile, {"volatility": 0.0, "var_95": 0.0})
    equity_eur = snapshot["equity_curve"][-1]["equity"] if snapshot["equity_curve"] else None
    top_notes = snapshot["investment_committee_notes"][:top_notes_limit]
    return EMAIL_TEMPLATE.render(
        period_label=period_label,
        risk_profile_label=RISK_PROFILE_LABELS_IT.get(risk_profile, risk_profile.capitalize()),
        scenario=scenario,
        equity_eur=equity_eur,
        perf=snapshot["performance_stats"],
        top_notes=top_notes,
        narrative=narrative,
        dashboard_url=DASHBOARD_URL,
        repo_url=REPO_URL,
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
