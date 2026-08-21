from typing import Dict, List

from pydantic_settings import BaseSettings, SettingsConfigDict

# Sector-diverse large caps — the equity sleeve of the Black-Litterman universe.
DEFAULT_EQUITY_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL",  # Technology
    "JPM", "BAC", "V",                # Financials
    "XOM", "CVX",                     # Energy
    "JNJ", "UNH",                     # Healthcare
    "PG", "KO",                       # Consumer Staples
    "HD", "MCD",                      # Consumer Discretionary
    "CAT", "HON",                     # Industrials
    "DIS",                            # Communication Services
]

DEFAULT_SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI"]
DEFAULT_FACTOR_ETFS = ["MTUM", "VLUE", "QUAL", "USMV"]
DEFAULT_ETF_UNIVERSE = DEFAULT_SECTOR_ETFS + DEFAULT_FACTOR_ETFS

DEFAULT_SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Technology",
    "JPM": "Financials", "BAC": "Financials", "V": "Financials",
    "XOM": "Energy", "CVX": "Energy",
    "JNJ": "Healthcare", "UNH": "Healthcare",
    "PG": "Consumer Staples", "KO": "Consumer Staples",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "CAT": "Industrials", "HON": "Industrials",
    "DIS": "Communication Services",
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Healthcare",
    "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLI": "Industrials",
}

DEFAULT_FACTOR_MAP: Dict[str, str] = {
    "MTUM": "Momentum", "VLUE": "Value", "QUAL": "Quality", "USMV": "Low Volatility",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper: bool = True

    anthropic_api_key: str = ""

    finnhub_api_key: str = ""
    alphavantage_api_key: str = ""

    resend_api_key: str = ""
    email_from: str = ""
    email_to: str = ""

    # Universe
    equity_universe: List[str] = DEFAULT_EQUITY_UNIVERSE
    etf_universe: List[str] = DEFAULT_ETF_UNIVERSE
    benchmark_symbol: str = "SPY"
    lookback_days: int = 252

    # Portfolio construction
    active_risk_profile: str = "balanced"  # conservative | balanced | aggressive
    max_weight_pct: float = 0.15
    var_confidence_level: float = 0.95
    max_drawdown_circuit_breaker_pct: float = 0.18
    min_trade_usd: float = 5.0

    db_path: str = "data/trading_desk.sqlite"
    snapshot_path: str = "data/dashboard_snapshot.json"


settings = Settings()
