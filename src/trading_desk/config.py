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
DEFAULT_SECTOR_ETF_MAP: Dict[str, str] = {
    "Technology": "XLK", "Financials": "XLF", "Energy": "XLE", "Healthcare": "XLV",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Industrials": "XLI",
}
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

# Approximate S&P 500 sector weights — a static reference (S&P/State Street
# publish updated GICS sector weights regularly; this is a representative
# snapshot, not a live feed) used as the benchmark side of Brinson
# attribution. Sectors not in our universe's sector map (Utilities, Real
# Estate, Materials) are folded into "Other" since we hold no proxy for them.
DEFAULT_BENCHMARK_SECTOR_WEIGHTS: Dict[str, float] = {
    "Technology": 0.32,
    "Financials": 0.13,
    "Healthcare": 0.11,
    "Consumer Discretionary": 0.10,
    "Communication Services": 0.09,
    "Industrials": 0.08,
    "Consumer Staples": 0.06,
    "Energy": 0.03,
    "Other": 0.08,
}

# Static reference table — company/fund full names don't change often enough
# to justify a live lookup, and ETFs aren't reliably covered by equity
# profile endpoints anyway.
DEFAULT_NAME_MAP: Dict[str, str] = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "NVDA": "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc.", "JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corporation",
    "V": "Visa Inc.", "XOM": "Exxon Mobil Corporation", "CVX": "Chevron Corporation",
    "JNJ": "Johnson & Johnson", "UNH": "UnitedHealth Group Incorporated",
    "PG": "Procter & Gamble Company", "KO": "The Coca-Cola Company",
    "HD": "The Home Depot Inc.", "MCD": "McDonald's Corporation",
    "CAT": "Caterpillar Inc.", "HON": "Honeywell International Inc.",
    "DIS": "The Walt Disney Company",
    "XLK": "Technology Select Sector SPDR Fund", "XLF": "Financial Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund", "XLV": "Health Care Select Sector SPDR Fund",
    "XLY": "Consumer Discretionary Select Sector SPDR Fund",
    "XLP": "Consumer Staples Select Sector SPDR Fund", "XLI": "Industrial Select Sector SPDR Fund",
    "MTUM": "iShares MSCI USA Momentum Factor ETF", "VLUE": "iShares MSCI USA Value Factor ETF",
    "QUAL": "iShares MSCI USA Quality Factor ETF", "USMV": "iShares MSCI USA Min Vol Factor ETF",
    "SPY": "SPDR S&P 500 ETF Trust",
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
