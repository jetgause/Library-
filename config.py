import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))

# Alpaca Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Discord Alerts
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///pulse_trading.db")

# Trading Configuration
PAPER_TRADING_ENABLED = os.getenv("PAPER_TRADING_ENABLED", "true").lower() == "true"
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000"))

# Tier Configuration
TIER_FREE_TOOLS = 5
TIER_PRO_TOOLS = 30
TIER_PRO_PLUS_TOOLS = 52

# Model Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "./models")
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"

# Data Configuration
DATA_PATH = os.getenv("DATA_PATH", "./data")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "pulse_trading.log")

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
