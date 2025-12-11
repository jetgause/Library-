import os
import sys
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
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

# Security - REQUIRED CONFIGURATION
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    print("CRITICAL ERROR: SECRET_KEY environment variable must be set!")
    print("Generate with: python3 -c \"import secrets; print(secrets.token_urlsafe(32))\"")
    sys.exit(1)

WEAK_KEYS = ["your-secret-key-change-in-production", "change-this-in-production", "secret", "password", "secret-key"]
if SECRET_KEY.lower() in WEAK_KEYS or len(SECRET_KEY) < 32:
    print("CRITICAL ERROR: SECRET_KEY is too weak or uses a default value!")
    print("Generate a strong key with: python3 -c \"import secrets; print(secrets.token_urlsafe(32))\"")
    sys.exit(1)

ALLOWED_ORIGINS_STR = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(",")]
if "*" in ALLOWED_ORIGINS:
    print("WARNING: Wildcard CORS (*) is INSECURE! Use explicit domains in production.")