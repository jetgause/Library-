import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Detect environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
IS_PRODUCTION = ENVIRONMENT == "production"

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

# Security - Enhanced validation
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    print("=" * 70)
    print("CRITICAL ERROR: SECRET_KEY environment variable is required!")
    print("=" * 70)
    print("\nTo fix this issue:")
    print("  1. Run the setup script: python3 setup_env.py --auto")
    print("  OR")
    print("  2. Generate a key manually:")
    print('     python3 -c "import secrets; print(secrets.token_urlsafe(32))"')
    print("  3. Add it to your .env file: SECRET_KEY=<generated-key>")
    print("\n" + "=" * 70)
    sys.exit(1)

WEAK_KEYS = [
    "your-secret-key-change-in-production",
    "change-this-in-production",
    "secret",
    "password",
    "secret-key",
    "test",
    "admin"
]

if SECRET_KEY.lower() in WEAK_KEYS:
    print("=" * 70)
    print("CRITICAL ERROR: SECRET_KEY is using a default/weak value!")
    print("=" * 70)
    print(f"\nYour key: {SECRET_KEY}")
    print("\nThis key is publicly known and INSECURE!")
    print("\nGenerate a new secure key:")
    print('  python3 -c "import secrets; print(secrets.token_urlsafe(32))"')
    print("\n" + "=" * 70)
    sys.exit(1)

if len(SECRET_KEY) < 32:
    print("=" * 70)
    print("CRITICAL ERROR: SECRET_KEY is too short!")
    print("=" * 70)
    print(f"\nCurrent length: {len(SECRET_KEY)} characters")
    print("Required length: 32+ characters")
    print("\nGenerate a new secure key:")
    print('  python3 -c "import secrets; print(secrets.token_urlsafe(32))"')
    print("\n" + "=" * 70)
    sys.exit(1)

# CORS Configuration with validation
ALLOWED_ORIGINS_STR = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000"
)
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(",")]

if "*" in ALLOWED_ORIGINS:
    if IS_PRODUCTION:
        print("=" * 70)
        print("CRITICAL ERROR: Wildcard CORS (*) not allowed in production!")
        print("=" * 70)
        print("\nCurrent ALLOWED_ORIGINS contains wildcard '*'")
        print("\nThis is a CRITICAL security vulnerability!")
        print("\nSet specific origins in your .env file:")
        print("  ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com")
        print("\n" + "=" * 70)
        sys.exit(1)
    else:
        print("\n⚠️  WARNING: Wildcard CORS (*) detected in development mode")
        print("   This should NOT be used in production!\n")

# Success message
if ENVIRONMENT == "development":
    print("✅ Configuration loaded successfully (DEVELOPMENT mode)")
elif ENVIRONMENT == "production":
    print("✅ Configuration loaded successfully (PRODUCTION mode)")