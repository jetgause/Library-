#!/usr/bin/env python3
"""
PULSE Platform - Environment Setup Script
==========================================

Automatically generates a secure .env file with proper configuration.

Usage:
    python3 setup_env.py
    
Or with auto-accept defaults:
    python3 setup_env.py --auto

Author: Security Automation
Created: 2025-12-11
"""

import os
import sys
import secrets
from pathlib import Path

def generate_secret_key(length=32):
    """Generate a cryptographically secure secret key."""
    return secrets.token_urlsafe(length)

def create_env_file(auto=False):
    """Create .env file with secure configuration."""
    
    print("=" * 70)
    print("PULSE Platform - Secure Environment Configuration")
    print("=" * 70)
    print()
    
    # Check if .env already exists
    env_path = Path(".env")
    if env_path.exists():
        print("⚠️  .env file already exists!")
        if not auto:
            response = input("Do you want to overwrite it? (yes/no): ").lower()
            if response not in ['yes', 'y']:
                print("❌ Setup cancelled.")
                return False
        print("Backing up existing .env to .env.backup...")
        if Path(".env.backup").exists():
            os.remove(".env.backup")
        os.rename(".env", ".env.backup")
    
    print("\n🔐 Generating secure configuration...\n")
    
    # Generate secure SECRET_KEY
    secret_key = generate_secret_key(32)
    print(f"✅ Generated SECRET_KEY: {secret_key[:10]}...{secret_key[-10:]}")
    
    # Get configuration values
    config = {
        'SECRET_KEY': secret_key,
        'ALLOWED_ORIGINS': 'http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000',
        'API_HOST': '127.0.0.1',
        'API_PORT': '8000',
        'API_WORKERS': '4',
        'ALPACA_API_KEY': '',
        'ALPACA_SECRET_KEY': '',
        'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets',
        'DISCORD_WEBHOOK_URL': '',
        'DATABASE_URL': 'sqlite:///pulse_trading.db',
        'PAPER_TRADING_ENABLED': 'true',
        'INITIAL_CAPITAL': '100000',
        'MODEL_PATH': './models',
        'USE_GPU': 'false',
        'DATA_PATH': './data',
        'CACHE_ENABLED': 'true',
        'LOG_LEVEL': 'INFO',
        'LOG_FILE': 'pulse_trading.log'
    }
    
    if not auto:
        print("\n📝 Optional Configuration (press Enter to skip):\n")
        
        alpaca_key = input("Alpaca API Key (optional): ").strip()
        if alpaca_key:
            config['ALPACA_API_KEY'] = alpaca_key
            alpaca_secret = input("Alpaca Secret Key: ").strip()
            config['ALPACA_SECRET_KEY'] = alpaca_secret
        
        discord = input("Discord Webhook URL (optional): ").strip()
        if discord:
            config['DISCORD_WEBHOOK_URL'] = discord
    
    # Write .env file
    print("\n💾 Writing configuration to .env file...")
    
    env_content = f"""# PULSE Trading Platform - Environment Configuration
# Generated: {Path(__file__).name}
# WARNING: Never commit this file to version control!

# ============================================================================
# CRITICAL SECURITY SETTINGS
# ============================================================================

SECRET_KEY={config['SECRET_KEY']}
ALLOWED_ORIGINS={config['ALLOWED_ORIGINS']}

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_HOST={config['API_HOST']}
API_PORT={config['API_PORT']}
API_WORKERS={config['API_WORKERS']}

# ============================================================================
# TRADING CONFIGURATION
# ============================================================================

ALPACA_API_KEY={config['ALPACA_API_KEY']}
ALPACA_SECRET_KEY={config['ALPACA_SECRET_KEY']}
ALPACA_BASE_URL={config['ALPACA_BASE_URL']}

# ============================================================================
# NOTIFICATIONS
# ============================================================================

DISCORD_WEBHOOK_URL={config['DISCORD_WEBHOOK_URL']}

# ============================================================================
# DATABASE & STORAGE
# ============================================================================

DATABASE_URL={config['DATABASE_URL']}
MODEL_PATH={config['MODEL_PATH']}
DATA_PATH={config['DATA_PATH']}
CACHE_ENABLED={config['CACHE_ENABLED']}

# ============================================================================
# TRADING SETTINGS
# ============================================================================

PAPER_TRADING_ENABLED={config['PAPER_TRADING_ENABLED']}
INITIAL_CAPITAL={config['INITIAL_CAPITAL']}
USE_GPU={config['USE_GPU']}

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL={config['LOG_LEVEL']}
LOG_FILE={config['LOG_FILE']}
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ .env file created successfully!")
    
    # Verify configuration
    print("\n🔍 Verifying configuration...")
    try:
        # Test import
        import config as cfg
        print("✅ Configuration validated!")
        print(f"   - SECRET_KEY: Set ({len(cfg.SECRET_KEY)} characters)")
        print(f"   - ALLOWED_ORIGINS: {len(cfg.ALLOWED_ORIGINS)} origin(s)")
        print(f"   - API_HOST: {cfg.API_HOST}")
        print(f"   - API_PORT: {cfg.API_PORT}")
        return True
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        print("\nYou may need to:")
        print("1. Check that python-dotenv is installed: pip install python-dotenv")
        print("2. Verify config.py is in the current directory")
        return False

def main():
    """Main entry point."""
    auto = '--auto' in sys.argv
    
    if not auto:
        print("\nThis script will create a secure .env configuration file.")
        print("Press Ctrl+C at any time to cancel.\n")
    
    try:
        success = create_env_file(auto=auto)
        
        if success:
            print("\n" + "=" * 70)
            print("✅ CONFIGURATION COMPLETE!")
            print("=" * 70)
            print("\nYour application is now configured securely.")
            print("\nNext steps:")
            print("1. Review the .env file (optional)")
            print("2. Start the API server: python api_server.py")
            print("3. Access the API at: http://127.0.0.1:8000")
            print("\n⚠️  Remember: NEVER commit the .env file to git!")
            return 0
        else:
            print("\n❌ Configuration failed. Please check the errors above.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
