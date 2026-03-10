# PULSE Trading Platform

**Advanced 0DTE Options Trading System with Reinforcement Learning**

A sophisticated, production-ready options trading platform featuring 52 specialized trading tools, 13 Greek calculation engines, full SABR volatility surfaces, and autonomous reinforcement learning optimization.

---

## 🚀 Quick Start (2 Minutes)

### Prerequisites
- Python 3.8+
- pip
- (Linux) system build tools for compiling native Python dependencies (`make`, `gcc`)

If `make` or `gcc` are missing on Ubuntu/Debian, install them first:

```bash
sudo apt-get update
sudo apt-get install -y build-essential
```

Then verify:

```bash
make --version
gcc --version
```

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/jetgause/Library-.git
cd Library-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (automated)
python3 setup_env.py --auto

# 4. Start the application
python3 api_server.py
```

### Verify Installation
```bash
# Run tests
pytest tests/ -v

# Check API health
curl http://127.0.0.1:8000/health

# View API docs
open http://127.0.0.1:8000/docs
```

## 🔒 Security

This platform has mandatory security requirements:
- **SECRET_KEY** must be configured (minimum 32 characters)
- **CORS origins** must be explicitly set (no wildcards in production)
- **API binding** defaults to localhost

See [SECURITY_SETUP.md](SECURITY_SETUP.md) for complete security configuration.

---

## 📊 Features

- **52 optimized trading tools** (Gamma Wall Pin, Charm Flow, Vanna Shock, etc.)
- **Smart marketplace** with tool rental system
- **Real-time options data analysis**
- **Paper trading engine**
- **Web UI with PWA support**
- **Comprehensive security framework**

---

## 🧪 Testing

Run the full test suite:
```bash
pytest tests/ -v --cov
```

Run specific test suites:
```bash
pytest tests/test_security.py -v    # Security tests
pytest tests/test_factory.py -v     # Tool factory tests
pytest tests/test_marketplace.py -v # Marketplace tests
```

---

## 🚀 **What Makes PULSE Unique**

- **52 Trading Tools** across 5 tiers (Free/Pro/Pro+) targeting specific market microstructure events
- **13 Greek Engines**: Delta, Gamma, Vega, Theta, Rho, Charm, Vanna, Vomma, Volga, Speed, Zomma, Color, Ultima
- **Market Maker Detection**: Real dealer hedging flow analysis (GEX, Charm, Vanna)
- **SABR Volatility Surface**: Full calibration with 0DTE-safe implementation
- **GARCH Simulation**: Volatility clustering for realistic backtesting
- **Reinforcement Learning**: PPO-based autonomous strategy optimization
- **Auto-Distillation**: Decision trees auto-approved if >15% improvement
- **Paper Trading System**: Full P&L tracking with SQLite persistence
- **Discord Alerts**: Rich embeds with real-time signal notifications
- **Modal Deployment**: One-line cloud deployment with GPU support
- **100K+ Simulations**: Rigorous Monte Carlo backtesting framework

---

## 📊 **Architecture Overview**

```
pulse-trading-platform/
├── core/
│   ├── trainer/
│   │   ├── engines.py                    # 13 Greek calculation engines
│   │   ├── trainer.py                    # RL training loop (PPO)
│   │   ├── mm_engines/
│   │   │   └── mm_hedge_engines.py       # GEX, Charm, Vanna detection
│   │   ├── god_tier/
│   │   │   └── god_tier_alpha_engines.py # Vol triggers, gamma flips
│   │   ├── pirate_alpha/
│   │   │   └── true_pirate_alpha_no_latency.py  # Daily-data engines
│   │   └── volatility/
│   │       ├── sabr_volatility.py        # SABR surface calibration
│   │       └── garch_simulator.py        # GARCH(1,1) simulation
│   └── distillation/
│       └── distillation_engine.py        # Auto-approval system
├── tools/
│   └── library/
│       ├── ultra_0dte_final_10.py        # 10 ultra 0DTE tools (Pro+)
│       ├── zero_dte_nuclear_10.py        # 10 nuclear tools (Pro+)
│       ├── god_tier_nuclear_5.py         # 5 god-tier tools (Pro+)
│       ├── pirate_alpha_5.py             # 5 pirate tools (Free)
│       └── [12+ more tool files]
├── api_server.py                         # FastAPI server
├── production_tool_executor.py           # Tool execution engine
├── paper_trading.py                      # Paper trading system
├── pulse_optimizer_v2.py                 # 100K simulation optimizer
└── deploy/
    └── modal/
        └── modal_deployment.py           # Cloud deployment
```

---

## ⚡ **Local Runtime Flow**

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2. Configure Environment**

```bash
cp .env.example .env
# Edit .env with your API keys
```

### **3. Run Paper Trading**

```bash
python paper_trading.py
```

### **4. Start API Server**

```bash
python api_server.py
```

The API will be available at `http://localhost:8000`

---

## 🎯 **Tier System**

| Tier | Tools | Features | Price |
|------|-------|----------|-------|
| **Free** | 5 tools | Paper trading only, basic Greeks | $0 |
| **Pro** | 30 tools | Paper trading, Discord alerts, full Greeks | $99/mo |
| **Pro+** | 52 tools | Live Alpaca trading, all engines, priority support | $299/mo |

---

## 🛠️ **Tool Categories**

### **Ultra 0DTE Final 10** (Pro/Pro+)
High-frequency tools for same-day expiration events:
- Gamma Wall Pin Collapse
- Charm Flow Nuclear Breach
- Vanna Shock Reversal
- Vol Trigger Cascade
- Dealer Short Gamma Forced
- Pinning Collapse Vacuum
- 0DTE Magnet + Charm
- Triple Cascade Black Swan
- Silence Before The Storm
- Hidden Gamma Squeeze Ignition

### **God-Tier Nuclear 5** (Pro+)
Institutional-grade rare event detection:
- Gamma Flip to Short
- Vol-of-Vol Death Spiral Fade
- Fed Wire Leak + Gamma Reset
- Dark Pool Gamma Flush
- Silence Before The Storm v2

### **Pirate Alpha 5** (Free tier included)
Daily-data-only strategies (no latency required):
- Regime Phase Shift
- Earnings Crash Detector
- Liquidity Death Cross
- Vol Control Forced Selling
- Market Structure Fracture

*Plus 32 more tools across multiple categories...*

---

## 🧠 **Reinforcement Learning System**

The platform includes a complete RL training pipeline:

1. **Environment**: 30-dimensional observation space with all Greeks, flows, and market data
2. **Agent**: PPO (Proximal Policy Optimization) from Stable Baselines 3
3. **Reward**: Multi-objective (Sharpe ratio + win rate + frequency)
4. **Distillation**: Auto-extract decision trees from trained policy
5. **Auto-Approval**: Deploy if improvement > 15%

```python
from core.trainer.trainer import train_rl_agent

# Train new strategy
results = train_rl_agent(
    tool_name="my_new_strategy",
    n_episodes=10000,
    target_sharpe=2.0
)
```

---

## 📈 **Backtesting**

Run comprehensive 100K simulation backtest:

```bash
python pulse_optimizer_v2.py
```

**Output:**
```
================================================================================
PULSE TOOLS OPTIMIZER V2.0 — PRODUCTION BACKTEST
================================================================================
Simulations: 100,000
Tools: All 10 ULTRA 0DTE tools
================================================================================

Tool Name                      Signals   Win%    Target  Gap      Avg Ret    Sharpe
----------------------------------------------------------------------------------
#1   Gamma Wall Pin            1,234     96.8%   96.8%   +0.0%    2.70%     3.24   ✅
#2   Charm Flow Nuclear        892       94.2%   94.2%   +0.0%    2.80%     3.45   ✅
...
```

---

## 🚀 **Deployment**

### **Deploy to Modal (Recommended)**

```bash
modal deploy deploy/modal/modal_deployment.py --name pulse-alpha-2025
```

**Custom Domain:**
```bash
# Add CNAME: pulse.trading → pulse-alpha-2025.modal.run
```

### **Run Locally**

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 📡 **API Endpoints**

### **Execute Tool**
```bash
POST /api/v1/execute
{
  "tool_id": 1,
  "user_id": "user123",
  "data": { ... }
}
```

### **Implemented Endpoints (Current Server)**
```text
POST /api/v1/execute
POST /api/v1/consensus
POST /api/v1/paper/{user_id}/positions
POST /api/v1/paper/{user_id}/positions/{position_id}/close
GET  /api/v1/paper/{user_id}/positions
GET  /api/v1/paper/{user_id}/account
GET  /health
GET  /health/security
GET  /
WS   /ws
```

### **Get Consensus**
```bash
POST /api/v1/consensus
{
  "tool_ids": [1, 2, 3],
  "data": { ... }
}
```

### **Health Check**
```bash
GET /health
```

### **Paper Trading API**
```bash
POST /api/v1/paper/{user_id}/positions
POST /api/v1/paper/{user_id}/positions/{position_id}/close
GET  /api/v1/paper/{user_id}/positions
GET  /api/v1/paper/{user_id}/account
```


### **Regime + Time Aware Signal Context**

Consensus and tool responses now include a context block that classifies:
- **Time bucket**: `0dte`, `weekly`, `swing`, `long_dated`
- **Market regime**: `high_volatility`, `trend`, `range`
- **Pricing model family** used for that horizon (intraday microstructure Greeks vs Black-Scholes term proxy vs long-dated surface proxy)

This enables day-specific routing and model selection in execution pipelines.

### **WebSocket Live Updates**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  console.log('Signal:', JSON.parse(event.data));
};

ws.onopen = () => {
  ws.send(JSON.stringify({
    tool_id: 10,
    symbol: 'SPY',
    data: { price: 530.2, rsi: 62 }
  }));
};
```

---

## 🔔 **Discord Alerts**

Configure webhooks in `.env`:

```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook
```

**Alert Format:**
```
🚨 SIGNAL FIRED: Gamma Wall Pin Collapse
━━━━━━━━━━━━━━━━━━━━━━━━
📊 Signal Score: 0.87
💰 Spot: $450.25
📈 Greeks:
   • Gamma: 0.0234
   • Charm: -1.2B
   • Vanna: 1.8B
━━━━━━━━━━━━━━━━━━━━━━━━
⏰ 2025-12-09 13:45:23 UTC
```

---

## 🧪 **Development**

### **Add/Refine a Tool (Current Architecture)**

The current runtime tool path is `production_tool_executor.py`.
Add a new `_signal_*` function and register it in `tool_logic`.

```python
from production_tool_executor import _extract_float, _clamp_score


def _signal_my_tool(data):
    price = _extract_float(data, "price", 0.0)
    vwap = _extract_float(data, "vwap", price)
    signal = 1 if price > vwap else -1 if price < vwap else 0
    score = _clamp_score(0.5 + min(0.5, abs(price - vwap) / max(vwap, 1e-9)))
    return signal, score, {"price": price, "vwap": vwap}

# then register:
# tool_logic[11] = _signal_my_tool
```

### **Use Greek Engines Correctly**

```python
from core.trainer.engines import GreekEngines

gamma = GreekEngines.gamma(S=530.0, K=530.0, T=1/365, r=0.045, sigma=0.22)
charm = GreekEngines.charm(S=530.0, K=530.0, T=1/365, r=0.045, sigma=0.22)
vanna = GreekEngines.vanna(S=530.0, K=530.0, T=1/365, r=0.045, sigma=0.22)
```

## ⚠️ **Disclaimers**

**EDUCATIONAL & RESEARCH PURPOSES ONLY**

- This platform is for educational and research purposes only
- Past simulated performance does NOT guarantee future results
- Options trading involves substantial risk of loss
- All simulation results are hypothetical and based on synthetic data
- No warranties or guarantees are provided
- Users are responsible for their own trading decisions
- Always consult a licensed financial advisor before trading

**Backtesting Methodology:**
- 100,000+ Monte Carlo simulations per tool
- Synthetic market data generation
- Realistic volatility clustering (GARCH)
- Statistical rigor prioritized over marketing claims

---

## 📚 **Documentation**

- [Architecture Overview](docs/ARCHITECTURE_OVERVIEW.md)
- [Security Setup Guide](SECURITY_SETUP.md)
- [Repository Manifest](REPOSITORY_MANIFEST.md)
- [Performance Testing](PERFORMANCE_TESTING.md)

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Run tests (`pytest -q`)
5. Open a pull request
