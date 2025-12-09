# PULSE Trading Platform

**Advanced 0DTE Options Trading System with Reinforcement Learning**

A sophisticated, production-ready options trading platform featuring 52 specialized trading tools, 13 Greek calculation engines, full SABR volatility surfaces, and autonomous reinforcement learning optimization.

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

## ⚡ **Quick Start**

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

### **WebSocket Live Updates**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
  console.log('Signal:', JSON.parse(event.data));
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

### **Create a New Tool**

```python
from tools.library.base import BaseTool

class MyCustomTool(BaseTool):
    TOOL_METADATA = {
        "id": 53,
        "name": "My Custom Tool",
        "tier": "pro_plus",
        "description": "Custom strategy"
    }
    
    def run(self, data):
        # Access Greek engines
        from core.trainer.engines import GreekEngines
        gamma = GreekEngines.calculate_gamma(data)
        
        # Your logic
        if gamma > threshold:
            return 1  # Signal
        return 0  # No signal
```

### **Train & Distill**

```python
from core.trainer.trainer import train_rl_agent
from core.distillation.distillation_engine import distill_to_tree

# Train
policy = train_rl_agent("my_tool", n_episodes=5000)

# Distill to decision tree
tree = distill_to_tree(policy, max_depth=5)

# Auto-approve if improvement > 15%
if tree.improvement > 0.15:
    tree.deploy()
```

---

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

- [Tool Development Manual](docs/TOOL_DEVELOPMENT_MANUAL.md)
- [Backtest Methodology](docs/BACKTEST_PROOF.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [API Reference](docs/API_REFERENCE.md)

---

## 📜 **License**

MIT License - See LICENSE file

---

## 🤝 **Contributing**

Contributions welcome! Please read CONTRIBUTING.md first.

---

## 📧 **Contact**

- **GitHub**: [@jetgause](https://github.com/jetgause)
- **Repository**: [Library-](https://github.com/jetgause/Library-)

---

**Built with ❤️ for the options trading community**