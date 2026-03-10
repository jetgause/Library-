from production_tool_executor import execute_tool


def test_tool_output_contains_time_and_regime_context():
    data = {
        "rsi": 25,
        "tte_days": 0.5,
        "vix": 36,
        "realized_vol": 0.5,
    }
    res = execute_tool(1, data, "SPY")
    assert res is not None
    meta = res["metadata"]
    assert meta["time_bucket"] == "0dte"
    assert meta["market_regime"] == "high_volatility"
    assert meta["pricing_model"] == "intraday_microstructure_greeks"


def test_pricing_model_changes_by_timeframe():
    short = execute_tool(2, {"macd": 1, "signal_line": 0, "tte_days": 3}, "SPY")
    long = execute_tool(2, {"macd": 1, "signal_line": 0, "tte_days": 90}, "SPY")
    assert short is not None and long is not None
    assert short["metadata"]["pricing_model"] == "black_scholes_greeks"
    assert long["metadata"]["pricing_model"] == "vol_surface_long_dated_proxy"
