from production_tool_executor import execute_tool, get_consensus


def test_execute_tool_is_deterministic_for_same_input():
    data = {"rsi": 22, "price": 101.0}
    a = execute_tool(1, data, "SPY")
    b = execute_tool(1, data, "SPY")

    assert a is not None and b is not None
    assert a["signal"] == b["signal"]
    assert a["score"] == b["score"]


def test_get_consensus_returns_expected_shape():
    data = {
        "rsi": 28,
        "macd": 0.5,
        "signal_line": 0.1,
        "volume": 2000,
        "avg_volume": 1000,
        "price": 105,
        "prev_price": 102,
        "sma_fast": 104,
        "sma_slow": 100,
        "vwap": 103,
    }
    res = get_consensus([1, 2, 3, 4, 5], data, "SPY")
    assert res["tool_count"] == 5
    assert res["consensus_signal"] in (-1, 0, 1)
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["tools"]) == 5
