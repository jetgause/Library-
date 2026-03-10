from production_tool_executor import execute_tool, get_consensus


def _sample_chain_payload():
    return {
        "spot": 100.0,
        "price": 100.0,
        "ret_5m": 0.003,
        "iv": 0.22,
        "iv_change": 0.04,
        "risk_free_rate": 0.045,
        "tte_days": 1.0,
        "strikes": [95, 100, 105],
        "oi_call": [12000, 18000, 14000],
        "oi_put": [9000, 7000, 5000],
    }


def test_advanced_tools_use_options_metrics():
    data = _sample_chain_payload()

    for tool_id in (6, 7, 8, 9, 10):
        res = execute_tool(tool_id, data, "SPY")
        assert res is not None
        assert res["signal"] in (-1, 0, 1)
        assert 0.0 <= res["score"] <= 1.0
        assert isinstance(res["metadata"], dict)


def test_consensus_includes_advanced_tools():
    data = _sample_chain_payload()
    consensus = get_consensus([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], data, "SPY")
    assert consensus["tool_count"] == 10
    assert consensus["consensus_signal"] in (-1, 0, 1)
    assert len(consensus["tools"]) == 10
