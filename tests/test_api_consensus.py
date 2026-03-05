from fastapi.testclient import TestClient

from api_server import app


def test_consensus_endpoint_returns_weighted_signal():
    client = TestClient(app)
    payload = {
        "tool_ids": [1, 2, 3, 4, 5],
        "symbol": "SPY",
        "data": {
            "rsi": 26,
            "macd": 0.8,
            "signal_line": 0.2,
            "volume": 2200,
            "avg_volume": 1000,
            "price": 108,
            "prev_price": 104,
            "sma_fast": 107,
            "sma_slow": 102,
            "vwap": 106,
        },
    }

    resp = client.post("/api/v1/consensus", json=payload)
    assert resp.status_code == 200
    body = resp.json()

    assert body["symbol"] == "SPY"
    assert body["tool_count"] == 5
    assert body["consensus_signal"] in (-1, 0, 1)
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["tools"]) == 5
