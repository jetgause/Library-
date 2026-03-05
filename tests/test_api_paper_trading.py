from fastapi.testclient import TestClient

from api_server import app


def test_paper_trading_position_lifecycle():
    client = TestClient(app)
    user_id = "test_user_api"

    open_resp = client.post(
        f"/api/v1/paper/{user_id}/positions",
        json={"symbol": "SPY", "quantity": 1, "price": 100.0, "side": "long"},
    )
    assert open_resp.status_code == 200
    position_id = open_resp.json()["position"]["position_id"]

    list_resp = client.get(f"/api/v1/paper/{user_id}/positions")
    assert list_resp.status_code == 200
    assert list_resp.json()["count"] >= 1

    close_resp = client.post(
        f"/api/v1/paper/{user_id}/positions/{position_id}/close",
        json={"price": 101.0, "exit_reason": "manual"},
    )
    assert close_resp.status_code == 200
    assert close_resp.json()["trade"]["position_id"] == position_id

    account_resp = client.get(f"/api/v1/paper/{user_id}/account")
    assert account_resp.status_code == 200
    body = account_resp.json()
    assert body["user_id"] == user_id
    assert "total_equity" in body
    assert "stats" in body
