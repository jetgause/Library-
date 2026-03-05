from fastapi.testclient import TestClient

from api_server import app


def test_websocket_signal_stream_roundtrip():
    client = TestClient(app)
    with client.websocket_connect('/ws') as ws:
        welcome = ws.receive_json()
        assert welcome['type'] == 'welcome'

        ws.send_json({
            'tool_id': 1,
            'symbol': 'SPY',
            'data': {'rsi': 20, 'price': 100},
        })
        signal_msg = ws.receive_json()
        assert signal_msg['type'] == 'signal'
        payload = signal_msg['payload']
        assert payload['tool_id'] == 1
        assert payload['signal'] in (-1, 0, 1)
