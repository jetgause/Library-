from pathlib import Path


def test_readme_mentions_current_core_endpoints():
    content = Path('README.md').read_text()
    expected = [
        'POST /api/v1/execute',
        'POST /api/v1/consensus',
        'POST /api/v1/paper/{user_id}/positions',
        'POST /api/v1/paper/{user_id}/positions/{position_id}/close',
        'GET  /api/v1/paper/{user_id}/positions',
        'GET  /api/v1/paper/{user_id}/account',
        'WS   /ws',
    ]
    for route in expected:
        assert route in content
