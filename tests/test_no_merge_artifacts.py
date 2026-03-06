from pathlib import Path


def test_api_server_has_no_merge_conflict_markers_or_duplicate_blocks():
    content = Path('api_server.py').read_text()

    # Ensure merge conflict artifacts are absent
    assert '<<<<<<<' not in content
    assert '=======' not in content
    assert '>>>>>>>' not in content

    # Ensure key API models/endpoints appear only once (guards against accidental duplicated paste blocks)
    assert content.count('class PaperPositionRequest(BaseModel):') == 1
    assert content.count('class PaperCloseRequest(BaseModel):') == 1
    assert content.count('class PaperAccountResponse(BaseModel):') == 1
    assert content.count('def _get_paper_engine(user_id: str) -> PaperTradingEngine:') == 1
    assert content.count('@app.post("/api/v1/consensus", response_model=ConsensusResponse)') == 1
    assert content.count('@app.websocket("/ws")') == 1
