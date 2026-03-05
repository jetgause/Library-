"""
PULSE Trading Platform - FastAPI Server
Production-ready API server with WebSocket support, user management, and tool execution
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import json
from datetime import datetime
import uvicorn
import logging
import os
import secrets

# Ensure local/dev runs have a secure key before strict config import
os.environ.setdefault("SECRET_KEY", secrets.token_urlsafe(32))

# Import internal modules
from config import *
from paper_trading import PaperTradingEngine

# Setup logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PULSE Trading API",
    description="Advanced 0DTE Options Trading Platform API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
paper_trading_engines: Dict[str, PaperTradingEngine] = {}
active_websockets: List[WebSocket] = []


# REQUEST/RESPONSE MODELS
class ToolExecuteRequest(BaseModel):
    tool_id: int = Field(..., description="Tool ID to execute (1-52)")
    user_id: str = Field(..., description="User identifier")
    symbol: str = Field(default="SPY", description="Trading symbol")
    data: Dict[str, Any] = Field(default={}, description="Market data for tool execution")


class ToolResponse(BaseModel):
    tool_id: int
    tool_name: str
    signal: int
    score: float
    timestamp: str
    metadata: Dict[str, Any] = {}

class PaperPositionRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    quantity: int = Field(..., gt=0)
    price: float = Field(..., gt=0)
    side: str = Field(default="long", pattern="^(long|short)$")
    stop_loss: Optional[float] = Field(default=None, gt=0)
    take_profit: Optional[float] = Field(default=None, gt=0)
    tool_id: Optional[int] = None
    tool_name: Optional[str] = None


class PaperCloseRequest(BaseModel):
    price: float = Field(..., gt=0)
    exit_reason: str = Field(default="manual", max_length=50)


class PaperAccountResponse(BaseModel):
    user_id: str
    open_positions: List[Dict[str, Any]]
    total_open_positions: int
    total_realized_pnl: float
    total_equity: float
    stats: Dict[str, Any]


def _get_paper_engine(user_id: str) -> PaperTradingEngine:
    """Get or create a paper trading engine for a user."""
    if user_id not in paper_trading_engines:
        safe_user = "".join(ch for ch in user_id if ch.isalnum() or ch in ("_", "-")) or "default"
        db_dir = Path("data")
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / f"paper_trading_{safe_user}.db"
        paper_trading_engines[user_id] = PaperTradingEngine(db_path=str(db_path))
    return paper_trading_engines[user_id]


@app.post("/api/v1/execute", response_model=ToolResponse)
async def execute_tool(request: ToolExecuteRequest):
    """Execute a single trading tool"""
    try:
        from production_tool_executor import execute_tool as exec_tool
        
        result = exec_tool(
            tool_id=request.tool_id,
            data=request.data,
            symbol=request.symbol
        )
        
        if result is None:
            raise HTTPException(status_code=404, detail=f"Tool {request.tool_id} not found")
        
        logger.info(f"Tool {request.tool_id} executed: signal={result['signal']}")
        
        return ToolResponse(
            tool_id=request.tool_id,
            tool_name=result.get('tool_name', f'Tool {request.tool_id}'),
            signal=result['signal'],
            score=result.get('score', 0.0),
            timestamp=datetime.now().isoformat(),
            metadata=result.get('metadata', {})
        )
        
    except Exception as e:
        logger.error(f"Error executing tool: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/paper/{user_id}/positions")
async def open_paper_position(user_id: str, request: PaperPositionRequest):
    """Open a paper trading position for a user."""
    engine = _get_paper_engine(user_id)
    position = engine.open_position(
        symbol=request.symbol,
        quantity=request.quantity,
        price=request.price,
        side=request.side,
        stop_loss=request.stop_loss,
        take_profit=request.take_profit,
        tool_id=request.tool_id,
        tool_name=request.tool_name,
    )
    if position is None:
        raise HTTPException(status_code=400, detail="Insufficient capital to open position")
    return {"status": "opened", "position": position.to_dict(), "cash": engine.cash}


@app.post("/api/v1/paper/{user_id}/positions/{position_id}/close")
async def close_paper_position(user_id: str, position_id: str, request: PaperCloseRequest):
    """Close a paper trading position for a user."""
    engine = _get_paper_engine(user_id)
    trade = engine.close_position(position_id=position_id, price=request.price, exit_reason=request.exit_reason)
    if trade is None:
        raise HTTPException(status_code=404, detail=f"Position {position_id} not found")
    return {"status": "closed", "trade": trade.to_dict(), "cash": engine.cash}


@app.get("/api/v1/paper/{user_id}/positions")
async def list_open_paper_positions(user_id: str):
    """List open paper positions for a user."""
    engine = _get_paper_engine(user_id)
    positions = engine.get_open_positions()
    return {"user_id": user_id, "positions": positions, "count": len(positions)}


@app.get("/api/v1/paper/{user_id}/account", response_model=PaperAccountResponse)
async def get_paper_account(user_id: str):
    """Get account summary and performance stats for a user's paper account."""
    engine = _get_paper_engine(user_id)
    positions = engine.get_open_positions()
    stats = engine.get_performance_stats()
    return PaperAccountResponse(
        user_id=user_id,
        open_positions=positions,
        total_open_positions=len(positions),
        total_realized_pnl=engine.get_total_pnl(),
        total_equity=engine.get_total_equity(),
        stats=stats,
    )


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with security status."""
    import config
    
    security_checks = {
        "secret_key_configured": len(config.SECRET_KEY) >= 32,
        "cors_secure": "*" not in config.ALLOWED_ORIGINS,
        "api_localhost_bound": config.API_HOST in ["127.0.0.1", "localhost"],
    }
    
    all_secure = all(security_checks.values())
    
    return {
        "status": "healthy" if all_secure else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "environment": config.ENVIRONMENT,
        "security": {
            "status": "secure" if all_secure else "warnings",
            "checks": security_checks
        }
    }


@app.get("/health/security")
async def security_health():
    """Detailed security health endpoint."""
    import config
    
    return {
        "secret_key_length": len(config.SECRET_KEY),
        "cors_origins_count": len(config.ALLOWED_ORIGINS),
        "has_wildcard_cors": "*" in config.ALLOWED_ORIGINS,
        "api_host": config.API_HOST,
        "is_production": config.IS_PRODUCTION,
        "environment": config.ENVIRONMENT
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "PULSE Trading API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=API_HOST,
        port=API_PORT,
        workers=API_WORKERS,
        reload=False,
        log_level=LOG_LEVEL.lower()
    )
