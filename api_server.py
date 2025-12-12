"""
PULSE Trading Platform - FastAPI Server
Production-ready API server with WebSocket support, user management, and tool execution
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
import json
from datetime import datetime
import uvicorn
import logging

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
