"""
PULSE UI - Main Application
===========================

FastAPI web application for PULSE Trading Platform UI.
Provides web interface for tool management, monitoring, and analytics.

Author: jetgause
Created: 2025-12-10
"""

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn
from datetime import datetime
from typing import List, Dict, Any
import json
import asyncio


# Initialize FastAPI app
app = FastAPI(
    title="PULSE Trading Platform",
    description="Web UI for PULSE tool management and monitoring",
    version="1.0.0"
)

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Setup templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# In-memory data store (replace with actual database in production)
class DataStore:
    def __init__(self):
        self.tools = {}
        self.metrics = {
            'cpu_usage': 45.2,
            'memory_usage': 62.1,
            'active_tools': 42,
            'requests_per_second': 275,
            'avg_response_time': 850,
            'error_rate': 2.1
        }
        self.activity_log = []
        self.connected_clients = []
    
    def add_tool(self, tool_data: Dict[str, Any]) -> str:
        tool_id = f"tool_{len(self.tools) + 1}"
        self.tools[tool_id] = {
            'id': tool_id,
            **tool_data,
            'created_at': datetime.utcnow().isoformat()
        }
        self.log_activity('tool_created', {'tool_id': tool_id, 'name': tool_data.get('name')})
        return tool_id
    
    def get_tool(self, tool_id: str) -> Dict[str, Any]:
        return self.tools.get(tool_id)
    
    def list_tools(self, category: str = None) -> List[Dict[str, Any]]:
        tools = list(self.tools.values())
        if category:
            tools = [t for t in tools if t.get('category') == category]
        return tools
    
    def update_metrics(self, new_metrics: Dict[str, Any]):
        self.metrics.update(new_metrics)
    
    def log_activity(self, activity_type: str, data: Dict[str, Any]):
        self.activity_log.append({
            'timestamp': datetime.utcnow().isoformat(),
            'type': activity_type,
            'data': data
        })
        # Keep only last 100 entries
        if len(self.activity_log) > 100:
            self.activity_log = self.activity_log[-100:]


# Global data store
store = DataStore()


# Initialize with sample tools
def init_sample_data():
    """Initialize with sample tools for demonstration."""
    sample_tools = [
        {'name': 'Data Cleaner', 'category': 'Data Processing', 'tier': 1, 'status': 'active'},
        {'name': 'ML Predictor', 'category': 'Machine Learning', 'tier': 2, 'status': 'active'},
        {'name': 'API Monitor', 'category': 'Monitoring', 'tier': 1, 'status': 'active'},
        {'name': 'Report Generator', 'category': 'Analytics', 'tier': 3, 'status': 'active'},
    ]
    
    for tool in sample_tools:
        store.add_tool(tool)


# Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "page_title": "Dashboard",
        "metrics": store.metrics,
        "tool_count": len(store.tools)
    })


@app.get("/tools", response_class=HTMLResponse)
async def tools_page(request: Request):
    """Tool browser page."""
    return templates.TemplateResponse("tools.html", {
        "request": request,
        "page_title": "Tools",
        "tools": store.list_tools()
    })


@app.get("/tools/create", response_class=HTMLResponse)
async def tool_create_page(request: Request):
    """Tool creation wizard page."""
    return templates.TemplateResponse("tool_create.html", {
        "request": request,
        "page_title": "Create Tool"
    })


@app.get("/monitoring", response_class=HTMLResponse)
async def monitoring_page(request: Request):
    """Live monitoring page."""
    return templates.TemplateResponse("monitoring.html", {
        "request": request,
        "page_title": "Monitoring",
        "metrics": store.metrics
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page."""
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "page_title": "Settings"
    })


# API Routes
@app.get("/api/tools")
async def api_list_tools(category: str = None):
    """API endpoint to list tools."""
    return JSONResponse({
        "status": "success",
        "tools": store.list_tools(category)
    })


@app.get("/api/tools/{tool_id}")
async def api_get_tool(tool_id: str):
    """API endpoint to get a specific tool."""
    tool = store.get_tool(tool_id)
    if tool:
        return JSONResponse({"status": "success", "tool": tool})
    return JSONResponse({"status": "error", "message": "Tool not found"}, status_code=404)


@app.post("/api/tools")
async def api_create_tool(request: Request):
    """API endpoint to create a new tool."""
    data = await request.json()
    tool_id = store.add_tool(data)
    return JSONResponse({
        "status": "success",
        "tool_id": tool_id,
        "message": "Tool created successfully"
    })


@app.get("/api/metrics")
async def api_get_metrics():
    """API endpoint to get current metrics."""
    return JSONResponse({
        "status": "success",
        "metrics": store.metrics,
        "timestamp": datetime.utcnow().isoformat()
    })


@app.get("/api/activity")
async def api_get_activity():
    """API endpoint to get activity log."""
    return JSONResponse({
        "status": "success",
        "activity": store.activity_log[-20:]  # Last 20 entries
    })


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time metric updates."""
    await websocket.accept()
    store.connected_clients.append(websocket)
    
    try:
        while True:
            # Send metrics every 2 seconds
            await websocket.send_json({
                "type": "metrics_update",
                "data": store.metrics,
                "timestamp": datetime.utcnow().isoformat()
            })
            await asyncio.sleep(2)
    
    except WebSocketDisconnect:
        store.connected_clients.remove(websocket)


@app.on_event("startup")
async def startup_event():
    """Initialize app on startup."""
    init_sample_data()
    print("🚀 PULSE UI Server Started")
    print(f"📊 Dashboard: http://localhost:8000")
    print(f"🔧 Tools: http://localhost:8000/tools")
    print(f"📡 Monitoring: http://localhost:8000/monitoring")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("👋 PULSE UI Server Shutting Down")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "tools_count": len(store.tools),
        "connected_clients": len(store.connected_clients)
    })


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
