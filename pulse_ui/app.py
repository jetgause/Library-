import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse

# Import core components
from pulse_core.pulse_engine import PulseEngine
from pulse_core.models import (
    TaskCreate,
    TaskUpdate,
    TaskResponse,
    TaskListResponse,
    AnalyticsResponse,
    NotificationResponse,
    HealthResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="Pulse Task Manager",
    description="AI-powered intelligent task management system",
    version="1.0.0"
)

# FMA PATCH 1: HTTPS Enforcement & Security Headers (CRITICAL)
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Force HTTPS and add security headers."""
    # Force HTTPS redirect (except localhost)
    if not request.url.scheme == "https" and request.url.hostname not in ["localhost", "127.0.0.1"]:
        url = request.url.replace(scheme="https")
        return RedirectResponse(url, status_code=301)
    
    response = await call_next(request)
    
    # Add security headers
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=(), payment=(self)"
    
    # Content Security Policy
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdn.jsdelivr.net https://pay.google.com; "
        "style-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdnjs.cloudflare.com; "
        "img-src 'self' data: https:; "
        "font-src 'self' https://cdnjs.cloudflare.com; "
        "connect-src 'self' wss: https://pay.google.com; "
        "frame-src https://pay.google.com; "
        "object-src 'none'; "
        "base-uri 'self';"
    )
    
    return response

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates and static files
templates = Jinja2Templates(directory="pulse_ui/templates")
app.mount("/static", StaticFiles(directory="pulse_ui/static"), name="static")

# Initialize Pulse Engine
pulse_engine = PulseEngine()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(task: TaskCreate):
    """Create a new task."""
    try:
        created_task = await pulse_engine.create_task(task)
        await manager.broadcast({
            "type": "task_created",
            "data": created_task
        })
        return created_task
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks", response_model=TaskListResponse)
async def get_tasks(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    category: Optional[str] = None
):
    """Get all tasks with optional filters."""
    try:
        tasks = await pulse_engine.get_tasks(
            status=status,
            priority=priority,
            category=category
        )
        return {"tasks": tasks, "total": len(tasks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Get a specific task by ID."""
    try:
        task = await pulse_engine.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/tasks/{task_id}", response_model=TaskResponse)
async def update_task(task_id: str, task_update: TaskUpdate):
    """Update an existing task."""
    try:
        updated_task = await pulse_engine.update_task(task_id, task_update)
        if not updated_task:
            raise HTTPException(status_code=404, detail="Task not found")
        await manager.broadcast({
            "type": "task_updated",
            "data": updated_task
        })
        return updated_task
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task."""
    try:
        success = await pulse_engine.delete_task(task_id)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found")
        await manager.broadcast({
            "type": "task_deleted",
            "data": {"task_id": task_id}
        })
        return {"message": "Task deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Get task analytics and insights."""
    try:
        analytics = await pulse_engine.get_analytics()
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/notifications", response_model=List[NotificationResponse])
async def get_notifications():
    """Get recent notifications."""
    try:
        notifications = await pulse_engine.get_notifications()
        return notifications
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back or process the message
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    await pulse_engine.initialize()
    print("🚀 Pulse Task Manager started successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    await pulse_engine.cleanup()
    print("👋 Pulse Task Manager shutdown complete!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
