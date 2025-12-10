from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add parent directory to path to import from pulse_backend
sys.path.append(str(Path(__file__).parent.parent))

from pulse_backend.db_manager import DatabaseManager
from pulse_backend.metrics_collector import MetricsCollector
from pulse_backend.alert_manager import AlertManager

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pulse Monitoring Dashboard",
    description="Real-time system monitoring and alerting dashboard",
    version="1.0.0"
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and static files
templates = Jinja2Templates(directory="pulse_ui/templates")
app.mount("/static", StaticFiles(directory="pulse_ui/static"), name="static")

# Initialize backend components
db_manager = DatabaseManager()
metrics_collector = MetricsCollector(db_manager)
alert_manager = AlertManager(db_manager)

# Background task flag
background_task_running = False

import time
from collections import defaultdict
from typing import Optional

# WebSocket Security Configuration
WS_MAX_CONNECTIONS_PER_IP = 5
WS_MAX_MESSAGE_SIZE = 10240  # 10KB
WS_MAX_MESSAGES_PER_MINUTE = 60
WS_CONNECTION_TIMEOUT = 300  # 5 minutes idle timeout

# WebSocket connection manager with security
class SecureConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}
        self.connections_per_ip: Dict[str, int] = defaultdict(int)
        self.message_counts: Dict[WebSocket, List[float]] = defaultdict(list)
        self.last_activity: Dict[WebSocket, float] = {}
    
    def get_client_ip(self, websocket: WebSocket) -> str:
        """Extract client IP from WebSocket connection."""
        if websocket.client:
            return websocket.client.host
        return "unknown"
    
    def check_connection_limit(self, client_ip: str) -> bool:
        """Check if client IP has exceeded connection limit."""
        return self.connections_per_ip[client_ip] < WS_MAX_CONNECTIONS_PER_IP
    
    def check_rate_limit(self, websocket: WebSocket) -> bool:
        """Check if websocket has exceeded message rate limit."""
        now = time.time()
        # Clean old timestamps (older than 1 minute)
        self.message_counts[websocket] = [
            ts for ts in self.message_counts[websocket]
            if now - ts < 60
        ]
        # Check rate limit
        return len(self.message_counts[websocket]) < WS_MAX_MESSAGES_PER_MINUTE
    
    def record_message(self, websocket: WebSocket):
        """Record message timestamp for rate limiting."""
        self.message_counts[websocket].append(time.time())
        self.last_activity[websocket] = time.time()
    
    def check_idle_timeout(self, websocket: WebSocket) -> bool:
        """Check if connection has been idle too long."""
        last_activity = self.last_activity.get(websocket, time.time())
        return (time.time() - last_activity) > WS_CONNECTION_TIMEOUT
    
    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None) -> bool:
        """
        Accept WebSocket connection with security checks.
        
        Args:
            websocket: WebSocket connection
            user_id: Optional authenticated user ID
            
        Returns:
            True if connection accepted, False if rejected
        """
        client_ip = self.get_client_ip(websocket)
        
        # Check connection limit per IP
        if not self.check_connection_limit(client_ip):
            await websocket.close(code=1008, reason="Too many connections from your IP")
            return False
        
        # Accept connection
        await websocket.accept()
        
        # Track connection
        self.active_connections.append(websocket)
        self.connections_per_ip[client_ip] += 1
        self.last_activity[websocket] = time.time()
        
        # Store connection info
        self.connection_info[websocket] = {
            "client_ip": client_ip,
            "user_id": user_id,
            "connected_at": datetime.now().isoformat(),
        }
        
        return True
    
    def disconnect(self, websocket: WebSocket):
        """Disconnect and cleanup WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Cleanup tracking
        if websocket in self.connection_info:
            client_ip = self.connection_info[websocket]["client_ip"]
            self.connections_per_ip[client_ip] = max(0, self.connections_per_ip[client_ip] - 1)
            del self.connection_info[websocket]
        
        if websocket in self.message_counts:
            del self.message_counts[websocket]
        
        if websocket in self.last_activity:
            del self.last_activity[websocket]
    
    async def broadcast(self, message: dict, exclude: Optional[WebSocket] = None):
        """
        Broadcast message to all connected clients.
        
        Args:
            message: Message to broadcast
            exclude: Optional WebSocket to exclude from broadcast
        """
        disconnected = []
        
        for connection in self.active_connections:
            if connection == exclude:
                continue
            
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        # Cleanup disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def send_to_user(self, user_id: str, message: dict):
        """Send message to specific user by user_id."""
        for connection, info in self.connection_info.items():
            if info.get("user_id") == user_id:
                try:
                    await connection.send_json(message)
                except Exception:
                    self.disconnect(connection)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": len(self.active_connections),
            "connections_per_ip": dict(self.connections_per_ip),
            "active_users": len(set(
                info.get("user_id") 
                for info in self.connection_info.values() 
                if info.get("user_id")
            ))
        }

manager = SecureConnectionManager()

# Updated WebSocket endpoint with security
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Secure WebSocket endpoint for real-time updates.
    
    Security features:
    - Connection limit per IP (max 5)
    - Message rate limiting (60/minute)
    - Message size validation (10KB max)
    - Idle timeout (5 minutes)
    - Input validation and sanitization
    """
    # TODO: Extract user_id from auth token/session
    user_id = None  # Replace with actual authentication
    
    # Attempt to connect with security checks
    connected = await manager.connect(websocket, user_id)
    if not connected:
        return
    
    client_ip = manager.get_client_ip(websocket)
    
    try:
        while True:
            # Check for idle timeout
            if manager.check_idle_timeout(websocket):
                await websocket.close(code=1000, reason="Connection idle timeout")
                break
            
            # Receive message with timeout
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})
                continue
            
            # Validate message size
            if len(data) > WS_MAX_MESSAGE_SIZE:
                await websocket.send_json({
                    "type": "error",
                    "message": "Message too large"
                })
                continue
            
            # Check rate limit
            if not manager.check_rate_limit(websocket):
                await websocket.send_json({
                    "type": "error",
                    "message": "Rate limit exceeded. Max 60 messages per minute."
                })
                await asyncio.sleep(1)  # Throttle
                continue
            
            # Record message for rate limiting
            manager.record_message(websocket)
            
            # Parse and validate message
            try:
                message = json.loads(data)
                
                # Validate message structure
                if not isinstance(message, dict):
                    raise ValueError("Message must be a JSON object")
                
                message_type = message.get("type")
                if not message_type:
                    raise ValueError("Message must have a 'type' field")
                
                # Process message based on type
                if message_type == "ping":
                    await websocket.send_json({"type": "pong"})
                
                elif message_type == "subscribe":
                    # Subscribe to specific event types
                    await websocket.send_json({
                        "type": "subscribed",
                        "message": "Successfully subscribed to updates"
                    })
                
                elif message_type == "echo":
                    # Echo back for testing (remove in production)
                    await websocket.send_json({
                        "type": "echo",
                        "data": message.get("data", "")
                    })
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    })
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except ValueError as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": "Error processing message"
                })
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error for {client_ip}: {str(e)}")
    finally:
        manager.disconnect(websocket)

# Add WebSocket statistics endpoint
@app.get("/api/ws/stats")
@limiter.limit("30/minute")
async def get_websocket_stats(request: Request):
    """Get WebSocket connection statistics (admin only)."""
    # TODO: Add admin authentication check
    return manager.get_stats()

# Background task for metrics collection
async def collect_metrics_task():
    """Background task to collect metrics periodically."""
    global background_task_running
    background_task_running = True
    
    while background_task_running:
        try:
            # Collect metrics
            metrics = metrics_collector.collect_all_metrics()
            
            # Store metrics in database
            db_manager.store_metrics(metrics)
            
            # Check for alerts
            alerts = alert_manager.check_alerts(metrics)
            
            # Broadcast metrics to connected clients
            await manager.broadcast({
                "type": "metrics_update",
                "data": metrics,
                "timestamp": datetime.now().isoformat()
            })
            
            # Broadcast alerts if any
            if alerts:
                await manager.broadcast({
                    "type": "alerts",
                    "data": alerts,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Wait before next collection
            await asyncio.sleep(5)  # Collect every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in metrics collection task: {str(e)}")
            await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting Pulse Monitoring Dashboard...")
    
    # Initialize database
    db_manager.initialize()
    
    # Start background metrics collection
    asyncio.create_task(collect_metrics_task())
    
    logger.info("Pulse Monitoring Dashboard started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    global background_task_running
    background_task_running = False
    logger.info("Pulse Monitoring Dashboard shutting down...")

# REST API Endpoints

@app.get("/", response_class=HTMLResponse)
@limiter.limit("60/minute")
async def read_root(request: Request):
    """Render the main dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/metrics/current")
@limiter.limit("120/minute")
async def get_current_metrics(request: Request):
    """Get current system metrics."""
    try:
        metrics = metrics_collector.collect_all_metrics()
        return JSONResponse(content={
            "status": "success",
            "data": metrics,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting current metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/history")
@limiter.limit("60/minute")
async def get_metrics_history(
    request: Request,
    metric_type: str = "cpu",
    hours: int = 24
):
    """Get historical metrics data."""
    try:
        history = db_manager.get_metrics_history(metric_type, hours)
        return JSONResponse(content={
            "status": "success",
            "data": history,
            "metric_type": metric_type,
            "hours": hours
        })
    except Exception as e:
        logger.error(f"Error getting metrics history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts")
@limiter.limit("60/minute")
async def get_alerts(request: Request, status: str = "active"):
    """Get alerts filtered by status."""
    try:
        alerts = db_manager.get_alerts(status)
        return JSONResponse(content={
            "status": "success",
            "data": alerts,
            "count": len(alerts)
        })
    except Exception as e:
        logger.error(f"Error getting alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/alerts/{alert_id}/acknowledge")
@limiter.limit("30/minute")
async def acknowledge_alert(request: Request, alert_id: int):
    """Acknowledge an alert."""
    try:
        alert_manager.acknowledge_alert(alert_id)
        return JSONResponse(content={
            "status": "success",
            "message": f"Alert {alert_id} acknowledged"
        })
    except Exception as e:
        logger.error(f"Error acknowledging alert: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
@limiter.limit("120/minute")
async def health_check(request: Request):
    """Health check endpoint."""
    return JSONResponse(content={
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
