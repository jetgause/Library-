from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import sys
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add parent directory to path to import book_tracker
sys.path.append(str(Path(__file__).parent.parent))
from book_tracker import BookTracker

app = FastAPI()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour", "20/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize book tracker
tracker = BookTracker()

# Pydantic models
class BookRating(BaseModel):
    rating: int

class BookStatus(BaseModel):
    status: str

class BookSearch(BaseModel):
    query: str

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    html_file = static_dir / "index.html"
    return FileResponse(html_file)

@app.get("/api/books/current")
@limiter.limit("30/minute")
async def get_current_book(request: Request):
    """Get the current book being read"""
    book = tracker.get_current_book()
    if book:
        return JSONResponse(content=book)
    return JSONResponse(content={"message": "No current book"}, status_code=404)

@app.get("/api/books")
@limiter.limit("30/minute")
async def get_all_books(request: Request):
    """Get all books in the library"""
    books = tracker.get_all_books()
    return JSONResponse(content={"books": books})

@app.post("/api/books/next")
@limiter.limit("5/minute")
async def next_book(request: Request):
    """Move to the next book"""
    success = tracker.next_book()
    if success:
        book = tracker.get_current_book()
        return JSONResponse(content=book)
    return JSONResponse(content={"message": "No more books available"}, status_code=404)

@app.post("/api/books/previous")
@limiter.limit("5/minute")
async def previous_book(request: Request):
    """Move to the previous book"""
    success = tracker.previous_book()
    if success:
        book = tracker.get_current_book()
        return JSONResponse(content=book)
    return JSONResponse(content={"message": "No previous book available"}, status_code=404)

@app.post("/api/books/{book_id}/rating")
@limiter.limit("10/minute")
async def rate_book(book_id: int, rating: BookRating, request: Request):
    """Rate a book"""
    if rating.rating < 1 or rating.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    
    success = tracker.rate_book(book_id, rating.rating)
    if success:
        return JSONResponse(content={"message": "Rating updated successfully"})
    raise HTTPException(status_code=404, detail="Book not found")

@app.post("/api/books/{book_id}/status")
@limiter.limit("10/minute")
async def update_status(book_id: int, status: BookStatus, request: Request):
    """Update book reading status"""
    valid_statuses = ["unread", "reading", "completed"]
    if status.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Status must be one of {valid_statuses}")
    
    success = tracker.update_status(book_id, status.status)
    if success:
        return JSONResponse(content={"message": "Status updated successfully"})
    raise HTTPException(status_code=404, detail="Book not found")

@app.get("/api/stats")
async def get_stats():
    """Get reading statistics"""
    stats = tracker.get_stats()
    return JSONResponse(content=stats)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
