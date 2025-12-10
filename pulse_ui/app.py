import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import json
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pulse UI")

# Serve static files (CSS, JS, images, etc.)
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Data storage
books_data = []
current_book_index = 0

def load_books():
    """Load books from JSON file"""
    global books_data
    json_path = Path(__file__).parent / "books.json"
    try:
        with open(json_path, 'r') as f:
            books_data = json.load(f)
            logger.info(f"Loaded {len(books_data)} books from JSON")
    except FileNotFoundError:
        logger.error(f"books.json not found at {json_path}")
        books_data = []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing books.json: {e}")
        books_data = []

def save_books():
    """Save books to JSON file"""
    json_path = Path(__file__).parent / "books.json"
    try:
        with open(json_path, 'w') as f:
            json.dump(books_data, f, indent=2)
        logger.info(f"Saved {len(books_data)} books to JSON")
    except Exception as e:
        logger.error(f"Error saving books.json: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load books on startup"""
    load_books()

@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    html_path = Path(__file__).parent / "templates" / "index.html"
    return FileResponse(html_path)

# CORS middleware configuration with secure whitelist
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/api/books")
async def get_books():
    """Get all books"""
    return {"books": books_data}

@app.get("/api/books/current")
async def get_current_book():
    """Get the current book being displayed"""
    global current_book_index
    if not books_data:
        raise HTTPException(status_code=404, detail="No books available")
    if current_book_index >= len(books_data):
        current_book_index = 0
    return books_data[current_book_index]

@app.post("/api/books/next")
async def next_book():
    """Move to the next book"""
    global current_book_index
    if not books_data:
        raise HTTPException(status_code=404, detail="No books available")
    current_book_index = (current_book_index + 1) % len(books_data)
    return books_data[current_book_index]

@app.post("/api/books/previous")
async def previous_book():
    """Move to the previous book"""
    global current_book_index
    if not books_data:
        raise HTTPException(status_code=404, detail="No books available")
    current_book_index = (current_book_index - 1) % len(books_data)
    return books_data[current_book_index]

@app.post("/api/books/{book_id}/rating")
async def update_rating(book_id: str, request: Request):
    """Update book rating"""
    body = await request.json()
    rating = body.get("rating")
    
    if rating is None or not (0 <= rating <= 5):
        raise HTTPException(status_code=400, detail="Rating must be between 0 and 5")
    
    for book in books_data:
        if book.get("id") == book_id:
            book["rating"] = rating
            save_books()
            return {"status": "success", "book": book}
    
    raise HTTPException(status_code=404, detail="Book not found")

@app.post("/api/books/{book_id}/status")
async def update_status(book_id: str, request: Request):
    """Update book reading status"""
    body = await request.json()
    status = body.get("status")
    
    valid_statuses = ["not_started", "reading", "completed"]
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Status must be one of {valid_statuses}")
    
    for book in books_data:
        if book.get("id") == book_id:
            book["status"] = status
            save_books()
            return {"status": "success", "book": book}
    
    raise HTTPException(status_code=404, detail="Book not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
