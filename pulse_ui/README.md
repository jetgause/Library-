# PULSE UI - Trading Platform Dashboard

Modern web interface for PULSE tool management system.

## Quick Start

```bash
cd pulse_ui
pip install fastapi uvicorn jinja2
python app.py
```

Open browser to http://localhost:8000

## Features

- Real-time monitoring with WebSocket
- Dark mode theme
- Interactive Chart.js visualizations
- Tool management interface
- Responsive design
- Toast notifications

## Installation

```bash
pip install fastapi uvicorn jinja2
```

## Usage

Development:
```bash
python app.py
```

Production:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

- GET /api/metrics - System metrics
- GET /api/tools - List tools
- POST /api/tools - Create tool
- GET /health - Health check

## Project Structure

```
pulse_ui/
├── app.py
├── templates/
│   ├── base.html
│   ├── dashboard.html
│   ├── tools.html
│   ├── tool_create.html
│   ├── monitoring.html
│   └── settings.html
└── static/
    ├── css/custom.css
    └── js/dashboard.js
```

## License

MIT License

Version: 1.0.0
Date: 2025-12-10
Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-12-10 03:36:52
Current User's Login: jetgause
