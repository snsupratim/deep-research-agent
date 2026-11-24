# Deep Research Agent API

An autonomous multi-source intelligence report generator powered by FastAPI, LangGraph, and Groq LLMs.

## Overview

The Research Agent Server is a fully modularised backend application that executes a complex reasoning pipeline to generate comprehensive, properly cited intelligence reports. It uses **LangGraph** to navigate the research process through multiple specialised agent nodes:

- **Planner Node**: Breaks down the primary query into targeted sub-queries for web and academic searches.
- **Researcher Node**: Executes concurrent queries against the Web and arXiv, collating high-confidence sources.
- **Critic Node**: Aggressively cross-references claims between sources to find contradictions, tensions, or disagreements.
- **Writer Node**: Synthesises all context and flagged conflicts into a structured, highly analytical report.

## Key Features

- **Modular LangGraph Architecture**: Easy to maintain and extend individual agent behaviors.
- **Real-time Status Streaming**: Connect via WebSockets (`/api/v1/ws/{report_id}`) to see live transitions between agent states (planning → researching → critiquing → writing).
- **MongoDB Integration**: Asynchronous database operations using Motor, natively supporting Cloud MongoDB URIs.
- **Robust Export Options**: Export completed reports natively into PDF or DOCX format.
- **Rate Limited & CORS Enabled**: Production-ready API structure out-of-the-box.

## Tech Stack

- **Framework**: FastAPI (Python 3.11)
- **Agent Orchestration**: LangGraph / LangChain
- **LLM Engine**: Google Gemini (gemini-2.0-flash by default)
- **Database**: MongoDB (via Motor)
- **Deployment**: Docker / Docker Compose

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/research-agent-server.git
   cd research-agent-server
   ```

2. **Environment Variables**:
   Copy the example environment file and fill in your details:
   ```bash
   cp .env.example .env
   ```
   **Required variables in `.env`:**
   - `GEMINI_API_KEY`: Your Google Gemini API Key.
   - `MONGODB_URI`: Your MongoDB connection string (e.g., Cloud MongoDB Atlas).

3. **Run with Docker Compose (Recommended)**:
   This builds the FastAPI application and exposes it on port 8000.
   ```bash
   docker-compose up -d --build
   ```

4. **Run Locally (Without Docker)**:
   ```bash
   pip install -r requirements.txt
   uvicorn main:app --reload
   ```

## API Endpoints

### REST API

- `POST /api/v1/reports`: Trigger a new research run (Returns immediately with `report_id`).
  - Body: `{"query": "Your research topic", "max_sources": 15}`
- `GET /api/v1/reports/{report_id}`: Fetch the full details and generated content of a report.
- `GET /api/v1/reports`: List all generated reports (paginated).
- `GET /api/v1/reports/{report_id}/export?format=pdf`: Export a completed report to `pdf` or `docx`.
- `DELETE /api/v1/reports/{report_id}`: Delete a report.

### WebSocket (Real-time Status Updates)

Connect to the WebSocket endpoint to receive real-time JSON payloads as the LangGraph transitions through the agent nodes:

- `ws://localhost:8000/api/v1/reports/{report_id}/stream`

Example incoming payloads:
```json
{"status": "researching"}
{"status": "critiquing"}
{"status": "writing"}
{"status": "done"}
```

## Contributing

Contributions, issues, and feature requests are welcome!
