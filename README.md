# Tech challenge for "MBA Engenharia de Software com IA - Full Cycle"

This is a python project to ingest a PDF into a vector database (postgres + PGVector). You can search using a CLI script.

## Requirements

- Python
- Docker
- Gemini API Key

## Setup and run

To setup and run the project easily you will need to follow these steps:

### 1. Environment setup

- Create and activate a virtual environment: `python3 -m venv venv && source venv/Scripts/activate`
- Create your `.env` file based on `.env.example`: `cp .env.example .env`
- Setup your Gemini credentials on `.env`
- Spin up the postgres + pgvector: `docker compose up -d`

### 2. Ingest the PDF into the

```
python src/ingest.py
```

### 3. Run the chat (CLI)

```
python src/chat.py
```
