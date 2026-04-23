# Igbo Archives Notes Agent

[![AI-Powered](https://img.shields.io/badge/AI-Autonomous%20Agents-blueviolet)](https://google.github.io/google-adk/)
[![Powered by Gemma](https://img.shields.io/badge/Powered%20by-Gemma--4-blue)](https://deepmind.google/models/gemma/gemma-4/)

An enterprise-grade, fully autonomous, **multimodal** AI pipeline designed for **Daily Cultural Archiving**. This system preserves Igbo heritage by intelligently finding un-noted archives, performing targeted web research for deep historical context, and publishing validated notes to the [Igbo Archives](https://igboarchives.com.ng) platform.

## 🏗️ Architecture
The system follows a **Sequential Pipeline** leveraging Google ADK:

- **Orchestrator (Traffic Cop)**: Scans the archives API using an MCP client to find an archive needing notes. It dynamically detects the archive's media type (Image, Audio, or Video) and routes it to the corresponding contextual analyst.
- **Vision & Audio Analysts**: Quarantined multimodal inspectors that examine the media file while anchored to the existing metadata. (Uses Google GenAI SDK to safely upload massive byte payloads).
- **Researcher**: An entity-extracting retrieval agent that scrapes the original archive URL and scours the internet for highly specific geographical and historical context.
- **Writer & Critic Loop**: Synthesizes the gathered research into purely factual, zero-fluff community notes formatted as Editor.js blocks. An internal critic ruthlessly validates the drafts against strict archival standards to prevent hallucinations.
- **Publisher**: The final executioner. Pushes the fully approved Editor.js payload to the remote Igbo Archives database.

## 🛠️ Tech Stack
-   **Framework**: Google ADK
-   **LLM Engine**: Google Gemma 4 (Resilient fallback configuration routing between `gemma-4-31b-it` and `gemma-4-26b-a4b-it`) for text/vision. Google Gemini Flash (`gemini-3.1-flash-lite-preview`) natively handles audio processing.

## 🚀 Installation & Usage

### 1. Setup
```bash
git clone [https://github.com/Nwokike/Notes-Agent.git](https://github.com/Nwokike/Notes-Agent.git)
cd notes-agent
uv sync
```

### 2. Run the Agent
The app auto-detects its environment. It runs a Telegram Polling Bot locally or a Webhook on a cloud service like Render.
```bash
# Start the Telegram Bot Interface
uv run python app.py
```
