# Igbo Archives Notes Agent

[![AI-Powered](https://img.shields.io/badge/AI-Autonomous%20Agents-blueviolet)](https://google.github.io/google-adk/)
[![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq%20%7C%20Llama--4-f37626)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, fully autonomous AI pipeline designed for **Daily Cultural Archiving**. This system preserves Igbo heritage by intelligently finding un-noted archives, performing live web research for historical context, and publishing validated notes to the [Igbo Archives](https://igboarchives.com.ng) platform via an MCP client.

## 🏗️ Architecture: The Agent Hive
The system follows a **Sequential Pipeline** leveraging Google ADK routing:

- **Discoverer**: Scans the archives API cross-referencing available notes and finds an archive needing historically rich notes.
- **Context Researcher (RAG)**: A DuckDuckGo-powered retrieval agent that scours the internet for missing geographical and historical context.
- **Synthesis Writer**: Generates comprehensive, factually accurate critical community notes formatted strictly in Editor.js payload structures.
- **Publisher**: The final executioner. Pushes the approved payload to the remote MCP server.

## 🛠️ Tech Stack
-   **Framework**: Google ADK (2026)
-   **LLM Engine**: Groq (GPT-oss for Reasoning/Publishing, Moonshot Kimi for Synthesis) via internal fallback mapping.

## 🚀 Installation & Usage

### 1. Setup
```bash
# Clone & Sync
git clone https://github.com/Nwokike/Notes-Agent.git
cd notes-agent
uv sync

```

### 2. Run the Production Suite
The app auto-detects its environment. It runs a Polling Bot locally, or a Webhook + Cloud Tasks worker in production.
```bash
# Start the Telegram Bot + Dynamic Status Streaming
uv run python app.py
```

### 3. Debug with ADK Web UI
Visualize the agent trace, session states, and artifact generation locally:
```bash
# Start the ADK Dev Server
uv run adk web
```

## 🧠 State Management
This pipeline utilizes ADK's Shared Session State to pass variables (`raw_metadata`, `vision_report`, `research_context`, `critic_status`) seamlessly between agents without prompt stuffing, ensuring maximum token efficiency and clean separation of concerns.
