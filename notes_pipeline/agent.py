import os
import json
import asyncio
from google.adk.agents import Agent, SequentialAgent, Context
from google.adk.models.lite_llm import LiteLlm

from .subagents.researcher.agent import researcher_agent
from .subagents.writer.agent import writer_loop
from .subagents.publisher.agent import publisher_agent
from .mcp_client import call_mcp_tool

__all__ = ["root_agent"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Supervisor using kimi or gpt-oss-20b
orchestrator_model = LiteLlm(
    model="groq/openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    fallbacks=["groq/qwen/qwen3-32b", "gemini/gemma-4-26b-it", "gemini/gemma-4-31b-it"]
)

async def fetch_unnoted_archive(ctx: Context) -> dict:
    """Fetches all archives and notes, finds an archive without notes, and sets it in global state."""
    try:
        arr_resp = await call_mcp_tool("igbo-archives", "list_archives")
        if "error" in arr_resp:
            return {"error": f"Failed to fetch archives: {arr_resp['error']}"}
            
        all_archives = arr_resp.get("results", []) if isinstance(arr_resp, dict) and "results" in arr_resp else arr_resp
        
        notes_resp = await call_mcp_tool("igbo-archives", "list_archive_notes")
        if "error" in notes_resp:
            return {"error": f"Failed to fetch notes: {notes_resp['error']}"}
            
        all_notes = notes_resp.get("results", []) if isinstance(notes_resp, dict) and "results" in notes_resp else notes_resp
        
        archives_with_notes = set([str(note.get("archive_id")) for note in all_notes if note.get("archive_id")])
        archives_with_notes.update(set([note.get("archive_id") for note in all_notes if note.get("archive_id")]))
        
        target_archive = None
        for archive in all_archives:
            if archive.get("id") not in archives_with_notes and str(archive.get("id")) not in archives_with_notes:
                target_archive = archive
                break
                
        if not target_archive:
            return {"status": "DONE", "message": "All archives currently have notes!"}
            
        # Sets the discovered archive directly into state so the researcher can access it
        ctx.state["discovered_archive"] = target_archive
        ctx.state["target_archive_id"] = target_archive["id"]
        
        return {"status": "FOUND", "archive_id": target_archive["id"]}
        
    except Exception as e:
        return {"error": str(e)}

# The nested pipeline
notes_pipeline_agent = SequentialAgent(
    name="execute_notes_pipeline",
    description="The Master Notes Pipeline. Executes research, drafting, validation, and publication.",
    sub_agents=[researcher_agent, writer_loop, publisher_agent]
)

# The root orchestrator
orchestrator = Agent(
    name="orchestrator",
    model=orchestrator_model,
    description="The root supervisor for the Igbo Archives Autonomous Notes System.",
    sub_agents=[notes_pipeline_agent],
    tools=[fetch_unnoted_archive],
    instruction="""
ROLE:
You are the Chief Orchestrator of the Igbo Notes Autonomous Creation System.

GOAL:
Find an archive that has no notes, ensure it's loaded, and trigger the creation pipeline.

STRICT WORKFLOW:
1. DATA FETCH: Call `fetch_unnoted_archive` to find an archive that needs historical notes.
2. VALIDATION: Check the response from `fetch_unnoted_archive`. If it returns "DONE" or an error, output "TERMINATE: No archives need notes or an error occurred." and STOP.
3. PIPELINE TRIGGER: If an archive is successfully found ("FOUND"), call the `execute_notes_pipeline` agent to begin the research, writing, and publication process.

RULES:
Keep your direct responses clinical, professional, and brief.
""".strip()
)

root_agent = orchestrator