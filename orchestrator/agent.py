from google.adk.agents import Agent, SequentialAgent, Context
from google.adk.models import Gemini
from .utils.resilience import ResilientGemini

from .researcher.agent import researcher_agent
from .writer.agent import writer_loop
from .publisher.agent import publisher_agent
from .vision.agent import execute_vision_analysis
from .mcp_client import call_mcp_tool
import os
import requests
import tempfile

__all__ = ["root_agent"]

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
            
        # Fetch the FULL archive metadata
        full_archive_resp = await call_mcp_tool("igbo-archives", "retrieve_archives", {"kwargs": {"slug": target_archive["slug"]}})
        if isinstance(full_archive_resp, dict) and "id" in full_archive_resp:
            target_archive = full_archive_resp
            
        # Download image for vision processing
        image_url = target_archive.get("image") or target_archive.get("thumbnail")
        if image_url:
            try:
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    temp_dir = tempfile.gettempdir()
                    image_path = os.path.join(temp_dir, f"archive_{target_archive['id']}.jpg")
                    with open(image_path, "wb") as f:
                        f.write(response.content)
                    ctx.state["image_path"] = image_path
            except Exception as e:
                print(f"Image download failed: {e}")

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
    model=ResilientGemini(
        model="models/gemma-4-26b-a4b-it",
        fallbacks=["models/gemma-4-31b-it"]
    ),
    description="The root supervisor for the Igbo Archives Autonomous Notes System.",
    sub_agents=[notes_pipeline_agent],
    tools=[fetch_unnoted_archive, execute_vision_analysis],
    instruction="""
ROLE:
Chief Orchestrator of the Igbo Notes Autonomous Creation System.

STRICT WORKFLOW:
1. DATA FETCH: Call `fetch_unnoted_archive` to find an unnoted archive. 
2. BLIND VISION ANALYSIS: Call `execute_vision_analysis`. This performs a purely visual, unbiased check of the image.
3. VALIDATION: Cross-reference the discovered archive metadata with the vision report. If they mismatch or vision fails, state the reason and STOP.
4. PIPELINE TRIGGER: If valid, trigger `execute_notes_pipeline`.

RULES: Keep responses clinical, professional, and brief. No conversational filler.
""".strip()
)

root_agent = orchestrator
