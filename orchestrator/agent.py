import os
import requests
import tempfile
from google.adk.agents import Agent, SequentialAgent, Context
from google.adk.models import Gemini
from .utils.resilience import ResilientGemini

from .researcher.agent import researcher_agent
from .writer.agent import writer_loop
from .publisher.agent import publisher_agent
from .vision.agent import execute_vision_analysis
from .audio.agent import execute_audio_analysis  # NEW: We will create this file next
from .mcp_client import call_mcp_tool

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
            
        # 1. MULTIMODAL DETECTION & DOWNLOAD
        # Try to get audio, video, then fallback to image/thumbnail
        media_url = target_archive.get("audio") or target_archive.get("video") or target_archive.get("image") or target_archive.get("thumbnail")
        media_type = target_archive.get("archive_type", "image").lower()
        
        ctx.state["media_type"] = media_type
        ctx.state["media_path"] = "NONE"
        
        if media_url:
            try:
                # Increased timeout to handle heavier audio/video files
                response = requests.get(media_url, timeout=30)
                if response.status_code == 200:
                    temp_dir = tempfile.gettempdir()
                    # Extract basic extension from URL or fallback
                    ext = media_url.split('.')[-1][:4] if '.' in media_url else 'bin'
                    ext = ''.join(e for e in ext if e.isalnum())
                    
                    media_path = os.path.join(temp_dir, f"archive_{target_archive['id']}.{ext}")
                    with open(media_path, "wb") as f:
                        f.write(response.content)
                    
                    # Set the new universal paths
                    ctx.state["media_path"] = media_path
                    ctx.state["image_path"] = media_path # Kept for legacy fallback
            except Exception as e:
                print(f"Media download failed: {e}")

        # Sets the discovered archive directly into state so the researcher can access it
        ctx.state["discovered_archive"] = target_archive
        ctx.state["target_archive_id"] = target_archive["id"]
        
        return {
            "status": "FOUND", 
            "archive_id": target_archive["id"],
            "media_type": media_type,
            "media_path": ctx.state["media_path"]
        }
        
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
    tools=[fetch_unnoted_archive, execute_vision_analysis, execute_audio_analysis],
    instruction="""
ROLE:
Chief Orchestrator of the Igbo Notes Autonomous Creation System.

STRICT WORKFLOW:
1. DATA FETCH: Call `fetch_unnoted_archive` to find an unnoted archive. 
2. MEDIA ROUTING & ANALYSIS: Review the `media_type` returned by `fetch_unnoted_archive`.
   - If `media_type` is "image": Call the `execute_vision_analysis` tool blind.
   - If `media_type` is "audio" or "video": Call the `execute_audio_analysis` tool blind.
   - If `media_type` is "document": Skip media analysis and move to VALIDATION.
3. VALIDATION: Cross-reference the discovered archive metadata with the media report. If they completely mismatch or media analysis fails, state the reason and STOP.
4. PIPELINE TRIGGER: If valid, trigger `execute_notes_pipeline`.

RULES: Keep responses clinical, professional, and brief. No conversational filler.
""".strip()
)

root_agent = orchestrator
