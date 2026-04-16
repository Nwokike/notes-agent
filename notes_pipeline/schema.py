from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

def get_initial_state() -> dict:
    """
    The absolute source of truth for the ADK session memory.
    Ensures all variables are initialized before any agent runs.
    """
    return {
        "target_archive_id": None,
        "archive_metadata": {},
        "active_agent": "",
        "completed_agents": [],
        "research_context": "",
        "draft_notes_payload": [], # List of dictionary representations of ArchiveNote
    }

class EditorJsBlockData(BaseModel):
    text: str = Field(description="The actual text content. Can contain HTML tags like <b>, <i>, or <a> for links.")

class EditorJsBlock(BaseModel):
    id: str = Field(description="Unique string ID for the block (e.g., 'historical_fact_1')")
    type: str = Field(default="paragraph", description="Always use 'paragraph'")
    data: EditorJsBlockData

class EditorJsContent(BaseModel):
    blocks: List[EditorJsBlock] = Field(description="List of EditorJS blocks representing the note content.")
    time: Optional[int] = Field(default=1773870000000)
    version: Optional[str] = Field(default="2.31.0")

class ArchiveNoteCreate(BaseModel):
    """
    Output Schema aligned with Igbo Archives POST /api/v1/archive_notes/
    Since we are generating 2-4 notes, the Writer will output a list of these.
    """
    archive_id: int = Field(description="The strict ID of the archive.")
    content_json: EditorJsContent = Field(description="Block-based content using Editor.js.")