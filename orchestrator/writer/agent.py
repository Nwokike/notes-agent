import json
import uuid
from google.genai import types
from google.adk.agents import Agent, LoopAgent, BaseAgent
from google.adk.models import Gemini
from ..utils.resilience import ResilientGemini
from google.adk.events import Event, EventActions
from typing import List
from ..schema import ArchiveNoteCreate, EditorJsContent, EditorJsBlock, EditorJsBlockData

__all__ = ["writer_loop"]

def draft_notes(archive_id: int, note_texts: List[str]) -> str:
    """Takes a list of finalized note string contents, packages them into Editor.js json format, and saves them to session draft."""
    try:
        drafts = []
        for text in note_texts:
            note = ArchiveNoteCreate(
                archive_id=archive_id,
                content_json=EditorJsContent(
                    blocks=[
                        EditorJsBlock(
                            id=f"block_{uuid.uuid4().hex[:8]}",
                            type="paragraph",
                            data=EditorJsBlockData(text=text)
                        )
                    ]
                )
            )
            drafts.append(note.model_dump())
        
        return json.dumps({"status": "SUCCESS", "drafts": drafts})
    except Exception as e:
        return json.dumps({"status": "FAILURE", "error": str(e)})

writer = Agent(
    name="WriterAgent",
    model=ResilientGemini(
        model="models/gemma-4-31b-it",
        fallbacks=["models/gemma-4-26b-a4b-it"]
    ),
    description="Agent: Synthesizes research and vision into concise community notes.",
    tools=[draft_notes],
    output_key="draft_notes",
    instruction="""
ROLE: Lead Archival Writer.

GOAL: Synthesize provided data into high-quality supplemental notes.

AVAILABLE DATA:
- Metadata: {discovered_archive}
- Blind Vision Report: {vision_report}
- Research Record: {research_context}

STRICT WRITING RULES:
1. SUPPLEMENTAL ONLY: Provide only new historical or cultural context. Do not reiterate metadata fields.
2. NO INTRODUCTORY FLUFF: Banned phrases include any general statements about culture, heritage, or spirituality. Go straight to specific facts.
3. NO AI-ISMS: Maintain a clinical, academic tone.
4. CITATION: Append the source URL using an HTML anchor tag with target="_blank" after a double line break.
5. TOOL CALL: Call `draft_notes` with the archive ID and note content.
""".strip()
)

critic = Agent(
    name="CriticAgent",
    model=ResilientGemini(
        model="models/gemma-4-26b-a4b-it",
        fallbacks=["models/gemma-4-31b-it"]
    ),
    description="Agent: Validates notes against archival standards.",
    output_key="critic_status",
    instruction="""
ROLE: Elite Archival Validator.

GOAL: Review drafts for factual density and zero fluff.

STRICT REJECTION CRITERIA:
1. REJECT if the note contains generalities or introductory filler.
2. REJECT if the note reiterates metadata or visual descriptions already in the original record.
3. REJECT if facts are found that do not exist in the Research Record or Vision Report.
4. REJECT if the citation formatting is incorrect.

OUTPUT: Reply with APPROVED if flawless, otherwise list rejection reasons.
""".strip()
)

class CriticEscalationChecker(BaseAgent):
    """A deterministic agent that checks the critic's status and terminates the loop."""
    name: str = "escalation_checker"
    
    async def _run_async_impl(self, context):
        status = context.session.state.get("critic_status", "")
        
        if "APPROVED" in status.upper():
            yield Event(
                author=self.name, 
                actions=EventActions(escalate=True)
            )
        else:
            yield Event(
                author=self.name,
                content=types.Content(
                    role="system",
                    parts=[types.Part.from_text(text="Draft not approved. Continuing refinement loop.")]
                )
            )

escalation_checker = CriticEscalationChecker()

writer_loop = LoopAgent(
    name="WriterLoop",
    max_iterations=3,
    sub_agents=[writer, critic, escalation_checker],
    description="Loop Agent: Generates and evaluates community notes."
)