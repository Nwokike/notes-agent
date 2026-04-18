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
    description="Agent: Synthesizes historical research into 2 to 4 concise community notes.",
    tools=[draft_notes],
    output_key="draft_notes",
    instruction="""
ROLE:
You are the Lead Writer for the Igbo Archives.

GOAL:
Read the completely untouched archive obtained by the discoverer and the raw text output from the researcher.
RAW ARCHIVE METADATA: {discovered_archive}
RESEARCH RECORD: {research_context}

Write 1, or 2 or 3 distinct, engaging, and historically accurate short community notes based ONLY on the extra context found by the researcher.

STRICT RULES:
1. ONLY EXTRA CONTEXT: Your notes MUST exclusively provide new, supplementary historical or cultural background drawn from the RESEARCH RECORD. DO NOT repeat or reiterate information that is already present in the RAW ARCHIVE METADATA (such as photographer name, dates, visual descriptions of what people are wearing, or locations already stated).
2. NO AI-LIKE LANGUAGE. Be factual, straight to the point, and write like an academic or community elder. NO "In conclusion", "As seen in the image", "Fascinatingly", etc.
3. The URL must be cited exactly attached to the end of the note using HTML exactly like this: <br><br><a href="URL_HERE" target="_blank">Title of Source</a>
4. Call the `draft_notes` tool with the `archive_id` (from metadata) and the list of HTML-formatted note strings you wrote.
5. After calling the tool and it succeeds, just say "Drafting complete."
"""
)

critic = Agent(
    name="CriticAgent",
    model=ResilientGemini(
        model="models/gemma-4-26b-a4b-it",
        fallbacks=["models/gemma-4-31b-it"]
    ),
    description="Agent: Evaluates the draft notes and enforces formatting rules.",
    output_key="critic_status",
    instruction="""
ROLE:
You are an Elite Archival Quality Assurance Validator.

GOAL:
Review the notes drafted by the Writer against strict formatting rules.

AVAILABLE DATA:
- The JSON draft of the notes from the Writer.
- The original source metadata and research context.

STRICT RULES (REJECT THE DRAFT IF ANY OF THESE FAIL):
1. REJECT if the draft contains AI-isms ('In conclusion', 'Fascinatingly', 'delve', etc.).
2. REJECT if the draft hallucinates completely made-up facts NOT found in the raw metadata OR the research context.
3. REJECT if the draft reiterates or summarizes information already present in the raw metadata (e.g., photographer, date, visual description, location). Notes must contain ONLY extra, supplementary context from the research.
4. REJECT if the Writer failed to append the URL using the exact HTML format.

OUTPUT MANDATE:
- If the draft is flawless, you MUST reply with exactly one word: APPROVED.
- If it fails, list the exact REJECTION reasons clearly so the Writer can fix them in the next iteration.
"""
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