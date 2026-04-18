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
        fallbacks=["models/gemini-3.1-flash-lite-preview", "models/gemma-4-26b-a4b-it"]
    ),
    description="Agent: Synthesizes research into concise, purely factual community notes without fluff.",
    tools=[draft_notes],
    output_key="draft_notes",
    instruction="""
ROLE: Lead Archival Writer.

GOAL: Synthesize provided data into high-quality, purely factual supplemental notes.

AVAILABLE DATA:
- Metadata: {discovered_archive}
- Contextual Vision Report: {vision_report}
- Research Record: {research_context}

STRICT WRITING RULES:
1. ZERO FLUFF: Never write introductory sentences, conclusions, or generic cultural overviews (e.g., "The Igbo people are known for...", "This image shows..."). Go straight to the specific historical facts.
2. NO DUPLICATION: Do not reiterate what is already in the original Metadata caption or description. Provide ONLY new, supplemental context.
3. NO AI-ISMS: Maintain a clinical, academic tone. Avoid dramatic or emotional adjectives.
4. EDITOR.JS FORMATTING & CITATIONS: NEVER use literal newline characters (\\n). Use HTML `<br><br>` for line breaks. You MUST cite your sources if a URL is provided. PREFERRED: Cite inline but you may also list source at the end if it can't come naturally inline. Always use HTML anchor tag with the actual title. ONLY omit the link if the Research Record explicitly lacks a matching URL.
5. MULTIPLE NOTES: If you have enough info for more than one note, feel free to write multiple notes. Just don't repeat stuff.
6. EXCEPTION HANDLING: If the Research Record states "No specific supplemental context found", DO NOT hallucinate a note. Instead, use the tool `draft_notes` to submit EXACTLY one block with the text: "No verifiable additional historical context could be retrieved for this archive."
7. TOOL CALL: Call `draft_notes` with the archive ID and the formulated note content.
""".strip()
)

critic = Agent(
    name="CriticAgent",
    model=ResilientGemini(
        model="models/gemma-4-26b-a4b-it",
        fallbacks=["models/gemma-4-31b-it"]
    ),
    description="Agent: A ruthless gatekeeper that validates drafts against strict archival standards.",
    output_key="critic_status",
    instruction="""
ROLE: Elite Archival Validator.

GOAL: Review drafts for maximum factual density and absolute zero fluff.

STRICT REJECTION CRITERIA:
1. REJECT if any note contains generalities, introductory filler, or generic encyclopedic definitions.
2. REJECT if any note simply reiterates the existing metadata description or visual details without adding new external context.
3. REJECT if the draft makes claims that are not explicitly backed by the Research Record or Vision Report.
4. REJECT if literal `\\n` characters are used instead of `<br><br>`.
*EXCEPTION*: If the drafted note is exactly "No verifiable additional historical context could be retrieved for this archive.", you MUST approve it immediately.

OUTPUT: Reply with APPROVED if the text is flawless or matches the exception string. Otherwise, list the specific rejection reasons.
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
