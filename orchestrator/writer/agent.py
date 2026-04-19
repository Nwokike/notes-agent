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
1. ZERO FLUFF: Never write introductory sentences, conclusions, or generic cultural overviews. Go straight to the specific extra context to the archive.
2. FLEXIBLE NOTE COUNT: Write anywhere from 1 to 4 notes depending entirely on the depth of the Research. If the research contains multiple distinct extra context for the archive, separate them into individual notes. If it only contains enough for one solid context, write exactly one note. Never merge clearly unrelated facts, and never artificially stretch a single fact into multiple notes just to meet a quota.
3. ORGANIC CITATIONS: If the Research agent provides a source URL, weave it naturally and flexibly into the narrative (e.g., citing it inline, parenthetically, or organically as part of a sentence like 'According to...'). Vary your citation style so it sounds human. Always use an HTML anchor tag with the actual title.
4. NO FORCED CITATIONS: If no specific URL is provided for a fact, do NOT try to force a citation and do NOT mention the lack of a link. Just state the fact clearly.
5. NO DUPLICATION: Do not reiterate what is already in the archive Metadata. Provide ONLY new, supplemental context.
6. FORMATTING: NEVER use literal newline characters (\\n). Use HTML `<br><br>` for line breaks. 
7. EXCEPTION HANDLING: If the Research agent explicitly states no new context was found, do not hallucinate. Use the `draft_notes` tool to submit EXACTLY one block with the text: "No verifiable additional historical context could be retrieved for this archive."
8. TOOL CALL: Call `draft_notes` with the archive ID and your formulated note(s).
""".strip()
)

critic = Agent(
    name="CriticAgent",
    model=ResilientGemini(
        model="models/gemini-3.1-flash-lite-preview",
        fallbacks=["models/gemma-4-26b-a4b-it", "models/gemma-4-31b-it"]
    ),
    description="Agent: A ruthless gatekeeper that validates drafts against strict archival standards.",
    output_key="critic_status",
    instruction="""
ROLE: Elite Archival Validator.

GOAL: Review drafts for maximum factual density and absolute zero fluff.

STRICT REJECTION CRITERIA:
1. REJECT if any note contains generalities, introductory filler, or generic encyclopedic definitions.
2. REJECT if unrelated, distinct context are crammed into a single note (they should be split into different notes).
3. REJECT if the writer artificially generated multiple notes when the context clearly only supported one single point or should be one note.
4. REJECT if citations feel robotic, repetitive in structure across notes, or are awkwardly forced when no actual URL was provided in the source data.
5. REJECT if the draft simply reiterates the existing archive metadata without adding new external context.
6. REJECT if literal `\\n` characters are used instead of `<br><br>`.
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
