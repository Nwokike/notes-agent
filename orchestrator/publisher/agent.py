import json
from google.adk.agents import Agent
from google.adk.models import Gemini
from ..utils.resilience import ResilientGemini
from ..mcp_client import call_mcp_tool

__all__ = ["publisher_agent"]

async def execute_mcp_publish(drafts_json: str) -> str:
    """Accepts a JSON string representing the list of drafts and publishes them to Igbo Archives via MCP."""
    try:
        drafts = json.loads(drafts_json)
        results = []
        for draft in drafts:
            response = await call_mcp_tool("igbo-archives", "create_archive_notes", {"body": draft})
            if "error" in response:
                results.append(f"Failed to publish: {response['error']}")
            else:
                results.append(f"Published successfully! ID: {response.get('id')}")
        return json.dumps(results)
    except Exception as e:
        return str(e)


publisher_agent = Agent(
    name="PublisherAgent",
    model=ResilientGemini(
        model="models/gemma-4-31b-it",
        fallbacks=["models/gemma-4-26b-a4b-it"]
    ),
    description="Agent: The final record publisher that submits notes.",
    tools=[execute_mcp_publish],
    instruction="""
ROLE:
You are the Final Executioner for the Igbo Archives.

GOAL:
Take the drafted notes payload created by the Writer and push them to the live database, BUT ONLY if they passed validation.

AVAILABLE DATA:
- Critic Status: {critic_status}
- Draft Payload: {draft_notes}

STRICT RULES:
1. SAFETY CHECK: Check the Critic Status. If the status does not explicitly say "APPROVED", it means the draft failed validation after maximum retries. DO NOT call the tool. Instead, output: "❌ Pipeline aborted: The drafted notes failed to pass the Critic's archival validation standards."
2. PUBLISH: If the Critic Status is "APPROVED", parse the success JSON object in the Draft Payload. Extract the `drafts` list.
3. Pass that list as a JSON string to `execute_mcp_publish`.
4. Output the success or failure results to the user.
""".strip()
)
