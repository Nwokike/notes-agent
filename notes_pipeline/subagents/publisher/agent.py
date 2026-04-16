import os
import json
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext
from ...mcp_client import call_mcp_tool

__all__ = ["publisher_agent"]

publisher_model = LiteLlm(
    model="groq/openai/gpt-oss-20b",
    fallbacks=["groq/qwen/qwen3-32b", "groq/moonshotai/kimi-k2-instruct", "gemini/gemma-4-26b-it", "gemini/gemma-4-31b-it"]
)



# Let's adjust the tool here directly.
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
    model=publisher_model,
    description="Agent: The final record publisher that submits notes.",
    tools=[execute_mcp_publish],
    instruction="""
ROLE:
You are the Final Executioner for the Igbo Archives.

GOAL:
Take the drafted notes payload created by the Writer and push them to the live database using your tool.

AVAILABLE DATA:
- The `draft_notes` string from the Writer containing the JSON of the drafted notes.
PAYLOAD: {draft_notes}

STRICT RULES:
1. Parse the success JSON object returned by the Writer. Extract the `drafts` list.
2. Pass that list as a JSON string to `execute_mcp_publish`.
3. Output the success or failure results to the user.
"""
)