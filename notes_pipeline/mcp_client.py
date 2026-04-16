import os
import httpx
import json
import asyncio
from dotenv import load_dotenv
from typing import Dict, Any, Optional

load_dotenv()

MCP_URL = "https://igboarchives.com.ng/api/mcp/"
API_TOKEN = os.getenv("IGBO_ARCHIVES_TOKEN")

async def call_mcp_tool(server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Robust JSON-RPC client for the Igbo Archives MCP server.
    Includes built-in asynchronous backoff to prevent API rate limiting.
    """
    if not API_TOKEN:
        return {"error": "IGBO_ARCHIVES_TOKEN not found in environment."}

    await asyncio.sleep(1.5)

    headers = {
        "Authorization": f"Token {API_TOKEN}",
        "Content-Type": "application/json"
    }

    # Standard JSON-RPC payload for MCP via HTTP
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments or {}
        }
    }

    try:
        # Extended timeout to 45 seconds to account for heavier payloads
        async with httpx.AsyncClient(timeout=45.0) as client:
            
            # Standard JSON-RPC call
            response = await client.post(MCP_URL, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            if "error" in result:
                return {"error": result["error"]}
            
            # MCP response format: result -> content -> [text/json]
            content = result.get("result", {}).get("content", [])
            if content and "text" in content[0]:
                try:
                    return json.loads(content[0]["text"])
                except json.JSONDecodeError:
                    return {"raw_text": content[0]["text"]}
            
            return result.get("result", {})
            
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP Error {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": f"Network/MCP Error: {str(e)}"}