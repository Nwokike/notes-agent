import os
import mimetypes
import logging
from google import genai
from google.genai import types
import asyncio
from google.adk.agents import Context

logger = logging.getLogger(__name__)

async def execute_audio_analysis(ctx: Context) -> str:
    """Contextual auditory analysis grounded in the archive's metadata."""
    # Look for the universal media_path
    audio_path = ctx.state.get("media_path")
    discovered_archive = ctx.state.get("discovered_archive", {})
    
    if not audio_path or audio_path == "NONE" or not os.path.exists(audio_path):
        return "ERROR: Audio file not found for analysis."
        
    try:
        # Read the raw audio bytes
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            
        # Detect MIME type dynamically
        mime_type, _ = mimetypes.guess_type(audio_path)
        if not mime_type:
            mime_type = "audio/mpeg" # Safe fallback
            
        client = genai.Client(http_options=types.HttpOptions(timeout=120_000))
        
        # Flash Lite first, standard Flash as fallback
        models_to_try = ["models/gemini-3.1-flash-lite-preview", "models/gemini-3-flash-preview"]
        
        # Build a context-aware prompt using the fetched metadata
        prompt = (
            "ROLE: Elite Contextual Heritage Audio Analyst.\n"
            "GOAL: Meticulously listen to the provided audio while cross-referencing it with its known archival metadata.\n"
            f"METADATA CONTEXT: {discovered_archive}\n"
            "STRICT RULES:\n"
            "1. TRANSCRIBE & DESCRIBE: Document any spoken words, languages/dialects (if identifiable), musical instruments, vocal tones, rhythmic patterns, and ambient sounds.\n"
            "2. CONTEXTUAL ANCHOR: Do NOT guess the context blindly. Use the Metadata Context to anchor your auditory observations.\n"
            "3. TONE: Output your analysis as a highly detailed, clinical, and objective observational report."
        )
        
        for model_name in models_to_try:
            try:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=[
                        prompt,
                        # Native GenAI SDK payload for audio
                        types.Part.from_bytes(
                            data=audio_bytes,
                            mime_type=mime_type
                        )
                    ]
                )
                
                result_text = response.text
                
                # HOIST SUCCESS TO UNIVERSAL KEY
                ctx.state["media_report"] = result_text
                return result_text
            except Exception as model_err:
                logger.warning(f"Audio analysis failed with model {model_name}: {model_err}")
                continue
                
        return "ERROR: Audio analysis failed after all retries."
    except Exception as e:
        return f"ERROR: {str(e)}"
