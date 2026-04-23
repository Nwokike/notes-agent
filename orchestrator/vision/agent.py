import os
import base64
import io
import logging
from google import genai
from google.genai import types
import PIL.Image
import asyncio
from google.adk.agents import Context

logger = logging.getLogger(__name__)

def _encode_and_compress_image(image_path: str, max_size=(1024, 1024)) -> str:
    """Resizes the image to fit model limits and converts to base64."""
    with PIL.Image.open(image_path) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.thumbnail(max_size)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

async def execute_vision_analysis(ctx: Context) -> str:
    """Contextual visual analysis grounded in the archive's metadata."""
    
    # FIX 1: Look for the universal media_path first
    image_path = ctx.state.get("media_path", ctx.state.get("image_path"))
    discovered_archive = ctx.state.get("discovered_archive", {})
    
    if not image_path or image_path == "NONE" or not os.path.exists(image_path):
        return "ERROR: Image not found for analysis."
        
    try:
        base64_image = _encode_and_compress_image(image_path)
        client = genai.Client(http_options=types.HttpOptions(timeout=120_000))
        models_to_try = ["models/gemma-4-31b-it", "models/gemma-4-26b-a4b-it"]
        
        # Build a context-aware prompt using the fetched metadata
        prompt = (
            "ROLE: Elite Contextual Heritage Visual Analyst.\n"
            "GOAL: Examine the provided image while cross-referencing it with its known archival metadata.\n"
            f"METADATA CONTEXT: {discovered_archive}\n"
            "STRICT RULES:\n"
            "1. Do NOT guess the context blindly. Use the Metadata Context to anchor your visual observations.\n"
            "2. Extract highly specific visual details that complement the metadata (e.g., specific clothing, architectural styles, distinct objects, or geographical markers visible in the background).\n"
            "3. NO introductory fluff. Do not say 'This image shows'. Output strictly as a clinical, factual observational report."
        )
        
        for model_name in models_to_try:
            try:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=[
                        prompt,
                        types.Part.from_bytes(
                            data=base64.b64decode(base64_image),
                            mime_type="image/jpeg"
                        )
                    ]
                )
                
                result_text = response.text
                
                # FIX 2: Hoist success to the universal media_report key
                ctx.state["media_report"] = result_text
                return result_text
            except Exception:
                continue
                
        return "ERROR: Vision analysis failed after all retries."
    except Exception as e:
        return f"ERROR: {str(e)}"
