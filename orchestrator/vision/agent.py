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
    """Unbiased blind cultural heritage visual analysis."""
    image_path = ctx.state.get("image_path")
    
    if not image_path or not os.path.exists(image_path):
        return "ERROR: Image not found for analysis."
        
    try:
        base64_image = _encode_and_compress_image(image_path)
        client = genai.Client(http_options=types.HttpOptions(timeout=120_000))
        models_to_try = ["models/gemma-4-31b-it", "models/gemma-4-26b-a4b-it"]
        
        for model_name in models_to_try:
            try:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=[
                        "ROLE: Elite Cultural Heritage Visual Analyst.\n"
                        "GOAL: Meticulously examine the provided image and extract a purely visual, unbiased cultural report.\n"
                        "STRICT RULES: No hallucination. No introductory fluff. Output as a clinical observational report.",
                        types.Part.from_bytes(
                            data=base64.b64decode(base64_image),
                            mime_type="image/jpeg"
                        )
                    ]
                )
                
                result_text = response.text
                ctx.state["vision_report"] = result_text
                return result_text
            except Exception:
                continue
                
        return "ERROR: Vision analysis failed after all retries."
    except Exception as e:
        return f"ERROR: {str(e)}"
