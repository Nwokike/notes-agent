from typing import AsyncGenerator, List
from google.adk.models import Gemini, BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import logging

logger = logging.getLogger(__name__)

class ResilientGemini(BaseLlm):
    """A wrapper for Gemini models that adds fallbacks and robust retries."""
    
    fallbacks: List[str] = []
    retry_attempts: int = 5

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        # Combine primary model with fallbacks
        models_to_try = [self.model] + self.fallbacks
        
        last_error = None
        for model_name in models_to_try:
            try:
                if model_name != self.model:
                    logger.info(f"🔄 Falling back to model: {model_name}")
                
                # Create a fresh Gemini instance for this attempt
                # We configure robust retries at the driver level for 503/429
                gemini = Gemini(
                    model=model_name,
                    retry_options=types.HttpRetryOptions(
                        attempts=self.retry_attempts,
                        initialDelay=2.0,
                        maxDelay=60.0,
                        expBase=2.0,
                        httpStatusCodes=[503, 429]
                    )
                )
                
                # Update the request to point to the current model name
                # This ensures the underlying driver uses the correct endpoint
                llm_request.model = model_name
                
                # Forward the call to the native Gemini driver
                async for response in gemini.generate_content_async(llm_request, stream):
                    yield response
                return # Exit on successful completion
                
            except Exception as e:
                logger.error(f"❌ ResilientGemini: Model {model_name} failed. Error: {e}")
                last_error = e
                # Continue to the next fallback model if available
                continue
        
        # If all models failed, raise the last encountered error
        if last_error:
            raise last_error
