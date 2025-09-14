"""LLM API utilities for HypotheSAEs."""

import os
import time
from google import genai

_CLIENT_GEMINI = None  # Module-level cache for the Google client

# GPT-Generated!
model_abbrev_to_id = {
    # Flash family (fast, cheaper, shorter context)
    "flash-2.5": "gemini-2.5-flash",

    # Pro family (more capable, longer context)
    "pro-2.5": "gemini-1.5-pro",

    # Experimental / other available variants
    "flash-1.5": "gemini-1.5-flash",
}

DEFAULT_MODEL = "gemini-2.5-flash"

def get_client():
    """Get the Gemini client, initializing it if necessary and caching it."""
    global _CLIENT_GEMINI
    if _CLIENT_GEMINI is not None:
        return _CLIENT_GEMINI

    api_key = os.environ.get('GEMINI_API_KEY')
    if api_key is None or '...' in api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable before using functions which require the Gemini API.")

    _CLIENT_GEMINI = genai.Client(api_key=api_key)
    return _CLIENT_GEMINI

def get_completion(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 18.0,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    **kwargs
) -> str:
    """
    Get completion from Gemini API with retry logic and timeout.
    
    Args:
        prompt: The prompt to send
        model: Model to use
        max_retries: Maximum number of retries on rate limit
        backoff_factor: Factor to multiply backoff time by after each retry
        timeout: Timeout for the request
        **kwargs: Additional arguments to pass to the OpenAI API; max_tokens, temperature, etc.
    Returns:
        Generated completion text
    
    Raises:
        Exception: If all retries fail
    """
    client = get_client()
    model_id = model_abbrev_to_id.get(model, model)
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    **kwargs
                )
            )
             
            text = response.text
            # Workaround for partial responses due to output token limit
            if text is None:
                text = "".join(c.text for c in response.candidates[0].content.parts if c.text)
                return text
            
            return text
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise e

            # pretty neat
            wait_time = timeout * (backoff_factor ** attempt)
            if attempt > 0:
                print(f"API error: {e}; retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)