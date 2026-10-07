"""Generate answer text with a Gemini model, streamed piece by piece.

Streaming matters for perceived speed: the first words of an answer arrive in
about a second on gemini-3.1-flash-lite, while the full answer takes longer.
"""

from collections.abc import Iterator
from typing import Protocol

from google import genai
from google.genai import types


class Generator(Protocol):
    """What answering needs from a text generation model."""

    model_name: str

    def stream(self, system_instruction: str, prompt: str) -> Iterator[str]:
        """Yield the response text in pieces, in order."""
        ...


class GeminiGenerator:
    """`Generator` backed by the Gemini API.

    Args:
        client: A configured `google.genai.Client` (see
            `rag_me.embeddings.create_gemini_client`). Injected so tests can fake it.
        model_name: Generation model ID, e.g. `gemini-3.1-flash-lite`.
    """

    def __init__(self, client: genai.Client, model_name: str) -> None:
        self._client = client
        self.model_name = model_name

    def stream(self, system_instruction: str, prompt: str) -> Iterator[str]:
        # Generation settings are left at the model's defaults: Google advises
        # against lowering temperature on Gemini 3 models, and a token cap would
        # also count "thinking" tokens on models that think, truncating answers.
        response_stream = self._client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                # No tools are passed, so the SDK's automatic function calling
                # would do nothing except log a warning on every request.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        for response_piece in response_stream:
            # Pieces that carry only metadata (usage, finish reason) have no text.
            if response_piece.text:
                yield response_piece.text
