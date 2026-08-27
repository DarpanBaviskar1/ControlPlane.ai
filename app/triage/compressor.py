"""Compressor for COMPRESS_AND_EDIT triage state.

Sends summarisation prompt to SLM tier via Portkey and validates that no
new named entities appear in the compressed output using spaCy NER.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

async def compress_and_edit(
    response: str,
    max_tokens: int,
) -> str:
    """Compress the response and verify no new named entities are introduced."""
    # Mock compression implementation
    # In a real implementation:
    # 1. Call SLM via Portkey with a summarisation prompt.
    # 2. Extract NER entities from original response.
    # 3. Extract NER entities from compressed response.
    # 4. If any entity in compressed is not in original, reject (fallback to original or hard block).
    
    logger.info(f"Compressing response to {max_tokens} tokens.")
    
    return response[:max_tokens]  # extremely naïve mock
