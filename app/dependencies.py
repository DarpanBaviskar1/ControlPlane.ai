"""FastAPI dependency providers.

- request_id_dep: injects a fresh UUID v4 per request
- get_policy_loader: returns the application-level PolicyLoader singleton
- get_profile_dep: resolves the UseCaseProfile for the inbound request,
  raising HTTP 422 if the profile is unknown
- get_pii_engine: returns the application-level PIIMaskingEngine singleton
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.models import UseCaseProfile


def request_id_dep() -> str:
    """Generate a UUID v4 request ID."""
    return str(uuid.uuid4())


def get_policy_loader(request: Request):
    """Return the PolicyLoader stored in app.state."""
    return request.app.state.policy_loader


def get_pii_engine(request: Request):
    """Return the PIIMaskingEngine stored in app.state."""
    return request.app.state.pii_engine


async def profile_from_name(
    use_case_profile: str,
    policy_loader=Depends(get_policy_loader),
) -> UseCaseProfile:
    """Resolve use_case_profile name → UseCaseProfile, or raise 422.

    ``use_case_profile`` is extracted from the query-string when used as a
    standalone dependency.  In the /v1/chat handler it is passed explicitly
    from the parsed request body so that the full ChatRequest is still
    validated by Pydantic first.
    """
    try:
        return await policy_loader.get_profile(use_case_profile)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_code": "UNRECOGNISED_PROFILE",
                "detail": f"use_case_profile '{use_case_profile}' does not match any configured profile",
            },
        )


async def resolve_profile(
    body: "ChatRequest",
    policy_loader=Depends(get_policy_loader),
) -> UseCaseProfile:
    """Dependency that resolves UseCaseProfile from a parsed ChatRequest body."""
    from app.models import ChatRequest  # local import avoids circular dep at module level
    try:
        return await policy_loader.get_profile(body.use_case_profile)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_code": "UNRECOGNISED_PROFILE",
                "detail": f"use_case_profile '{body.use_case_profile}' does not match any configured profile",
            },
        )
