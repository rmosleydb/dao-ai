"""
Shared execution handlers for the DAO AI shared agent execution layer.

This module provides invoke and stream handlers for a single persistent Databricks App
that can dynamically load and execute any agent registered in Unity Catalog.

Unlike handlers.py (which loads one agent config at startup), this module resolves
the target agent per-request from custom_inputs, pulls the config from the UC-registered
model artifact, builds the agent, and caches it by model version.

Required custom_inputs:
    dao_model (str): UC registered model name in catalog.schema.model_name format

Optional custom_inputs:
    dao_alias (str): Model alias to resolve (e.g. "Champion", "staging"). Defaults to "Champion".
    dao_version (str): Explicit model version number. Takes precedence over dao_alias if provided.
"""

import os
from typing import AsyncGenerator

import mlflow
from mlflow.genai.agent_server import get_request_headers, invoke, stream
from mlflow.models import ModelConfig
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from dao_ai.config import AppConfig
from dao_ai.logging import configure_logging, suppress_autolog_context_warnings
from dao_ai.models import LanggraphResponsesAgent

# Configure MLflow
mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
mlflow.langchain.autolog()
suppress_autolog_context_warnings()

configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

# Cache of built agents keyed by (model_name, version)
_agent_cache: dict[tuple[str, str], LanggraphResponsesAgent] = {}


def _inject_headers_into_request(request: ResponsesAgentRequest) -> None:
    """Inject request headers into custom_inputs for OBO auth propagation."""
    headers: dict[str, str] = get_request_headers()
    if headers:
        if request.custom_inputs is None:
            request.custom_inputs = {}
        if "configurable" not in request.custom_inputs:
            request.custom_inputs["configurable"] = {}
        request.custom_inputs["configurable"]["headers"] = headers


def _resolve_model_version(model_name: str, alias: str | None, version: str | None) -> str:
    """Resolve a UC model version number from alias or explicit version."""
    client = mlflow.MlflowClient()
    if version:
        return version
    resolved_alias = alias or "Champion"
    mv = client.get_model_version_by_alias(model_name, resolved_alias)
    return mv.version


def _build_agent(model_name: str, version: str) -> LanggraphResponsesAgent:
    """Download model artifact from UC, parse config, and build the agent."""
    model_uri = f"models:/{model_name}/{version}"
    local_path = mlflow.artifacts.download_artifacts(model_uri)
    config = AppConfig(**ModelConfig(development_config=f"{local_path}/model_config.yaml").to_dict())
    config.initialize()
    return config.as_responses_agent()  # type: ignore[return-value]


def _get_agent(model_name: str, alias: str | None, version: str | None) -> LanggraphResponsesAgent:
    """Return a cached agent, building it on first use for a given model version."""
    resolved_version = _resolve_model_version(model_name, alias, version)
    cache_key = (model_name, resolved_version)
    if cache_key not in _agent_cache:
        _agent_cache[cache_key] = _build_agent(model_name, resolved_version)
    return _agent_cache[cache_key]


def _extract_dao_inputs(request: ResponsesAgentRequest) -> tuple[str, str | None, str | None]:
    """
    Extract and validate dao_* routing fields from custom_inputs.

    Returns:
        (dao_model, dao_alias, dao_version)

    Raises:
        ValueError: If dao_model is not present in custom_inputs.
    """
    custom_inputs: dict = request.custom_inputs or {}
    dao_model: str | None = custom_inputs.get("dao_model")
    if not dao_model:
        raise ValueError(
            "dao_model is required in custom_inputs. "
            "Provide the UC registered model name as 'catalog.schema.model_name'."
        )
    dao_alias: str | None = custom_inputs.get("dao_alias")
    dao_version: str | None = custom_inputs.get("dao_version")
    return dao_model, dao_alias, dao_version


def _strip_dao_inputs(request: ResponsesAgentRequest) -> ResponsesAgentRequest:
    """Remove dao_* keys from custom_inputs before passing request to the agent."""
    if request.custom_inputs:
        cleaned = {k: v for k, v in request.custom_inputs.items() if not k.startswith("dao_")}
        request.custom_inputs = cleaned or None
    return request


@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Handle non-streaming requests by routing to the appropriate UC-registered agent."""
    _inject_headers_into_request(request)
    dao_model, dao_alias, dao_version = _extract_dao_inputs(request)
    agent = _get_agent(dao_model, dao_alias, dao_version)
    return await agent.apredict(_strip_dao_inputs(request))


@stream()
async def streaming(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    """Handle streaming requests by routing to the appropriate UC-registered agent."""
    _inject_headers_into_request(request)
    dao_model, dao_alias, dao_version = _extract_dao_inputs(request)
    agent = _get_agent(dao_model, dao_alias, dao_version)
    async for event in agent.apredict_stream(_strip_dao_inputs(request)):
        yield event
