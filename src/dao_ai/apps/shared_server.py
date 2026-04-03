"""
Shared agent execution server for DAO AI.

This module is the entry point for the shared execution Databricks App — a single
persistent app that can dynamically load and execute any agent registered in Unity
Catalog, without requiring a dedicated app or endpoint per agent.

Agents are resolved per-request via dao_* fields in custom_inputs. See shared_handlers.py
for the full request contract.

Also exposes:
  GET  /api/models   — lists UC registered models (for the chat UI model picker)
  POST /api/chat     — SSE streaming chat endpoint (used by the built-in chat UI)
  GET  /             — serves the built-in React chat UI

Usage (via DABs bundle):
    python -m dao_ai.apps.shared_server
"""

import json
import os
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from mlflow.genai.agent_server import AgentServer

# Import shared handlers to register the invoke and stream decorators.
# This MUST happen before creating the AgentServer instance.
import dao_ai.apps.shared_handlers  # noqa: E402, F401
from dao_ai.apps.shared_handlers import _get_agent, _resolve_model_version

# Create the AgentServer instance
agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=False)

# Module-level app variable enables multiple workers
app = agent_server.app

# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

_FRONTEND_DIST = Path(__file__).parent / "chat_ui" / "dist"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    index = _FRONTEND_DIST / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text())
    return HTMLResponse(content="<p>Chat UI not built. Run: cd chat_ui && npm install && npm run build</p>", status_code=503)


if _FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="assets")

# ---------------------------------------------------------------------------
# /api/models — list UC registered models for the model picker
# ---------------------------------------------------------------------------

@app.get("/api/models")
async def list_models():
    try:
        import mlflow
        client = mlflow.MlflowClient()
        models = client.search_registered_models()
        result = [
            {
                "name": m.name,
                "latest_versions": [
                    {"version": v.version, "aliases": v.aliases if hasattr(v, "aliases") else []}
                    for v in (m.latest_versions or [])
                ],
            }
            for m in models
            if any(
                t.key == "dao_ai"
                for v in (m.latest_versions or [])
                for t in (v.tags or [])
            )
        ]
        return JSONResponse({"models": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# /api/chat — SSE streaming chat endpoint for the UI
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(request: Request):
    """
    SSE streaming chat endpoint for the built-in chat UI.

    Request body:
      dao_model  (str, required): UC registered model name
      messages   (list):          conversation history
      thread_id  (str, optional): for conversation continuity

    Streams SSE events:
      {"type": "delta",  "content": "..."}
      {"type": "done",   "response": "full text"}
      {"type": "error",  "error": "..."}
    """
    data = await request.json()
    dao_model = data.get("dao_model")
    messages = data.get("messages", [])
    thread_id = data.get("thread_id")

    async def generate():
        def send(event: dict) -> str:
            return f"data: {json.dumps(event)}\n\n"

        if not dao_model:
            yield send({"type": "error", "error": "dao_model is required"})
            return

        try:
            from mlflow.types.responses import ResponsesAgentRequest

            agent = _get_agent(dao_model, None, None)

            input_items = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
                if msg.get("content")
            ]

            custom_inputs = {}
            if thread_id:
                custom_inputs["configurable"] = {"thread_id": thread_id}

            agent_request = ResponsesAgentRequest(
                messages=input_items,
                custom_inputs=custom_inputs or None,
            )

            full_response = ""
            async for event in agent.apredict_stream(agent_request):
                # Extract text deltas from ResponsesAgentStreamEvent
                delta = None
                if hasattr(event, "delta") and event.delta:
                    delta = event.delta
                elif isinstance(event, dict):
                    delta = event.get("delta") or event.get("content")

                if delta:
                    full_response += delta
                    yield send({"type": "delta", "content": delta})

            yield send({"type": "done", "response": full_response})

        except Exception as e:
            import traceback
            yield send({"type": "error", "error": str(e), "trace": traceback.format_exc()})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def main() -> None:
    """Entry point for running the shared agent execution server."""
    agent_server.run(app_import_string="dao_ai.apps.shared_server:app")


if __name__ == "__main__":
    main()
