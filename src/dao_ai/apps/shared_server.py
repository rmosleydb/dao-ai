"""
Shared agent execution server for DAO AI.

This module is the entry point for the shared execution Databricks App — a single
persistent app that can dynamically load and execute any agent registered in Unity
Catalog, without requiring a dedicated app or endpoint per agent.

Agents are resolved per-request via dao_* fields in custom_inputs. See shared_handlers.py
for the full request contract.

Usage (via DABs bundle):
    python -m dao_ai.apps.shared_server
"""

from mlflow.genai.agent_server import AgentServer

# Import shared handlers to register the invoke and stream decorators.
# This MUST happen before creating the AgentServer instance.
import dao_ai.apps.shared_handlers  # noqa: E402, F401

# Create the AgentServer instance
agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=True)

# Module-level app variable enables multiple workers
app = agent_server.app


def main() -> None:
    """Entry point for running the shared agent execution server."""
    agent_server.run(app_import_string="dao_ai.apps.shared_server:app")


if __name__ == "__main__":
    main()
