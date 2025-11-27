from __future__ import annotations

"""FastAPI surface for the simulation engine and config storage."""

import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.models.domain import ConfigPayload, SimulationResult, SimulationSettings
from backend.simulation.engine import simulate_config


class ConfigStore:
    """In-memory store to cache config and last simulation result between requests.

    The API is stateless across processes, but a single process instance benefits
    from holding the latest configuration and simulation outputs so that runs and
    exports do not require clients to re-upload. This simple holder centralizes
    error handling (raising 404s when nothing exists) and keeps FastAPI routes
    lean by delegating persistence logic here.
    """
    def __init__(self) -> None:
        """Initialize the store with empty slots for config and last result.

        This keeps construction trivial while allowing future swaps for durable
        backends (e.g., Redis or disk) without changing the API surface.
        """
        self.config: Optional[ConfigPayload] = None
        self.last_result: Optional[SimulationResult] = None

    def save(self, config: ConfigPayload) -> None:
        """Persist the provided config object for future simulation calls.

        The store does no merging; callers should provide the full payload each
        time so subsequent simulations operate on a complete, consistent config.
        """
        self.config = config

    def get(self) -> ConfigPayload:
        """Return the current config or raise a 404-style HTTPException if absent.

        Surfacing the HTTPException here centralizes the error semantics so route
        handlers remain thin.
        """
        if not self.config:
            raise HTTPException(status_code=404, detail="No configuration loaded")
        return self.config

    def save_result(self, result: SimulationResult) -> None:
        """Persist the most recent simulation result for later export or review.

        The latest result overwrites any previous one because the API treats the
        simulation output as ephemeral and tied to the last config submitted.
        """
        self.last_result = result

    def export(self) -> SimulationResult:
        """Return the last simulation result or raise if no simulations have been run.

        This powers the /export endpoint; keeping the guard here ensures callers
        cannot accidentally serialize `None`.
        """
        if not self.last_result:
            raise HTTPException(status_code=404, detail="No simulation has been run yet")
        return self.last_result


store = ConfigStore()
app = FastAPI(title="Worldline", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Lightweight liveness probe to verify the API is reachable.

    Used by dev servers and potential container orchestrators to quickly confirm
    the process is alive without touching any stateful resources.
    """
    return {"status": "ok"}


@app.post("/config")
def post_config(config: ConfigPayload) -> dict:
    """Persist the provided config payload in memory for later simulation.

    The endpoint performs no transformation: it trusts Pydantic validation to
    enforce schema correctness and simply caches the config so subsequent calls
    to /simulate or /config can reuse it without re-upload from the client.
    """
    store.save(config)
    return {"message": "configuration saved"}


@app.get("/config")
def get_config() -> ConfigPayload:
    """Return the last saved configuration or raise a 404 if none has been saved."""
    return store.get()


@app.post("/simulate")
def run_simulation(settings_override: SimulationSettings | None = None) -> SimulationResult:
    """Run simulations using the saved config (optionally overridden settings) and store the result.

    The function pulls the current config from the store, optionally merges in a
    request-provided settings override, and delegates to the simulation engine.
    Results are cached for export so clients can immediately fetch them without
    recomputation.
    """
    config = store.get()
    result = simulate_config(config, settings_override or config.simulation_settings)
    store.save_result(result)
    return result


@app.get("/export")
def export_result() -> SimulationResult:
    """Return the most recent simulation result or raise a 404 if none exists."""
    return store.export()
