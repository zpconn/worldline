from __future__ import annotations

import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.models.domain import ConfigPayload, SimulationResult, SimulationSettings
from backend.simulation.engine import simulate_config


class ConfigStore:
    def __init__(self) -> None:
        self.config: Optional[ConfigPayload] = None
        self.last_result: Optional[SimulationResult] = None

    def save(self, config: ConfigPayload) -> None:
        self.config = config

    def get(self) -> ConfigPayload:
        if not self.config:
            raise HTTPException(status_code=404, detail="No configuration loaded")
        return self.config

    def save_result(self, result: SimulationResult) -> None:
        self.last_result = result

    def export(self) -> SimulationResult:
        if not self.last_result:
            raise HTTPException(status_code=404, detail="No simulation has been run yet")
        return self.last_result


store = ConfigStore()
app = FastAPI(title="Career DAG Simulator", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/config")
def post_config(config: ConfigPayload) -> dict:
    store.save(config)
    return {"message": "configuration saved"}


@app.get("/config")
def get_config() -> ConfigPayload:
    return store.get()


@app.post("/simulate")
def run_simulation(settings_override: SimulationSettings | None = None) -> SimulationResult:
    config = store.get()
    result = simulate_config(config, settings_override or config.simulation_settings)
    store.save_result(result)
    return result


@app.get("/export")
def export_result() -> SimulationResult:
    return store.export()
