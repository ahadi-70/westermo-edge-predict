"""
Westermo EdgePredict — FastAPI Backend
Real-time anomaly detection for industrial wireless networks.
"""

import asyncio
import json
import random
import time
import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EdgePredict")

# ── App Setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Westermo EdgePredict API",
    description="Edge AI anomaly detection for industrial wireless networks",
    version="1.0.1",
)

# Improved CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ML Model (trained at startup) ─────────────────────────────────────────────

class AnomalyModel:
    def __init__(self):
        self.model: Optional[IsolationForest] = None
        self.trained_at: Optional[str] = None
        self.sample_count = 0
        self._train()

    def _train(self):
        try:
            logger.info("Training anomaly detection model...")
            np.random.seed(42)
            n = 2000
            signal = np.random.normal(-50, 5, n)
            loss   = np.random.normal(0.5, 0.2, n).clip(0, None)
            temp   = np.random.normal(40, 3, n)
            latency = np.random.normal(10, 2, n).clip(0, None)
            X = np.column_stack([signal, loss, temp, latency])

            self.model = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
            self.model.fit(X)
            self.trained_at = datetime.utcnow().isoformat() + "Z"
            self.sample_count = n
            logger.info(f"Model trained successfully with {n} samples.")
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise

    def predict(self, signal: float, packet_loss: float, temperature: float, latency: float):
        if self.model is None:
            return False, 0.0
        X = np.array([[signal, packet_loss, temperature, latency]])
        label = self.model.predict(X)[0]          # 1 = normal, -1 = anomaly
        score = float(self.model.score_samples(X)[0])  # more negative = more anomalous
        return label == -1, score

ml = AnomalyModel()

# ── In-memory log ──────────────────────────────────────────────────────────────

MAX_LOG = 500
event_log: List[dict] = []

def append_log(event: dict):
    event_log.append(event)
    if len(event_log) > MAX_LOG:
        event_log.pop(0)

# ── Data Models ───────────────────────────────────────────────────────────────

class ReadingIn(BaseModel):
    device_id: str
    signal_strength: float   # dBm
    packet_loss: float       # %
    temperature: float       # °C
    latency: float           # ms

class ReadingOut(BaseModel):
    device_id: str
    signal_strength: float
    packet_loss: float
    temperature: float
    latency: float
    is_anomaly: bool
    anomaly_score: float
    severity: str            # normal | warning | critical
    timestamp: str

def get_severity(score: float, is_anomaly: bool) -> str:
    if not is_anomaly:
        return "normal"
    return "critical" if score < -0.15 else "warning"

# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "EdgePredict API", "status": "running", "version": "1.0.1"}

@app.get("/model/info")
def model_info():
    return {
        "algorithm": "Isolation Forest",
        "n_estimators": 100,
        "contamination": 0.05,
        "trained_at": ml.trained_at,
        "training_samples": ml.sample_count,
        "features": ["signal_strength", "packet_loss", "temperature", "latency"],
    }

@app.post("/predict", response_model=ReadingOut)
def predict(reading: ReadingIn):
    try:
        is_anom, score = ml.predict(
            reading.signal_strength, reading.packet_loss,
            reading.temperature, reading.latency
        )
        event = ReadingOut(
            device_id=reading.device_id,
            signal_strength=reading.signal_strength,
            packet_loss=reading.packet_loss,
            temperature=reading.temperature,
            latency=reading.latency,
            is_anomaly=is_anom,
            anomaly_score=round(score, 4),
            severity=get_severity(score, is_anom),
            timestamp=datetime.utcnow().isoformat() + "Z",
        ).dict()
        append_log(event)
        return event
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during prediction")

@app.get("/events")
def get_events(limit: int = 100, anomalies_only: bool = False):
    data = event_log[-limit:]
    if anomalies_only:
        data = [e for e in data if e["is_anomaly"]]
    return {"count": len(data), "events": data}

@app.get("/stats")
def get_stats():
    if not event_log:
        return {"total": 0, "anomalies": 0, "anomaly_rate": 0}
    try:
        total = len(event_log)
        anoms = sum(1 for e in event_log if e["is_anomaly"])
        criticals = sum(1 for e in event_log if e["severity"] == "critical")
        return {
            "total": total,
            "anomalies": anoms,
            "criticals": criticals,
            "anomaly_rate": round(anoms / total * 100, 2),
            "avg_signal": round(sum(e["signal_strength"] for e in event_log) / total, 2),
            "avg_packet_loss": round(sum(e["packet_loss"] for e in event_log) / total, 2),
            "avg_temperature": round(sum(e["temperature"] for e in event_log) / total, 2),
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {"error": "Failed to calculate stats"}

@app.delete("/events")
def clear_events():
    event_log.clear()
    return {"message": "Log cleared"}

# ── WebSocket — live simulation ───────────────────────────────────────────────

DEVICES = ["WES-3152P-A", "WES-3152P-B", "WES-3152P-C", "WES-4960G-A"]

class SimState:
    """Controls whether the simulation is in 'fault injection' mode."""
    def __init__(self):
        self.fault_device: Optional[str] = None
        self.fault_type: Optional[str] = None   # "packet_loss" | "temperature" | "signal"
        self.fault_step = 0

sim_state = SimState()

def generate_reading(device_id: str) -> dict:
    """Generate one sensor reading, optionally with injected fault."""
    is_faulty = (device_id == sim_state.fault_device)

    if is_faulty and sim_state.fault_type == "packet_loss":
        sim_state.fault_step += 1
        packet_loss = min(0.5 + sim_state.fault_step * 0.6, 18.0)
        signal = random.gauss(-52, 3)
        temp   = random.gauss(42, 2)
        latency = random.gauss(12, 2)
    elif is_faulty and sim_state.fault_type == "temperature":
        sim_state.fault_step += 1
        temp = min(40 + sim_state.fault_step * 2.5, 80.0)
        signal = random.gauss(-52, 3)
        packet_loss = max(random.gauss(0.5, 0.2), 0)
        latency = random.gauss(12, 2)
    elif is_faulty and sim_state.fault_type == "signal":
        sim_state.fault_step += 1
        signal = max(-50 - sim_state.fault_step * 3, -92.0)
        temp   = random.gauss(42, 2)
        packet_loss = max(random.gauss(0.5, 0.2), 0)
        latency = random.gauss(12, 2)
    else:
        signal      = random.gauss(-50, 5)
        packet_loss = max(random.gauss(0.5, 0.2), 0)
        temp        = random.gauss(40, 3)
        latency     = max(random.gauss(10, 2), 0)

    is_anom, score = ml.predict(signal, packet_loss, temp, latency)
    event = {
        "device_id": device_id,
        "signal_strength": round(signal, 2),
        "packet_loss": round(packet_loss, 3),
        "temperature": round(temp, 2),
        "latency": round(latency, 2),
        "is_anomaly": is_anom,
        "anomaly_score": round(score, 4),
        "severity": get_severity(score, is_anom),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    append_log(event)
    return event


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted.")
    try:
        while True:
            # Check for control messages (non-blocking)
            try:
                # Use a small timeout to allow periodic data emission
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                cmd = json.loads(msg)
                if cmd.get("action") == "inject_fault":
                    sim_state.fault_device = cmd.get("device")
                    sim_state.fault_type   = cmd.get("fault_type", "packet_loss")
                    sim_state.fault_step   = 0
                    logger.info(f"Fault injected: {sim_state.fault_type} on {sim_state.fault_device}")
                elif cmd.get("action") == "clear_fault":
                    sim_state.fault_device = None
                    sim_state.fault_type   = None
                    sim_state.fault_step   = 0
                    logger.info("Fault cleared.")
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                logger.warning("Received invalid JSON on WebSocket.")
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")

            # Emit a reading for each device
            readings = [generate_reading(d) for d in DEVICES]
            await websocket.send_text(json.dumps({"type": "readings", "data": readings}))
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
