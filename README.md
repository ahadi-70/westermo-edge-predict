# Westermo EdgePredict: Industrial Wireless Maintenance

**Edge AI-Driven Predictive Maintenance for Industrial Wireless Networks**

## Project Overview
This project is a Proof-of-Concept (PoC) designed for **Westermo Neratec AG** wireless hardware. It bridges the gap between raw industrial telemetric data and actionable intelligence by deploying an **Edge AI anomaly detection layer**.

By utilizing the **Isolation Forest** algorithm, the system continuously monitors device health—including signal strength (RSSI), packet loss, temperature, and latency—to predict potential failures before they result in costly downtime.

## Why this project matters to Westermo
* **Predictive Maintenance:** Moves from reactive (fixing broken devices) to proactive (preventing failures) maintenance.
* **Edge-First Philosophy:** By running the ML inference directly on the edge, the system functions independently of cloud connectivity, ensuring reliability in the harshest industrial environments.
* **Lightweight & Efficient:** Designed for resource-constrained hardware, the model provides sub-1ms inference times.

## Key Features
- **Real-Time Dashboard:** An industrial-grade UI that visualizes device health and anomalies.
- **Fault Injection:** A simulation panel to test system robustness against real-world scenarios (packet loss, signal degradation, thermal spikes).
- **Offline Capable:** The frontend includes a client-side simulation engine, ensuring the demo works anywhere, regardless of internet connectivity.

## Tech Stack
- **AI/ML:** Scikit-learn (Isolation Forest)
- **Backend:** FastAPI (Async WebSocket streaming)
- **Frontend:** Pure HTML/JS (Chart.js for real-time visualization)
