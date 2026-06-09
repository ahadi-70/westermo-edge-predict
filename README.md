# Westermo EdgePredict: Industrial Wireless Maintenance

**Edge AI-Driven Predictive Maintenance for Industrial Wireless Networks**

## Project Overview
This project is a Proof-of-Concept (PoC) designed for **Westermo Neratec AG** wireless hardware. It bridges the gap between raw industrial telemetric data and actionable intelligence by deploying an **Edge AI anomaly detection layer**.

By utilizing the **Isolation Forest** algorithm, the system continuously monitors device health—including signal strength (RSSI), packet loss, temperature, and latency—to predict potential failures before they result in costly downtime.

## Why this project matters to Westermo
*   **Predictive Maintenance:** Moves from reactive (fixing broken devices) to proactive (preventing failures) maintenance.
*   **Edge-First Philosophy:** By running the ML inference directly on the edge, the system functions independently of cloud connectivity, ensuring reliability in the harshest industrial environments.
*   **Lightweight & Efficient:** Designed for resource-constrained hardware, the model provides sub-1ms inference times.

## Key Features
- **Real-Time Dashboard:** An industrial-grade UI that visualizes device health and anomalies.
- **Fault Injection:** A simulation panel to test system robustness against real-world scenarios (packet loss, signal degradation, thermal spikes).
- **Offline Capable:** The frontend includes a client-side simulation engine, ensuring the demo works anywhere, regardless of internet connectivity.

## Tech Stack
- **AI/ML:** Scikit-learn (Isolation Forest)
- **Backend:** FastAPI (Async WebSocket streaming)
- **Frontend:** Pure HTML/JS (Chart.js for real-time visualization)

## Setup and Running the Project

To set up and run this project, follow the steps below:

### 1. Clone the Repository

First, clone the GitHub repository to your local machine:

```bash
git clone https://github.com/ahadi-70/westermo-edge-predict.git
cd westermo-edge-predict
```

### 2. Extract Project Files

The core project files are contained within a zip archive. Extract them:

```bash
unzip "files (1).zip"
```

### 3. Install Python Dependencies

Ensure you have Python 3.8+ installed. Then, install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI Backend

Start the FastAPI application using Uvicorn. This will launch the API and WebSocket server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The backend will be accessible at `http://localhost:8000`.

### 5. Serve the Frontend

Open the `index.html` file in your web browser. You can do this by simply navigating to the file path in your browser (e.g., `file:///path/to/westermo-edge-predict/index.html`) or by using a simple HTTP server (e.g., Python's `http.server` module):

```bash
python -m http.server 8000 --directory .
```

If you run the frontend using a separate HTTP server, ensure it's serving from the same directory as `index.html` and that the `API_BASE` and `WS_URL` in `index.html` are correctly configured to point to your FastAPI backend.

### 6. Access the Application

Once both the backend and frontend are running, open your web browser and navigate to the address where you are serving `index.html` (e.g., `http://localhost:8000/index.html` if using Python's `http.server` in the project root, or `file:///path/to/westermo-edge-predict/index.html`).

## Troubleshooting

*   **Backend not starting:** Check the console output for any error messages from Uvicorn. Ensure all dependencies are installed.
*   **Frontend not connecting:** Verify that the FastAPI backend is running and accessible. Check your browser's developer console for WebSocket connection errors. Ensure `API_BASE` and `WS_URL` in `index.html` match your backend's address.
