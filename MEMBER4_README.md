# NAV-SHIELD Member 4: Trust Brain (State Estimation)

This repository contains the standalone implementation of **Member 4** for project **NAV-SHIELD**.

## 1. Overview
Member 4 serves as the **Trust Brain**—a recursive state estimator that fuses multi-sensor data (GNSS, IMU) with offline map-matching results to provide a high-confidence navigation state even in GNSS-denied environments (e.g., tunnels, Western Ghats).

## 2. Core Architecture
The system is built on a modular, high-performance navigation pipeline:
- **Unscented Kalman Filter (UKF)**: Handles non-linear motion models and sensor fusion with Joseph-stabilized covariance updates.
- **Multi-Hypothesis Engine (MHE)**: Manages road fork ambiguity by tracking multiple concurrent navigation hypotheses.
- **GNSS Mode FSM**: A 5-state machine (`GNSS_DOMINANT`, `HYBRID`, `GNSS_DEGRADED`, `GNSS_DENIED`, `RECOVERY`) that adapts the fusion strategy based on signal quality.
- **Self-Healing Fusion**: Implements a 10-cycle linear recovery ramp to prevent position jumps when GNSS returns.
- **Non-Holonomic Constraints (NHC)**: Enforces vehicle motion physics (no sideways drift).

## 3. Directory Structure
- `app/src/main/java/com/navshield/map/engine/`: Core UKF and Fusion logic.
- `app/src/main/java/com/navshield/map/contract/`: Data Transfer Objects (DTOs) for Member 1 and Member 2.
- `app/src/main/java/com/navshield/map/engine/math/`: custom Matrix and Geographic projection libraries.
- `python_prototype/`: Python-based simulator for Phase J verification scenarios.
- `docs/`: Historical audit and phase implementation reports.

## 4. Technical Specifications
- **State Vector**: `[x, y, velocity, heading_rad]`
- **Coordinate System**: Local metric projection (via Equirectangular conversion).
- **Validation**: 100% Pass rate on **Phase J** scenarios (Tunnel 200m, GNSS Jump, Road Fork, Pothole, Long Stop).

## 5. Standalone Output Contract
The module exposes the `NavShieldTrustState` containing:
- `final_latitude` / `final_longitude`
- `navigation_mode`
- `sensor_weights` (GNSS, IMU, Map, AI)
- `active_hypothesis`
- `estimated_position_error`

## 6. How to Test
### Android (Kotlin)
Run the following command in the project root:
```bash
./gradlew test
```

### Python Simulator
Requires `numpy`:
```bash
python python_prototype/simulator.py
```

---
**Status**: Member 4 work complete. Ready for integration with Upstream (M1, M2) and Downstream (M3, M5, M6).
