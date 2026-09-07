# SIH_member3
This module provides calibrated IMU measurements, phone-to-vehicle alignment, orientation, velocity, displacement, dead-reckoned position, motion state and IMU confidence.  GNSS heading is received from Member 2 through setGnssHeading().
# NAV-SHIELD — Member 3

## IMU Engine + Calibration + Dead Reckoning + Motion Intelligence

This repository contains the current **Member 3** work for the NAV-SHIELD Smart India Hackathon project.

Member 3 is responsible for the **inertial navigation core**: reading smartphone IMU sensors, filtering and calibrating the data, estimating orientation, preparing phone-to-vehicle alignment, estimating motion, and building the foundation for GNSS-denied dead reckoning.

> **Current status:** The implementation is currently developed as an Android/Kotlin prototype. GNSS heading integration with Member 2 is prepared through an input interface, but Member 2's live Android GNSS implementation has not yet been connected.

---

## 1. What Has Been Implemented So Far

The current `MainActivity.kt` contains the following work:

### Phase A — Accelerometer Acquisition

- Reads Android accelerometer data using `SensorManager`.
- Uses Android's `SENSOR_DELAY_GAME` sensor rate.
- Stores recent accelerometer readings in a circular/ring buffer.
- Applies a low-pass IIR filter.
- Uses the assignment's low-pass coefficient:

```text
α = 0.8
```

The UI displays raw and filtered accelerometer values as well as the current buffer usage.

---

### Phase B — Gyroscope Acquisition

- Reads 3-axis gyroscope measurements.
- Gyroscope values are handled in `rad/s`.
- Applies a high-pass filter.
- Estimates static gyro bias during the initial calibration period.
- Displays:
  - raw gyro
  - high-pass filtered gyro
  - gyro bias
  - corrected gyro

---

### Phase C — Initial Sensor Calibration

The application performs an initial stationary calibration period of approximately:

```text
5 seconds
```

During calibration it collects accelerometer and gyroscope samples and calculates their mean values.

The calibration information is retained as the initial sensor-bias estimate.

A distinction is maintained between the accelerometer's stationary measurement and the gravity component so that gravity is not incorrectly treated as dynamic acceleration during dead reckoning.

---

## 2. Phase D — Phone-to-Vehicle Alignment

The current implementation uses Android's `TYPE_ROTATION_VECTOR` as an orientation reference.

It obtains:

```text
Yaw
Pitch
Roll
```

from the Android rotation-vector sensor.

A phone-to-vehicle rotation matrix is then constructed.

### GNSS Heading Interface

Member 2 is expected to provide the real GNSS heading.

The current interface is:

```kotlin
fun setGnssHeading(heading: Float)
```

Example:

```kotlin
setGnssHeading(90f)
```

The heading is interpreted in degrees.

When GNSS heading is not available, the application displays:

```text
WAITING FOR MEMBER 2
```

No fake GNSS heading is hard-coded.

When GNSS heading is available, the implementation calculates the yaw offset between the phone orientation and vehicle heading and updates the phone-to-vehicle rotation matrix.

---

## 3. Phase E — Orientation Estimation

The current implementation includes a custom quaternion-based orientation update.

The gyroscope is integrated over time to maintain orientation.

A complementary correction approach is included using:

```text
98% gyro integration
2% accelerometer gravity correction
```

The purpose is to reduce long-term roll/pitch drift while retaining the responsiveness of gyro integration.

---

## 4. Phase F — Velocity and Displacement

The current implementation contains the foundation for inertial motion estimation:

1. Transform acceleration into the vehicle frame.
2. Estimate/remove gravity.
3. Integrate acceleration to obtain velocity.
4. Integrate velocity to obtain displacement.
5. Apply Zero-Velocity Update (ZUPT) when the motion classifier identifies a stationary state.

The implementation also includes a road-grade input interface so that grade information can be incorporated later.

---

## 5. Phase G — Dead Reckoning Position

The implementation contains GNSS-denied position propagation.

Starting from an initial/last-known latitude and longitude, the module can project position using:

```text
heading + displacement
```

with a spherical/haversine forward-position calculation.

The module also exposes interfaces for setting the initial position and external state.

---

## 6. Phase H — Motion and Vibration Classification

A threshold-based motion classifier is implemented.

It can produce states including:

```text
STATIONARY
NORMAL
ACCELERATING
BRAKING
TURNING
POTHOLE
PHONE_DISTURBANCE
```

The module also calculates a vibration level.

The current Android UI displays the detected motion state and vibration information.

---

## 7. Phase I — IMU Confidence

An IMU confidence value in the range:

```text
0.0 – 1.0
```

is implemented.

The confidence calculation considers factors related to:

- sensor noise
- motion state
- bias stability
- accumulated drift / dead-reckoning duration
- sensor sampling rate

The implementation also exposes diagnostic information such as sensor rate and long-duration dead-reckoning status.

> **Note:** Phase I is implemented but still requires final tuning/validation. The current code has not yet been considered fully validated against the assignment's physical accuracy targets.

---

## 8. Output / Integration Interfaces

The Member 3 implementation contains an output structure for the inertial-navigation results.

The intended output includes values such as:

```text
imu_latitude
imu_longitude
imu_heading
imu_velocity
imu_displacement
imu_bias_estimate
imu_confidence
motion_state
vibration_level
zupt_applied
```

There are also interfaces for external initialization and updates, including:

```kotlin
setGnssHeading(...)
setInitialPosition(...)
setInitialState(...)
setRoadGrade(...)
resetFromExternalState(...)
```

These interfaces are intended to make the IMU module independently testable and ready for integration with the other team members.

---

## 9. Current Android UI

The application currently displays the internal processing results for debugging and validation.

The UI includes information for:

- Accelerometer
- Gyroscope
- Calibration
- Phone orientation
- GNSS heading status
- Phone-to-vehicle alignment
- Motion state
- Vibration level
- IMU confidence
- Sampling-rate diagnostics
- Dead-reckoning diagnostics

The screen is scrollable so that the later Phase H and Phase I information can also be inspected.

---

## 10. Logging

The implementation includes CSV logging support for offline analysis.

The logging functionality is intended to help inspect sensor readings and calculated navigation/motion values during testing.

The code also provides functionality to obtain the CSV log file path and clear the current log.

---

## 11. Current Integration With Member 2

Member 2 has shared a GNSS engine prototype that produces information including:

```text
gnss_latitude
gnss_longitude
gnss_altitude
gnss_speed
gnss_heading
gnss_available
gnss_confidence
gnss_anomaly
gnss_recovery
gnss_mode
```

For the current Member 3 implementation, the most important input from Member 2 is:

```text
gnss_heading
```

The integration point is already prepared:

```kotlin
setGnssHeading(heading: Float)
```

### Current situation

Member 2's shared GNSS implementation is currently a Python prototype.

Therefore, the two modules are **not yet directly connected as live Android components**.

No changes to the Member 3 IMU pipeline are required just because Member 2's Python prototype has been shared.

The Android/Kotlin GNSS provider will be connected later through the defined interface.

---

## 12. Current Development Status

### Completed / Implemented

- [x] Android accelerometer acquisition
- [x] Accelerometer low-pass filtering
- [x] Accelerometer ring buffer
- [x] Gyroscope acquisition
- [x] Gyroscope high-pass filtering
- [x] Initial 5-second calibration
- [x] Rotation-vector orientation reference
- [x] GNSS heading input interface
- [x] Phone-to-vehicle alignment calculation
- [x] Phone-to-vehicle rotation matrix
- [x] Quaternion orientation integration
- [x] Complementary orientation correction
- [x] Vehicle-frame acceleration processing
- [x] Velocity estimation
- [x] Displacement estimation
- [x] ZUPT logic
- [x] Haversine-based dead reckoning
- [x] Motion classification
- [x] Vibration estimation
- [x] IMU confidence calculation
- [x] Scrollable debug UI
- [x] CSV logging support
- [x] External-state/input interfaces

### Not Yet Integrated / Fully Validated

- [ ] Live Android connection to Member 2's GNSS engine
- [ ] Full Member 2 → Member 3 live data flow
- [ ] Member 3 → Member 4 UKF integration
- [ ] Full road-grade integration from Member 1
- [ ] Final Phase-I confidence tuning
- [ ] Full physical validation on recorded/live driving data
- [ ] Complete multi-member system validation

---

## 13. Testing Planned

The module should eventually be tested using the following scenarios:

### Stationary Test

Keep the phone stationary for approximately 30 seconds.

Check:

```text
motion_state = STATIONARY
ZUPT = applied
velocity → 0
```

### Straight Drive Test

Use a known straight route and compare the dead-reckoned position against a reference.

### Turn Test

Perform a known 90° turn and compare the estimated heading change.

### Pothole Test

Introduce a short acceleration spike and check that:

```text
motion_state = POTHOLE
```

and vibration level increases.

### Phone Disturbance Test

Rotate/move the phone suddenly and check:

```text
motion_state = PHONE_DISTURBANCE
```

and reduced IMU confidence.

### Alignment Test

Place the phone at an offset angle relative to the vehicle axis and verify that GNSS-heading-based alignment corrects the yaw offset.

---

## 14. Project Responsibility

Member 3 owns:

```text
IMU Sensors
     ↓
Filtering
     ↓
Calibration
     ↓
Orientation
     ↓
Phone → Vehicle Alignment
     ↓
Dead Reckoning
     ↓
Motion Classification
     ↓
IMU Confidence
```

Member 3 does **not** own:

```text
GNSS Engine       → Member 2
UKF / Sensor Fusion → Member 4
Road Slope        → Member 1
LSTM / AI module  → Member 5
```

The interfaces between these modules should remain clearly separated.

---

## 15. Important Design Principle

The Member 3 module should remain independently testable.

GNSS heading is therefore treated as an **external input** rather than being hard-coded into the IMU engine.

This allows the current IMU work to be developed and tested independently while keeping the code ready for integration with Member 2's Android GNSS implementation.

---

## Package

Current Android package:

```text
com.example.nav_shield_1
```

Keep this package consistent with the Android manifest and activity configuration.

---

## Current Main File

The main implementation is currently contained in:

```text
app/src/main/java/com/example/nav_shield_1/MainActivity.kt
```

