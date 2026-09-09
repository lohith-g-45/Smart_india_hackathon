package com.navshield.map.contract

import com.navshield.map.engine.MemberStatus

/**
 * Output of Member 3 (INS / Dead Reckoning).
 */
data class Member3Result(
    val timestampMs: Long,
    val imuAvailable: Boolean,
    val imuLatitude: Double,
    val imuLongitude: Double,
    val imuHeadingDeg: Float,
    val imuVelocityMps: Float,
    val imuDisplacementM: Float,
    val imuConfidence: Float,
    val motionState: String, // STATIONARY, NORMAL, ACCELERATING, BRAKING, TURNING, POTHOLE, PHONE_DISTURBANCE
    val vibrationLevel: Float,
    val zuptApplied: Boolean,
    val accelBiasX: Float,
    val accelBiasY: Float,
    val accelBiasZ: Float,
    val gyroBiasX: Float,
    val gyroBiasY: Float,
    val gyroBiasZ: Float
)
