package com.navshield.map.contract

/**
 * Output of Member 5 (AI Drift Guardian LSTM).
 */
data class Member5Result(
    val predictedPositionErrorM: Float,
    val predictedVelocityErrorMps: Float,
    val timestamp: Long,
    val isAbnormal: Boolean,
    val isDriftDetected: Boolean,
    val state: String, // NORMAL, WARNING, ABNORMAL
    val isValid: Boolean = true
)
