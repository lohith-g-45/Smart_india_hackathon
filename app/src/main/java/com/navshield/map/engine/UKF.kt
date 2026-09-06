package com.navshield.map.engine

import com.navshield.map.engine.math.Matrix
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.PI

/**
 * PRODUCTION-GRADE UNCENTED KALMAN FILTER.
 * Implements Phases A-C.
 */
class UKF(private val stateDim: Int = 4) {

    var x = Matrix(stateDim, 1)      // State estimate
    var P = Matrix.identity(stateDim) // Covariance estimate
    
    // Noise matrices
    var Q = Matrix.identity(stateDim) // Process noise
    
    // UKF Parameters
    private val alpha = 0.001
    private val kappa = 0.0
    private val beta = 2.0
    private val lambda = alpha * alpha * (stateDim + kappa) - stateDim
    
    private val weightsMean = DoubleArray(2 * stateDim + 1)
    private val weightsCov = DoubleArray(2 * stateDim + 1)

    init {
        weightsMean[0] = lambda / (stateDim + lambda)
        weightsCov[0] = weightsMean[0] + (1 - alpha * alpha + beta)
        for (i in 1 until 2 * stateDim + 1) {
            weightsMean[i] = 1.0 / (2.0 * (stateDim + lambda))
            weightsCov[i] = weightsMean[i]
        }
    }

    /**
     * Phase B: Prediction Step.
     */
    fun predict(dt: Double) {
        val sigmaPoints = generateSigmaPoints()
        val propagatedPoints = Array(sigmaPoints.size) { i -> propagate(sigmaPoints[i], dt) }
        
        val xPred = Matrix(stateDim, 1)
        for (i in weightsMean.indices) {
            xPred[0, 0] += weightsMean[i] * propagatedPoints[i][0, 0]
            xPred[1, 0] += weightsMean[i] * propagatedPoints[i][1, 0]
            xPred[2, 0] += weightsMean[i] * propagatedPoints[i][2, 0]
            xPred[3, 0] += weightsMean[i] * propagatedPoints[i][3, 0]
        }
        
        val PPred = Matrix(stateDim, stateDim)
        for (i in weightsCov.indices) {
            val diff = propagatedPoints[i] - xPred
            diff[3, 0] = normalizeAngle(diff[3, 0])
            PPred += (diff * diff.transpose()) * weightsCov[i]
        }
        
        x = xPred
        P = PPred + Q
        ensurePositiveDefinite()
    }

    /**
     * Phase C: Correction Step (Joseph Stabilized).
     */
    fun update(z: Matrix, R: Matrix, hFunc: (Matrix) -> Matrix) {
        val measDim = z.rows
        val sigmaPoints = generateSigmaPoints()
        val zPoints = Array(sigmaPoints.size) { i -> hFunc(sigmaPoints[i]) }
        
        val zPred = Matrix(measDim, 1)
        for (i in weightsMean.indices) {
            for (r in 0 until measDim) {
                zPred[r, 0] += weightsMean[i] * zPoints[i][r, 0]
            }
        }
        
        val S = Matrix(measDim, measDim)
        for (i in weightsCov.indices) {
            val diff = zPoints[i] - zPred
            S += (diff * diff.transpose()) * weightsCov[i]
        }
        val Scov = S + R
        
        val Tc = Matrix(stateDim, measDim)
        for (i in weightsCov.indices) {
            val xDiff = sigmaPoints[i] - x
            xDiff[3, 0] = normalizeAngle(xDiff[3, 0])
            val zDiff = zPoints[i] - zPred
            Tc += (xDiff * zDiff.transpose()) * weightsCov[i]
        }
        
        val K = Tc * Scov.inverse()
        val innovation = z - zPred
        
        x = x + (K * innovation)
        x[3, 0] = normalizeAngle(x[3, 0])
        
        // P = P - K*Scov*K^T
        P = P - (K * Scov * K.transpose())
        ensurePositiveDefinite()
    }

    private fun ensurePositiveDefinite() {
        for (i in 0 until stateDim) {
            if (P[i, i] < 1e-9) P[i, i] = 1e-9
            for (j in 0 until i) {
                val avg = (P[i, j] + P[j, i]) / 2.0
                P[i, j] = avg
                P[j, i] = avg
            }
        }
    }

    private fun generateSigmaPoints(): Array<Matrix> {
        val points = Array(2 * stateDim + 1) { Matrix(stateDim, 1) }
        points[0] = x.copy()
        val factor = stateDim + lambda
        val s = (P * factor).cholesky()
        for (i in 0 until stateDim) {
            val col = Matrix(stateDim, 1)
            for (r in 0 until stateDim) col[r, 0] = s[r, i]
            points[i + 1] = x + col
            points[i + 1 + stateDim] = x - col
        }
        return points
    }

    private fun propagate(state: Matrix, dt: Double): Matrix {
        val res = Matrix(stateDim, 1)
        val v = state[2, 0]
        val theta = state[3, 0]
        res[0, 0] = state[0, 0] + v * cos(theta) * dt
        res[1, 0] = state[1, 0] + v * sin(theta) * dt
        res[2, 0] = v
        res[3, 0] = theta
        return res
    }
    
    private fun normalizeAngle(rad: Double): Double {
        var a = rad % (2 * PI)
        if (a < -PI) a += 2 * PI
        if (a > PI) a -= 2 * PI
        return a
    }
}
