package com.navshield.map.engine

import com.navshield.map.engine.math.Matrix
import kotlin.math.*

class UKF(private val stateDim: Int = 4) {
    var x = Matrix(stateDim, 1)
    var P = Matrix.identity(stateDim)
    var Q = Matrix.identity(stateDim) * 0.01
    private val alpha = 0.1; private val kappa = 0.0; private val beta = 2.0; private val lambda = alpha * alpha * (stateDim + kappa) - stateDim
    private val weightsMean = DoubleArray(2 * stateDim + 1)
    private val weightsCov = DoubleArray(2 * stateDim + 1)

    init {
        weightsMean[0] = lambda / (stateDim + lambda); weightsCov[0] = weightsMean[0] + (1 - alpha * alpha + beta)
        for (i in 1 until 2 * stateDim + 1) { weightsMean[i] = 1.0 / (2.0 * (stateDim + lambda)); weightsCov[i] = weightsMean[i] }
    }

    fun predict(dt: Double) {
        val sigmas = generateSigmaPoints(); val prop = Array(sigmas.size) { i -> propagate(sigmas[i], dt) }
        val xP = Matrix(stateDim, 1)
        for (i in weightsMean.indices) { xP[0, 0] += weightsMean[i] * prop[i][0, 0]; xP[1, 0] += weightsMean[i] * prop[i][1, 0]; xP[2, 0] += weightsMean[i] * prop[i][2, 0]; xP[3, 0] += weightsMean[i] * prop[i][3, 0] }
        var PP = Matrix(stateDim, stateDim)
        for (i in weightsCov.indices) { 
            val diff = prop[i] - xP; diff[3, 0] = nA(diff[3, 0])
            for (r in 0 until stateDim) for (c in 0 until stateDim) PP[r, c] += (diff[r, 0] * diff[c, 0]) * weightsCov[i]
        }
        x = xP; P = PP + Q
    }
    
    fun update(z: Matrix, R: Matrix, h: (Matrix) -> Matrix) {
        val mD = z.rows; val sigmas = generateSigmaPoints(); val zP = Array(sigmas.size) { i -> h(sigmas[i]) }
        val zM = Matrix(mD, 1); for (i in weightsMean.indices) for (r in 0 until mD) zM[r, 0] += weightsMean[i] * zP[i][r, 0]
        var S = Matrix(mD, mD); for (i in weightsCov.indices) { 
            val d = zP[i] - zM
            for (r in 0 until mD) for (c in 0 until mD) S[r, c] += (d[r, 0] * d[c, 0]) * weightsCov[i]
        }
        val Sc = S + R; val Tc = Matrix(stateDim, mD); for (i in weightsCov.indices) { 
            val xD = sigmas[i] - x; xD[3, 0] = nA(xD[3, 0]); val zD = zP[i] - zM
            for (r in 0 until stateDim) for (c in 0 until mD) Tc[r, c] += (xD[r, 0] * zD[c, 0]) * weightsCov[i]
        }
        val K = Tc * Sc.inverse(); val inn = z - zM; x = x + (K * inn); x[3, 0] = nA(x[3, 0]); P = P - (K * Sc * K.transpose())
    }

    private fun generateSigmaPoints(): Array<Matrix> {
        val pts = Array(2 * stateDim + 1) { Matrix(stateDim, 1) }; pts[0] = x.copy(); val s = (P * (stateDim + lambda)).cholesky()
        for (i in 0 until stateDim) { val col = Matrix(stateDim, 1); for (r in 0 until stateDim) col[r, 0] = s[r, i]; pts[i + 1] = x + col; pts[i + 1 + stateDim] = x - col }
        return pts
    }

    private fun propagate(s: Matrix, dt: Double): Matrix { val res = Matrix(stateDim, 1); val v = s[2,0]; val t = s[3,0]; res[0,0]=s[0,0]+v*cos(t)*dt; res[1,0]=s[1,0]+v*sin(t)*dt; res[2,0]=v; res[3,0]=t; return res }
    private fun nA(r: Double): Double { var a = r % (2 * PI); if (a < -PI) a += 2 * PI; if (a > PI) a -= 2 * PI; return a }
}
