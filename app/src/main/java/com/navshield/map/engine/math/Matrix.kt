package com.navshield.map.engine.math

import kotlin.math.sqrt
import kotlin.math.abs

class Matrix(val rows: Int, val cols: Int, val data: DoubleArray = DoubleArray(rows * cols)) {
    operator fun get(r: Int, c: Int): Double = data[r * cols + c]
    operator fun set(r: Int, c: Int, v: Double) { data[r * cols + c] = v }
    fun copy(): Matrix = Matrix(rows, cols, data.copyOf())
    operator fun plus(other: Matrix): Matrix { val res = Matrix(rows, cols); for (i in data.indices) res.data[i] = data[i] + other.data[i]; return res }
    operator fun minus(other: Matrix): Matrix { val res = Matrix(rows, cols); for (i in data.indices) res.data[i] = data[i] - other.data[i]; return res }
    operator fun times(other: Matrix): Matrix {
        val res = Matrix(rows, other.cols)
        for (r in 0 until rows) for (c in 0 until other.cols) { var sum = 0.0; for (k in 0 until cols) sum += this[r, k] * other[k, c]; res[r, c] = sum }
        return res
    }
    operator fun times(scalar: Double): Matrix { val res = Matrix(rows, cols); for (i in data.indices) res.data[i] = data[i] * scalar; return res }
    fun transpose(): Matrix { val res = Matrix(cols, rows); for (r in 0 until rows) for (c in 0 until cols) res[c, r] = this[r, c]; return res }
    fun inverse(): Matrix {
        val n = rows; val a = Array(n) { r -> DoubleArray(2 * n) { c -> if (c < n) this[r, c] else if (c - n == r) 1.0 else 0.0 } }
        for (i in 0 until n) {
            var pivot = a[i][i]; if (abs(pivot) < 1e-18) pivot = 1e-18
            for (j in 0 until 2 * n) a[i][j] /= pivot
            for (k in 0 until n) if (k != i) { val factor = a[k][i]; for (j in 0 until 2 * n) a[k][j] -= factor * a[i][j] }
        }
        val res = Matrix(n, n); for (r in 0 until n) for (c in 0 until n) res[r, c] = a[r][c + n]; return res
    }
    fun cholesky(): Matrix {
        val l = Matrix(rows, cols)
        for (i in 0 until rows) for (j in 0..i) {
            var sum = 0.0; for (k in 0 until j) sum += l[i, k] * l[j, k]
            if (i == j) { val v = this[i, i] - sum; l[i, j] = if (v > 1e-18) sqrt(v) else 1e-9 }
            else { val d = if (abs(l[j, j]) > 1e-18) l[j, j] else 1e-9; l[i, j] = (this[i, j] - sum) / d }
        }
        return l
    }
    fun trace(): Double { var s = 0.0; for (i in 0 until rows) s += this[i, i]; return s }
    companion object { fun identity(n: Int): Matrix { val m = Matrix(n, n); for (i in 0 until n) m[i, i] = 1.0; return m } }
}
operator fun Double.times(matrix: Matrix): Matrix = matrix * this
