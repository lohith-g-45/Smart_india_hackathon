package com.navshield.map.engine.math

import kotlin.math.sqrt

/**
 * A minimal matrix implementation for the Unscented Kalman Filter.
 */
class Matrix(val rows: Int, val cols: Int, val data: DoubleArray = DoubleArray(rows * cols)) {

    operator fun get(r: Int, c: Int): Double = data[r * cols + c]
    operator fun set(r: Int, c: Int, v: Double) {
        data[r * cols + c] = v
    }

    fun copy(): Matrix = Matrix(rows, cols, data.copyOf())

    operator fun plus(other: Matrix): Matrix {
        require(rows == other.rows && cols == other.cols)
        val res = Matrix(rows, cols)
        for (i in data.indices) res.data[i] = data[i] + other.data[i]
        return res
    }

    operator fun minus(other: Matrix): Matrix {
        require(rows == other.rows && cols == other.cols)
        val res = Matrix(rows, cols)
        for (i in data.indices) res.data[i] = data[i] - other.data[i]
        return res
    }

    operator fun times(other: Matrix): Matrix {
        require(cols == other.rows)
        val res = Matrix(rows, other.cols)
        for (r in 0 until rows) {
            for (c in 0 until other.cols) {
                var sum = 0.0
                for (k in 0 until cols) {
                    sum += this[r, k] * other[k, c]
                }
                res[r, c] = sum
            }
        }
        return res
    }

    operator fun times(scalar: Double): Matrix {
        val res = Matrix(rows, cols)
        for (i in data.indices) res.data[i] = data[i] * scalar
        return res
    }

    operator fun plusAssign(other: Matrix) {
        require(rows == other.rows && cols == other.cols)
        for (i in data.indices) data[i] += other.data[i]
    }

    operator fun minusAssign(other: Matrix) {
        require(rows == other.rows && cols == other.cols)
        for (i in data.indices) data[i] -= other.data[i]
    }

    fun transpose(): Matrix {
        val res = Matrix(cols, rows)
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                res[c, r] = this[r, c]
            }
        }
        return res
    }

    fun trace(): Double {
        require(rows == cols)
        var sum = 0.0
        for (i in 0 until rows) sum += this[i, i]
        return sum
    }

    /**
     * Cholesky Decomposition for Sigma point generation.
     * Returns the lower triangular matrix L such that L * L^T = this.
     */
    fun cholesky(): Matrix {
        require(rows == cols)
        val l = Matrix(rows, cols)
        for (i in 0 until rows) {
            for (j in 0..i) {
                var sum = 0.0
                for (k in 0 until j) sum += l[i, k] * l[j, k]
                if (i == j) {
                    val v = this[i, i] - sum
                    l[i, j] = if (v > 0) sqrt(v) else 0.0
                } else {
                    l[i, j] = (1.0 / l[j, j] * (this[i, j] - sum))
                }
            }
        }
        return l
    }

    fun inverse(): Matrix {
        require(rows == cols)
        if (rows == 1) {
            val res = Matrix(1, 1)
            res[0, 0] = 1.0 / this[0, 0]
            return res
        }
        val n = rows
        val a = Array(n) { r -> DoubleArray(2 * n) { c -> if (c < n) this[r, c] else if (c - n == r) 1.0 else 0.0 } }
        
        for (i in 0 until n) {
            var pivot = a[i][i]
            if (pivot == 0.0) {
                for (k in i + 1 until n) {
                    if (a[k][i] != 0.0) {
                        val tmp = a[i]
                        a[i] = a[k]
                        a[k] = tmp
                        break
                    }
                }
                pivot = a[i][i]
            }
            if (pivot == 0.0) throw ArithmeticException("Matrix is singular")
            
            for (j in 0 until 2 * n) a[i][j] /= pivot
            for (k in 0 until n) {
                if (k != i) {
                    val factor = a[k][i]
                    for (j in 0 until 2 * n) a[k][j] -= factor * a[i][j]
                }
            }
        }
        
        val res = Matrix(n, n)
        for (r in 0 until n) {
            for (c in 0 until n) {
                res[r, c] = a[r][c + n]
            }
        }
        return res
    }

    companion object {
        fun identity(n: Int): Matrix {
            val m = Matrix(n, n)
            for (i in 0 until n) m[i, i] = 1.0
            return m
        }
    }
}

operator fun Double.times(matrix: Matrix): Matrix = matrix * this
