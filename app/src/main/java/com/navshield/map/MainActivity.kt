package com.navshield.map

import android.Manifest
import android.os.Bundle
import android.widget.TextView
import android.widget.LinearLayout
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import com.navshield.map.engine.MainViewModel

class MainActivity : AppCompatActivity() {

    private val viewModel: MainViewModel by viewModels()

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val granted = permissions[Manifest.permission.ACCESS_FINE_LOCATION] == true
        if (granted) {
            viewModel.startNavigation()
        } else {
            updateStatus("PERMISSION DENIED")
        }
    }

    private lateinit var statusText: TextView
    private lateinit var latText: TextView
    private lateinit var lonText: TextView
    private lateinit var hypoText: TextView
    private lateinit var confidenceText: TextView
    private lateinit var modeText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        setupUI()

        viewModel.status.observe(this) { status ->
            statusText.text = "Status: $status"
        }

        viewModel.trustState.observe(this) { state ->
            if (state != null) {
                latText.text = "Lat: %.6f".format(state.final_latitude)
                lonText.text = "Lon: %.6f".format(state.final_longitude)
                hypoText.text = "Active Hypo: ${state.active_hypothesis}"
                confidenceText.text = "Confidence: %.2f".format(state.position_confidence)
                modeText.text = "Mode: ${state.navigation_mode}"
            }
        }

        permissionLauncher.launch(arrayOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        ))
    }

    private fun setupUI() {
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
        }

        val title = TextView(this).apply {
            text = "NAV-SHIELD"
            textSize = 24f
            setPadding(0, 0, 0, 32)
        }
        layout.addView(title)

        statusText = TextView(this)
        layout.addView(statusText)

        latText = TextView(this)
        layout.addView(latText)

        lonText = TextView(this)
        layout.addView(lonText)

        hypoText = TextView(this)
        layout.addView(hypoText)

        confidenceText = TextView(this)
        layout.addView(confidenceText)

        modeText = TextView(this)
        layout.addView(modeText)

        setContentView(layout)
    }

    private fun updateStatus(status: String) {
        statusText.text = "Status: $status"
    }
}
