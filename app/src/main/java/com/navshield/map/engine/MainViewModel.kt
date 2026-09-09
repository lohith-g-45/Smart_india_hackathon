package com.navshield.map.engine

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.navshield.map.contract.NavShieldTrustState
import com.navshield.map.members.RoutingEngine
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Lifecycle-aware ViewModel for managing the NAV-SHIELD navigation pipeline.
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val repository = SqliteRoadRepository(application)
    private val mapDataModule = MapDataModule(application, repository)
    private val sensorSource = AndroidSensorFusionSource(application)
    private val mapEngine = BasicMapMatchingEngine(repository)
    private val driftGuardian = Member5TfliteEngine(application)
    private val insEngine = Member3InsEngine()
    
    // Placeholder implementations for Member 3
    private val router = object : RoutingEngine {
        override fun updateGuidance(lat: Double, lon: Double, segmentId: String) {}
        override fun getStatus(): MemberStatus = MemberStatus.CONNECTED
    }

    private val pipeline = NavShieldPipeline(mapEngine, sensorSource, router, driftGuardian, insEngine)
    
    private val _trustState = MutableLiveData<NavShieldTrustState?>()
    val trustState: LiveData<NavShieldTrustState?> = _trustState

    private val _status = MutableLiveData<String>("INITIALIZING")
    val status: LiveData<String> = _status

    private var job: Job? = null

    init {
        viewModelScope.launch {
            _status.postValue("INITIALIZING DATA")
            mapDataModule.initialize()
            driftGuardian.initialize()
            insEngine.initialize(System.currentTimeMillis())
            _status.postValue("READY")
        }
    }

    fun startNavigation() {
        if (job != null) return
        
        sensorSource.start()
        
        job = viewModelScope.launch {
            _status.postValue("NAVIGATING")
            while (true) {
                val result = pipeline.processCycle(System.currentTimeMillis())
                if (result != null) {
                    _trustState.postValue(result.trustState)
                }
                delay(100) // 10Hz processing loop
            }
        }
    }

    fun stopNavigation() {
        job?.cancel()
        job = null
        sensorSource.stop()
        _status.postValue("READY")
    }

    override fun onCleared() {
        super.onCleared()
        stopNavigation()
    }
}
