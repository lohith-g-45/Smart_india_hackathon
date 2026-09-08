package com.navshield.map.engine

import com.navshield.map.contract.MapMatchQuery
import com.navshield.map.contract.MapMatchResult

/**
 * PRODUCTION-READY ARCHITECTURE for Member 1 Integration.
 * 
 * This class is designed to wrap the ONNX Runtime for learned re-ranking
 * and SQLite for spatial indexing.
 * 
 * STATUS: WAITING_FOR_MEMBER (Actual ONNX weights and SQLite DB are not yet available).
 */
class Member1OnnxEngine(private val repository: RoadRepository) : MapMatchingEngine {

    private val basicEngine = BasicMapMatchingEngine(repository)

    override fun isReady(): Boolean {
        // In production, this would check for the existence of:
        // 1. navshield_roads.sqlite
        // 2. map_matching_gnn.onnx
        return repository.isReady()
    }

    override fun match(query: MapMatchQuery): MapMatchResult {
        if (!isReady()) {
            throw IllegalStateException("Member 1 Engine is not ready (missing models/DB)")
        }
        
        // 1. Run basic map matching as fallback/baseline
        val result = basicEngine.match(query)

        // 2. Future: Run ONNX Inference for learned re-ranking
        // This is where the ONNX runtime would be called to re-score result.candidateRoads
        
        return result
    }
}
