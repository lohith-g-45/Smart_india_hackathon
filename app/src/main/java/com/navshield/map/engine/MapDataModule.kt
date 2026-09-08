package com.navshield.map.engine

import android.content.Context

/**
 * Lifecycle component for managing road graph data initialization.
 */
class MapDataModule(private val context: Context, private val repository: SqliteRoadRepository) {

    /**
     * Ensures the road database is populated and ready for map matching.
     */
    fun initialize() {
        if (!repository.isReady()) {
            val ingestor = RoadDataIngestor(context, repository)
            // Initializing with the sample hill roads dataset
            ingestor.ingest("sample_hill_roads.json")
        }
    }
}
