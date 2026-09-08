package com.navshield.map

import com.navshield.map.contract.RoadNode
import com.navshield.map.contract.RoadSegment
import com.navshield.map.engine.RoadDataIngestor
import com.navshield.map.engine.SqliteRoadRepository
import org.junit.Assert.*
import org.junit.Test
import org.mockito.Mockito.*

class RoadDataIngestorTest {

    @Test
    fun `parsing valid json producing nodes and segments`() {
        val json = """
        {
          "nodes": {
            "N1": {"lat": 12.0, "lon": 77.0},
            "N2": {"lat": 12.1, "lon": 77.0}
          },
          "ways": [
            {
              "id": "W1",
              "nodes": ["N1", "N2"],
              "tags": {"highway": "primary", "name": "Main St"}
            }
          ],
          "elevation_m": {
            "N1": 900.0,
            "N2": 950.0
          }
        }
        """.trimIndent()

        val mockRepo = mock(SqliteRoadRepository::class.java)
        val ingestor = RoadDataIngestor(mock(android.content.Context::class.java), mockRepo)

        ingestor.ingestJson(json)

        // Capture nodes and segments passed to repo
        val nodesCaptor = org.mockito.ArgumentCaptor.forClass(List::class.java)
        val segmentsCaptor = org.mockito.ArgumentCaptor.forClass(List::class.java)
        
        verify(mockRepo).insertRoadData(nodesCaptor.capture() as List<RoadNode>?, segmentsCaptor.capture() as List<Pair<RoadSegment, Pair<Double, Double>>>?)

        val nodes = nodesCaptor.value as List<RoadNode>
        val segments = segmentsCaptor.value as List<Pair<RoadSegment, Pair<Double, Double>>>

        assertEquals(2, nodes.size)
        // One forward, one reverse segment
        assertEquals(2, segments.size)
        
        val segment = segments[0].first
        assertEquals("W1_0", segment.segmentId)
        assertTrue(segment.length > 0)
        assertEquals(0.0, segment.bearing, 0.1) // North
        assertEquals(0.45, segment.slope, 0.1) // (50m / 11119m) * 100
    }
}
