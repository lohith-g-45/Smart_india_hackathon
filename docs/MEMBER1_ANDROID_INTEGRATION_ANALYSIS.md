# Member 1 Android Integration Analysis

## 1. Dependency Analysis
The Member 1 module relies on several heavy Python libraries:
* **NumPy/SciPy:** Core mathematical operations and spatial indexing (cKDTree).
* **NetworkX:** Graph processing for road networks.
* **PyTorch / PyTorch Geometric:** Used for GNN training and reference implementation.
* **ONNX Runtime:** Intended for production inference.

## 2. Practical Execution on Android
Executing the raw Python code directly on Android is **not recommended** for the final production build for the following reasons:
1. **Performance:** Python execution overhead on mobile is significant.
2. **Size:** Packaging full NumPy, SciPy, and NetworkX via Chaquopy would bloat the APK by >100MB.
3. **Complexity:** `torch-geometric` and `pyosmium` are extremely difficult to build for Android ARM architectures.

## 3. Recommended Integration Strategy (Bridge/Hybrid)
To maintain the ACTUAL Member 1 logic while being Android-performant:
* **Logic Porting:** The graph traversal (NetworkX) and spatial lookup (SciPy) should eventually be ported to Kotlin/Native or C++ if real-time performance is a bottleneck.
* **ONNX Inference:** The GNN re-ranking model (Phase F) should be exported to `.onnx` format. Member 4 will then use the **ONNX Runtime Android library** to run this model directly in Kotlin.
* **Data Access:** As seen in Member 1's `android_stub`, an **SQLite spatial database** should be used for on-device candidate lookups, mirroring the `SpatialIndex` logic.

## 4. Immediate Phase 1 Approach
We will define **Kotlin Data Contracts** that exactly match Member 1's `contract.py`. This ensures that whether we use a Python bridge (for testing) or an ONNX/Kotlin implementation (for production), the data boundary remains stable.

## 5. Conclusion
Direct Python execution is **possible** for prototyping (using Chaquopy), but **impractical** for a production "SHIELD" system. The bridge will be formed by:
1. Kotlin Data Classes (the "Contract").
2. ONNX models for ML components.
3. SQLite for the road graph/spatial data.
