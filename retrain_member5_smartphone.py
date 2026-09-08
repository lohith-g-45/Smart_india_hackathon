import subprocess
import sys
from pathlib import Path

def run_stage(script_name):
    print(f"\n>>> Running {script_name}...")
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f">>> {script_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f">>> ERROR in {script_name}: {e}")
        sys.exit(1)

def main():
    print("=" * 80)
    print("MEMBER 5 SMARTPHONE-ONLY RETRAINING ORCHESTRATOR")
    print("=" * 80)

    # 1. Update Preprocessing Metadata and Splits
    run_stage("stage4.py")

    # 2. Retrain LSTM with 107 Smartphone Features
    run_stage("stage8.py")

    # 3. Export to TFLite for Android Deployment
    run_stage("tflite.py")

    print("\n" + "=" * 80)
    print("MIGRATION COMPLETE")
    print("New Model: results/stage8/models/nav_shield_lstm_android_107.pt")
    print("New TFLite: results/stage8/tflite_android/nav_shield_107/nav_shield_lstm_android_107.tflite")
    print("=" * 80)

if __name__ == "__main__":
    main()
