"""
======================================================================
NAV-SHIELD STAGE 8
PYTORCH -> TENSORFLOW/KERAS -> TFLITE

VERSION 2

IMPORTANT DESIGN:
    The Stage 8 LSTM remains a REGRESSION model.

    Output 0:
        position_error_m

    Output 1:
        velocity_error_m_s

    TFLite does NOT directly classify NORMAL/ABNORMAL.

    Android performs the decision layer after TFLite inference.

Architecture:
    Input:
        30 x 137

    LSTM:
        input_size = 137
        hidden_size = 64
        num_layers = 2

    FC:
        64 -> 64
        ReLU
        64 -> 2

Outputs:
    0 = position_error_m
    1 = velocity_error_m_s

No retraining.
No raw data modification.
No previous-stage modification.
======================================================================
"""

from pathlib import Path
import sys
import json
import warnings

import numpy as np
import torch
import torch.nn as nn


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent

CHECKPOINT_PATH = (
    BASE_DIR
    / "results"
    / "stage8"
    / "models"
    / "nav_shield_lstm_best.pt"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "stage8"
    / "tflite"
)

MODEL_DIR = (
    OUTPUT_DIR
    / "nav_shield"
)

TFLITE_PATH = (
    MODEL_DIR
    / "nav_shield_lstm_stage8.tflite"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "stage8_model_metadata.json"
)

DECISION_CONFIG_PATH = (
    OUTPUT_DIR
    / "stage8_decision_config.json"
)

TARGET_SCALER_PATH = (
    BASE_DIR
    / "results"
    / "stage8"
    / "reports"
    / "stage8_target_scaler.joblib"
)


SEQUENCE_LENGTH = 30
INPUT_FEATURES = 137
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 2


TARGET_COLUMNS = [
    "position_error_m",
    "velocity_error_m_s",
]


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("NAV-SHIELD STAGE 8: TFLITE CONVERSION")
print("=" * 70)

print()
print("VERSION 2")
print()
print("Design:")
print("  TFLite = regression model")
print("  Android = decision layer")

print()
print("PROJECT ROOT:")
print(BASE_DIR)

print()
print("CHECKPOINT:")
print(CHECKPOINT_PATH)

print()
print("OUTPUT DIRECTORY:")
print(OUTPUT_DIR)


# ======================================================================
# CHECK FILE
# ======================================================================

if not CHECKPOINT_PATH.exists():

    print()
    print("[ERROR] Stage 8 checkpoint not found.")
    print()
    print(CHECKPOINT_PATH)

    sys.exit(1)


# ======================================================================
# IMPORT TENSORFLOW
# ======================================================================

print()
print("=" * 70)
print("CHECKING TENSORFLOW")
print("=" * 70)

try:

    import tensorflow as tf

except ImportError:

    print()
    print("[ERROR] TensorFlow is not installed.")

    print()
    print("Install TensorFlow inside tflite_env.")

    sys.exit(1)


print()
print("TensorFlow version:")
print(tf.__version__)


# ======================================================================
# LOAD CHECKPOINT
# ======================================================================

print()
print("=" * 70)
print("LOADING STAGE 8 CHECKPOINT")
print("=" * 70)

try:

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

except Exception as e:

    print()
    print("[ERROR] Could not load checkpoint.")
    print(str(e))

    sys.exit(1)


print()
print("Checkpoint type:")
print(type(checkpoint).__name__)


# ======================================================================
# EXTRACT STATE DICT
# ======================================================================

if isinstance(checkpoint, dict):

    if "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    elif "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    else:

        state_dict = checkpoint

else:

    print()
    print("[ERROR] Unsupported checkpoint format.")

    sys.exit(1)


# Remove DataParallel prefix if present.

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):

        key = key[7:]

    clean_state_dict[key] = value


state_dict = clean_state_dict


# ======================================================================
# EXPECTED KEYS
# ======================================================================

EXPECTED_KEYS = [

    "lstm.weight_ih_l0",
    "lstm.weight_hh_l0",
    "lstm.bias_ih_l0",
    "lstm.bias_hh_l0",

    "lstm.weight_ih_l1",
    "lstm.weight_hh_l1",
    "lstm.bias_ih_l1",
    "lstm.bias_hh_l1",

    "fc.0.weight",
    "fc.0.bias",

    "fc.2.weight",
    "fc.2.bias",
]


print()
print("=" * 70)
print("VERIFYING CHECKPOINT")
print("=" * 70)


missing_keys = [
    key
    for key in EXPECTED_KEYS
    if key not in state_dict
]


if missing_keys:

    print()
    print("[ERROR] Missing checkpoint keys:")

    for key in missing_keys:

        print("   ", key)

    print()
    print("Available keys:")

    for key in state_dict.keys():

        print("   ", key)

    sys.exit(1)


print()
print("[PASS] All expected checkpoint keys found.")


# ======================================================================
# PYTORCH MODEL
# ======================================================================

print()
print("=" * 70)
print("CREATING EXACT PYTORCH MODEL")
print("=" * 70)


class NavShieldLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True
        )

        self.fc = nn.Sequential(

            nn.Linear(
                HIDDEN_SIZE,
                HIDDEN_SIZE
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_SIZE,
                OUTPUT_SIZE
            )
        )


    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        return self.fc(last_output)


model = NavShieldLSTM()


try:

    model.load_state_dict(
        state_dict,
        strict=True
    )

except Exception as e:

    print()
    print("[ERROR] Could not load model weights.")
    print(str(e))

    sys.exit(1)


model.eval()


print()
print(model)

print()
print("[PASS] PyTorch checkpoint loaded.")


# ======================================================================
# PYTORCH SANITY TEST
# ======================================================================

print()
print("=" * 70)
print("PYTORCH SANITY TEST")
print("=" * 70)


np.random.seed(42)

dummy_input_np = np.random.randn(
    1,
    SEQUENCE_LENGTH,
    INPUT_FEATURES
).astype(
    np.float32
)

dummy_input = torch.from_numpy(
    dummy_input_np
)


with torch.no_grad():

    pytorch_output = model(
        dummy_input
    ).numpy()


print()
print("Input shape:")
print(dummy_input_np.shape)

print()
print("Output shape:")
print(pytorch_output.shape)

print()
print("PyTorch output:")
print(pytorch_output)


if pytorch_output.shape != (
    1,
    OUTPUT_SIZE
):

    print()
    print("[ERROR] Unexpected PyTorch output shape.")

    sys.exit(1)


print()
print("[PASS] PyTorch sanity test passed.")


# ======================================================================
# BUILD KERAS MODEL
# ======================================================================

print()
print("=" * 70)
print("CREATING TENSORFLOW/KERAS MODEL")
print("=" * 70)


tf.keras.backend.clear_session()


keras_input = tf.keras.Input(
    shape=(
        SEQUENCE_LENGTH,
        INPUT_FEATURES
    ),
    batch_size=1,
    dtype=tf.float32,
    name="input"
)


x = tf.keras.layers.LSTM(
    HIDDEN_SIZE,
    return_sequences=True,
    return_state=False,
    name="lstm_0"
)(
    keras_input
)


x = tf.keras.layers.LSTM(
    HIDDEN_SIZE,
    return_sequences=False,
    return_state=False,
    name="lstm_1"
)(
    x
)


x = tf.keras.layers.Dense(
    HIDDEN_SIZE,
    activation="relu",
    name="fc_0"
)(
    x
)


keras_output = tf.keras.layers.Dense(
    OUTPUT_SIZE,
    activation=None,
    name="fc_2"
)(
    x
)


keras_model = tf.keras.Model(
    inputs=keras_input,
    outputs=keras_output
)


# Build model.

_ = keras_model(
    tf.constant(
        dummy_input_np
    )
)


print()
print(keras_model.summary())


# ======================================================================
# COPY PYTORCH WEIGHTS TO KERAS
# ======================================================================

print()
print("=" * 70)
print("COPYING PYTORCH WEIGHTS TO KERAS")
print("=" * 70)


def torch_tensor(
    key
):

    return (
        state_dict[key]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


def convert_lstm_weights(
    keras_layer,
    layer_index
):

    """
    PyTorch LSTM gate order:

        input
        forget
        cell
        output

    Keras LSTM gate order:

        input
        forget
        cell
        output

    Therefore gate order is compatible.

    PyTorch:
        weight_ih = [4H, input]

    Keras:
        kernel = [input, 4H]

    So transpose is required.

    PyTorch has two biases:

        bias_ih
        bias_hh

    Keras uses one bias vector.

    Therefore:

        keras_bias =
            bias_ih + bias_hh
    """

    suffix = f"_l{layer_index}"

    weight_ih = torch_tensor(
        "lstm.weight_ih" + suffix
    )

    weight_hh = torch_tensor(
        "lstm.weight_hh" + suffix
    )

    bias_ih = torch_tensor(
        "lstm.bias_ih" + suffix
    )

    bias_hh = torch_tensor(
        "lstm.bias_hh" + suffix
    )


    kernel = weight_ih.T

    recurrent_kernel = weight_hh.T

    bias = (
        bias_ih
        +
        bias_hh
    )


    keras_layer.set_weights(
        [
            kernel,
            recurrent_kernel,
            bias
        ]
    )


    print()
    print(
        f"[PASS] LSTM layer {layer_index} weights copied."
    )


# Layer 0

convert_lstm_weights(
    keras_model.get_layer(
        "lstm_0"
    ),
    0
)


# Layer 1

convert_lstm_weights(
    keras_model.get_layer(
        "lstm_1"
    ),
    1
)


# ======================================================================
# COPY FC WEIGHTS
# ======================================================================

fc0_weight = torch_tensor(
    "fc.0.weight"
)

fc0_bias = torch_tensor(
    "fc.0.bias"
)

fc2_weight = torch_tensor(
    "fc.2.weight"
)

fc2_bias = torch_tensor(
    "fc.2.bias"
)


keras_model.get_layer(
    "fc_0"
).set_weights(
    [
        fc0_weight.T,
        fc0_bias
    ]
)


keras_model.get_layer(
    "fc_2"
).set_weights(
    [
        fc2_weight.T,
        fc2_bias
    ]
)


print()
print("[PASS] FC weights copied.")


# ======================================================================
# PYTORCH VS KERAS TEST
# ======================================================================

print()
print("=" * 70)
print("PYTORCH VS KERAS NUMERICAL TEST")
print("=" * 70)


keras_output_np = keras_model(
    dummy_input_np,
    training=False
).numpy()


difference = np.abs(
    pytorch_output
    -
    keras_output_np
)


max_difference = float(
    np.max(difference)
)

mean_difference = float(
    np.mean(difference)
)


print()
print("PyTorch output:")
print(pytorch_output)

print()
print("Keras output:")
print(keras_output_np)

print()
print("Maximum absolute difference:")
print(max_difference)

print()
print("Mean absolute difference:")
print(mean_difference)


if max_difference > 1e-3:

    print()
    print(
        "[ERROR] PyTorch and Keras outputs differ too much."
    )

    print()
    print(
        "TFLite conversion stopped."
    )

    sys.exit(1)


print()
print(
    "[PASS] PyTorch and Keras outputs match."
)


# ======================================================================
# SAVE SAVEDMODEL
# ======================================================================

SAVED_MODEL_DIR = (
    OUTPUT_DIR
    / "keras_model"
)


if SAVED_MODEL_DIR.exists():

    import shutil

    shutil.rmtree(
        SAVED_MODEL_DIR
    )


print()
print("=" * 70)
print("SAVING TENSORFLOW MODEL")
print("=" * 70)


try:

    tf.saved_model.save(
        keras_model,
        str(
            SAVED_MODEL_DIR
        )
    )

except Exception as e:

    print()
    print(
        "[ERROR] TensorFlow model save failed."
    )

    print(str(e))

    sys.exit(1)


print()
print("[PASS] TensorFlow model saved.")


# ======================================================================
# TFLITE CONVERSION
# ======================================================================

print()
print("=" * 70)
print("CONVERTING TENSORFLOW -> TFLITE")
print("=" * 70)


try:

    converter = (
        tf.lite.TFLiteConverter
        .from_saved_model(
            str(
                SAVED_MODEL_DIR
            )
        )
    )

    # Use normal TFLite built-in operators.
    #
    # We deliberately DO NOT use:
    #
    #     SELECT_TF_OPS
    #
    # because that caused TensorList/Flex problems
    # in the previous conversion pipeline.

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]

    tflite_model = converter.convert()

except Exception as e:

    print()
    print(
        "[ERROR] TFLite conversion failed."
    )

    print()
    print(str(e))

    sys.exit(1)


with open(
    TFLITE_PATH,
    "wb"
) as f:

    f.write(
        tflite_model
    )


print()
print(
    "[PASS] TFLite model created."
)

print()
print("TFLite file:")
print(TFLITE_PATH)

print()
print(
    "TFLite size: "
    f"{len(tflite_model) / (1024 * 1024):.3f} MB"
)


# ======================================================================
# TFLITE SANITY TEST
# ======================================================================

print()
print("=" * 70)
print("TFLITE SANITY TEST")
print("=" * 70)


try:

    interpreter = tf.lite.Interpreter(
        model_path=str(
            TFLITE_PATH
        )
    )

    interpreter.allocate_tensors()

except Exception as e:

    print()
    print(
        "[ERROR] Could not load TFLite model."
    )

    print(str(e))

    sys.exit(1)


input_details = (
    interpreter.get_input_details()
)

output_details = (
    interpreter.get_output_details()
)


print()
print("INPUT DETAILS")

print(
    "  name :",
    input_details[0]["name"]
)

print(
    "  shape:",
    input_details[0]["shape"]
)

print(
    "  dtype:",
    input_details[0]["dtype"]
)


print()
print("OUTPUT DETAILS")

print(
    "  name :",
    output_details[0]["name"]
)

print(
    "  shape:",
    output_details[0]["shape"]
)

print(
    "  dtype:",
    output_details[0]["dtype"]
)


expected_input_shape = np.array(
    [
        1,
        SEQUENCE_LENGTH,
        INPUT_FEATURES
    ]
)


if not np.array_equal(
    input_details[0]["shape"],
    expected_input_shape
):

    print()
    print(
        "[ERROR] Unexpected TFLite input shape."
    )

    sys.exit(1)


if not np.array_equal(
    output_details[0]["shape"],
    np.array(
        [
            1,
            OUTPUT_SIZE
        ]
    )
):

    print()
    print(
        "[ERROR] Unexpected TFLite output shape."
    )

    sys.exit(1)


interpreter.set_tensor(
    input_details[0]["index"],
    dummy_input_np
)


interpreter.invoke()


tflite_output = (
    interpreter
    .get_tensor(
        output_details[0]["index"]
    )
)


print()
print("TFLite output:")
print(tflite_output)


# ======================================================================
# PYTORCH VS TFLITE
# ======================================================================

print()
print("=" * 70)
print("PYTORCH VS TFLITE NUMERICAL TEST")
print("=" * 70)


tflite_difference = np.abs(
    pytorch_output
    -
    tflite_output
)


tflite_max_difference = float(
    np.max(
        tflite_difference
    )
)

tflite_mean_difference = float(
    np.mean(
        tflite_difference
    )
)


print()
print(
    "Maximum absolute difference:"
)

print(
    tflite_max_difference
)


print()
print(
    "Mean absolute difference:"
)

print(
    tflite_mean_difference
)


if tflite_max_difference > 1e-2:

    print()
    print(
        "[ERROR] TFLite output differs too much from PyTorch."
    )

    print()
    print(
        "The model will NOT be marked as valid."
    )

    sys.exit(1)


print()
print(
    "[PASS] TFLite output matches PyTorch."
)


# ======================================================================
# TARGET SCALER
# ======================================================================

print()
print("=" * 70)
print("CHECKING TARGET SCALER")
print("=" * 70)


target_scaler_available = (
    TARGET_SCALER_PATH.exists()
)


if target_scaler_available:

    try:

        import joblib

        target_scaler = joblib.load(
            TARGET_SCALER_PATH
        )

        target_mean = (
            target_scaler.mean_
            .astype(
                np.float64
            )
            .tolist()
        )

        target_scale = (
            target_scaler.scale_
            .astype(
                np.float64
            )
            .tolist()
        )


        print()
        print(
            "[PASS] Stage 8 target scaler found."
        )

        print()
        print(
            "Target mean:"
        )

        print(
            target_mean
        )

        print()
        print(
            "Target scale:"
        )

        print(
            target_scale
        )

    except Exception as e:

        print()
        print(
            "[WARNING] Could not read target scaler."
        )

        print(str(e))

        target_scaler_available = False

else:

    print()
    print(
        "[WARNING] Target scaler not found."
    )

    print()
    print(
        TARGET_SCALER_PATH
    )


# ======================================================================
# DECISION CONFIGURATION
# ======================================================================

print()
print("=" * 70)
print("CREATING ANDROID DECISION CONFIGURATION")
print("=" * 70)


"""
IMPORTANT:

The neural network outputs numerical navigation errors.

Android should perform the final decision.

We use a two-stage decision:

    NORMAL
        |
        | error exceeds warning threshold
        v
    WARNING
        |
        | error exceeds abnormal threshold
        v
    ABNORMAL / DRIFT

The thresholds below are deliberately configuration values,
NOT model weights.

They can be tuned later without retraining the LSTM.

For the first integration version, conservative defaults are used.

IMPORTANT:
These values are NOT claimed to be scientifically optimal.
They are engineering thresholds for the decision layer.

Tune them using validation data before final deployment.
"""


DECISION_CONFIG = {

    "version": 2,

    "model_outputs": {

        "output_0": {
            "name": "position_error_m",
            "unit": "m"
        },

        "output_1": {
            "name": "velocity_error_m_s",
            "unit": "m/s"
        }
    },

    "decision_logic": {

        "position_warning_m": 10.0,

        "position_abnormal_m": 20.0,

        "velocity_warning_m_s": 1.0,

        "velocity_abnormal_m_s": 2.0,

        "required_abnormal_signals": 1
    },

    "rules": {

        "normal":
            "Both errors are below warning thresholds.",

        "warning":
            "At least one error reaches warning threshold but abnormal threshold is not reached.",

        "abnormal":
            "At least one error reaches abnormal threshold.",

        "drift_detected":
            "ABNORMAL state caused by navigation-error threshold exceedance."
    },

    "android_output": {

        "state_values": [
            "NORMAL",
            "WARNING",
            "ABNORMAL"
        ],

        "drift_values": [
            "NO",
            "YES"
        ]
    },

    "target_scaler": {

        "required_for_inverse_transform":
            target_scaler_available,

        "mean":
            target_mean
            if target_scaler_available
            else None,

        "scale":
            target_scale
            if target_scaler_available
            else None
    }
}


with open(
    DECISION_CONFIG_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        DECISION_CONFIG,
        f,
        indent=4
    )


print()
print(
    "[PASS] Android decision configuration saved."
)

print()
print(
    DECISION_CONFIG_PATH
)


# ======================================================================
# CREATE METADATA
# ======================================================================

metadata = {

    "stage": 8,

    "model_name":
        "nav_shield_lstm",

    "architecture": {

        "input_features":
            INPUT_FEATURES,

        "sequence_length":
            SEQUENCE_LENGTH,

        "hidden_size":
            HIDDEN_SIZE,

        "lstm_layers":
            NUM_LAYERS,

        "fc":
            [
                "Linear(64,64)",
                "ReLU",
                "Linear(64,2)"
            ]
    },

    "outputs": TARGET_COLUMNS,

    "output_semantics": {

        "output_0":
            "position_error_m",

        "output_1":
            "velocity_error_m_s"
    },

    "deployment_design":
        "TFLite regression + Android decision layer",

    "model_retrained":
        False,

    "raw_data_modified":
        False,

    "previous_stages_modified":
        False,

    "uncategorised_dataset_used":
        False,

    "pytorch_sanity_test":
        True,

    "keras_equivalence_test":
        True,

    "tflite_inference_test":
        True,

    "pytorch_keras_max_absolute_difference":
        max_difference,

    "pytorch_keras_mean_absolute_difference":
        mean_difference,

    "pytorch_tflite_max_absolute_difference":
        tflite_max_difference,

    "pytorch_tflite_mean_absolute_difference":
        tflite_mean_difference,

    "tflite_model":
        str(TFLITE_PATH),

    "decision_config":
        str(DECISION_CONFIG_PATH),

    "target_scaler":
        str(TARGET_SCALER_PATH)
}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


print()
print(
    "[PASS] Metadata saved."
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print()
print("=" * 70)
print("STAGE 8 TFLITE CONVERSION COMPLETE")
print("=" * 70)

print()
print("SUCCESS")

print()
print("TFLite model:")
print(TFLITE_PATH)

print()
print("Metadata:")
print(METADATA_PATH)

print()
print("Decision configuration:")
print(DECISION_CONFIG_PATH)

print()
print("Architecture:")
print("  Input       : 30 x 137")
print("  LSTM hidden : 64")
print("  LSTM layers : 2")
print("  FC          : 64 -> 64 -> 2")

print()
print("Outputs:")
print("  Output 0    : position_error_m")
print("  Output 1    : velocity_error_m_s")

print()
print("Deployment:")
print("  TFLite      : numerical error prediction")
print("  Android     : NORMAL / WARNING / ABNORMAL")
print("  Android     : DRIFT YES / NO")

print()
print("Model retraining : NO")
print("Raw data modified : NO")
print("Previous stages modified : NO")
print("Uncategorised dataset : NOT USED")

print()
print("=" * 70)