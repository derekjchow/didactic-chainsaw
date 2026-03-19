#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "tensorflow>=2.18",
#     "numpy<2",
# ]
# ///
"""Export the first three layers of MobileNet as a quantized TFLite file."""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def build_first_three_layers() -> tf.keras.Model:
    """Return a model consisting of the first three layers of MobileNet."""
    full_model = MobileNet(weights="imagenet", input_shape=(224, 224, 3))
    # Layer 0 is the input layer; layers 1-3 are the first three functional layers.
    truncated = tf.keras.Model(
        inputs=full_model.input,
        outputs=full_model.layers[9].output,
        name="mobilenet_first3",
    )
    truncated.summary()
    return truncated


def representative_dataset():
    """Yield sample inputs for full-integer quantization calibration."""
    for _ in range(100):
        yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]


def export_tflite(model: tf.keras.Model) -> None:
    """Convert the model to a quantized TFLite file and write it to disk."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "mobilenet_first3.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Wrote quantized TFLite model to {out_path}")


def main() -> None:
    model = build_first_three_layers()
    export_tflite(model)


if __name__ == "__main__":
    main()
