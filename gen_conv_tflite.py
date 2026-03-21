#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "tensorflow>=2.21.0",
# ]
# ///
"""Generate a quantized int8 TFLite model with a single Conv2D layer.

The model has:
- Input: 320x320x1 (grayscale image), batch size 1
- Conv2D: 4x4 kernel, stride 3x3, 1 output filter, VALID padding
- Output quantized to int8
"""

import numpy as np
import tensorflow as tf


def build_model():
    inp = tf.keras.Input(shape=(320, 320, 1), batch_size=1, name="image")
    out = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=(4, 4),
        strides=(3, 3),
        padding="valid",
        use_bias=True,
        activation="relu6",
        name="conv",
    )(inp)
    return tf.keras.Model(inputs=inp, outputs=out)


def representative_dataset():
    for _ in range(100):
        yield [np.random.uniform(0, 1, size=(1, 320, 320, 1)).astype(np.float32)]


def main():
    model = build_model()
    model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    output_path = "model.tflite"
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Wrote {output_path} ({len(tflite_model)} bytes)")


if __name__ == "__main__":
    main()
