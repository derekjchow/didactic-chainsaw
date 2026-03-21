#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "tensorflow>=2.21.0",
#     "tqdm",
# ]
# ///
"""Compare inference results of two TFLite flatbuffers.

Loads both models, feeds identical random inputs, and checks that outputs
match within a tolerance.  Exit code 0 on match, 1 on mismatch.

Usage:
    uv run compare_tflite.py <model_a.tflite> <model_b.tflite>
"""

import argparse
import sys

import numpy as np
import tensorflow as tf
import tqdm


def load_interpreter(path: str) -> tf.lite.Interpreter:
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_a", help="Path to first TFLite model")
    parser.add_argument("model_b", help="Path to second TFLite model")
    parser.add_argument(
        "-n", "--iterations", type=int, default=10000,
        help="Number of comparison iterations (default: 10000)",
    )
    args = parser.parse_args()

    interp_a = load_interpreter(args.model_a)
    interp_b = load_interpreter(args.model_b)

    inputs_a = interp_a.get_input_details()
    inputs_b = interp_b.get_input_details()

    if len(inputs_a) != len(inputs_b):
        print(
            f"FAIL: model A has {len(inputs_a)} inputs, "
            f"model B has {len(inputs_b)}"
        )
        sys.exit(1)

    rng = np.random.default_rng(seed=42)
    for i in tqdm.tqdm(range(args.iterations)):
        # Feed identical random data to both interpreters.
        for ia, ib in zip(inputs_a, inputs_b):
            dtype = np.dtype(ia["dtype"])
            if np.issubdtype(dtype, np.floating):
                data = rng.uniform(-1.0, 1.0, size=ia["shape"]).astype(dtype)
            elif np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                data = rng.integers(
                    info.min, info.max, size=ia["shape"], dtype=dtype
                )
            else:
                data = rng.standard_normal(ia["shape"]).astype(dtype)

            interp_a.set_tensor(ia["index"], data)
            interp_b.set_tensor(ib["index"], data)

        interp_a.invoke()
        interp_b.invoke()

        outputs_a = interp_a.get_output_details()
        outputs_b = interp_b.get_output_details()

        all_ok = True
        for i, (oa, ob) in enumerate(zip(outputs_a, outputs_b)):
            ra = interp_a.get_tensor(oa["index"])
            rb = interp_b.get_tensor(ob["index"])

            if ra.shape != rb.shape:
                print(
                    f"Output {i}: FAIL shape mismatch {ra.shape} vs {rb.shape}"
                )
                all_ok = False
                continue

            if np.issubdtype(ra.dtype, np.integer):
                match = np.allclose(ra, rb, atol=1, rtol=0)
            else:
                match = np.allclose(ra, rb, atol=1e-5, rtol=1e-5)

            if not match:
                diff = np.abs(ra.astype(np.float64) - rb.astype(np.float64))
                print(
                    f"Output {i}: FAIL  "
                    f"(max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e})"
                )
                all_ok = False

        if not all_ok:
            print("\nMISMATCH detected!")
            sys.exit(1)

    print("Comparison success")

if __name__ == "__main__":
    main()
