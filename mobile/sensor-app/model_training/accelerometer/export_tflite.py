"""Convert the trained Keras model to TFLite for mobile inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from config import ARTIFACTS_DIR


def export_tflite(
    keras_model_path: Path = ARTIFACTS_DIR / "accel_drowsiness_cnn.keras",
    tflite_path: Path = ARTIFACTS_DIR / "accel_drowsiness_cnn.tflite",
    quantize: bool = False,
) -> Path:
    keras_model_path = Path(keras_model_path)
    tflite_path = Path(tflite_path)
    if not keras_model_path.exists():
        raise FileNotFoundError(f"Keras model not found: {keras_model_path}")

    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"Saved TFLite model: {tflite_path}")
    return tflite_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export accelerometer model to TFLite.")
    parser.add_argument("--keras-model", type=Path, default=ARTIFACTS_DIR / "accel_drowsiness_cnn.keras")
    parser.add_argument("--tflite-out", type=Path, default=ARTIFACTS_DIR / "accel_drowsiness_cnn.tflite")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()
    export_tflite(args.keras_model, args.tflite_out, args.quantize)


if __name__ == "__main__":
    main()
