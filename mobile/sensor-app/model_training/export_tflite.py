import argparse
from pathlib import Path

import tensorflow as tf

from config import MODELS_DIR


def export_tflite(
    keras_model_path: Path = MODELS_DIR / "drowsiness_cnn.keras",
    tflite_path: Path = MODELS_DIR / "drowsiness_cnn.tflite",
    quantize: bool = False, #whether or not to shrink the model using quantization
) -> Path:
    keras_model_path = Path(keras_model_path)
    tflite_path = Path(tflite_path)

    if not keras_model_path.exists():
        raise FileNotFoundError(f"Keras model not found: {keras_model_path}")

    # Load Keras model (the trained CNN) into memory
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model) #create converter object
    # enables post training quantization (shrink model to make faster)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert() # convert
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved TFLite model: {tflite_path}")
    return tflite_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert trained Keras model to TFLite.")
    parser.add_argument("--keras-model", type=Path, default=MODELS_DIR / "drowsiness_cnn.keras")
    parser.add_argument("--tflite-out", type=Path, default=MODELS_DIR / "drowsiness_cnn.tflite")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    export_tflite(args.keras_model, args.tflite_out, args.quantize)


if __name__ == "__main__":
    main()

