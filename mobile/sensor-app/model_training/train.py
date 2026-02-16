import argparse
import json
from pathlib import Path

import tensorflow as tf

from config import CLASS_NAMES, IMG_SIZE, MODELS_DIR, PROCESSED_DIR

# Build a CNN model, Load processed images into TensorFlow datasets, Trains model and saves it

def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)) -> tf.keras.Model: #define neural network structure

    # randomly augments training images for more robustly trained model, keras sequential definition
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    # main CNN model, builds a stack of layers
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape), # square image, 3 channels RGB
            data_augmentation,
            tf.keras.layers.Rescaling(1.0 / 255.0), # convert pixel values (normalize data for neural network 0-255 > 0-1)
            #First convolution
            tf.keras.layers.Conv2D(16, 3, activation="relu"), # detect feature (eye shape, mesh pattern) (increasing filter 16 > 32 > 64)
            tf.keras.layers.BatchNormalization(), # Stabilize training
            tf.keras.layers.MaxPooling2D(), # reduce image size keeping strongest feature
            tf.keras.layers.Dropout(0.1), # randomly disable neuron during training (prevents overfitting)
            # Second Convolution
            tf.keras.layers.Conv2D(32, 5, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.1),
            # Third Convolution
            tf.keras.layers.Conv2D(64, 7, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.1),
            #Flatten results to feed into a DNN
            tf.keras.layers.Flatten(), # turn 2d feature map into 1d vector
            tf.keras.layers.Dense(128, activation="relu"), # learns abstract combination
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation="relu"), # refine
            # Only 1 output neuron
            tf.keras.layers.Dense(1, activation="sigmoid"), # final output (sigmoid bc binary classification Active 1 - Drowsy 0)
        ]
    )
    #Define compilation 
    model.compile(
        optimizer="adam", # default
        loss="binary_crossentropy", # for 2 classes
        metrics=["accuracy"],
    )
    return model

# Load processed images into TensorFlow datasets
def make_datasets(
    processed_dir: Path,
    batch_size: int,
    validation_split: float,
    seed: int,
):
    # check if processed folder exists
    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    # training dataset split 80%
    train_ds = tf.keras.utils.image_dataset_from_directory( # image_dataset_from_directory assumes file structure (processed/: active/ drowsy/)
        processed_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=True,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        class_names=CLASS_NAMES,
    )

    # Validation split 20%
    val_ds = tf.keras.utils.image_dataset_from_directory(
        processed_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=False,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        class_names=CLASS_NAMES,
    )

    #improve performance, load next batch while training
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune) 
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds # return datasets


def train_and_save(
    processed_dir: Path = PROCESSED_DIR,
    artifacts_dir: Path = MODELS_DIR,
    epochs: int = 20,
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
) -> tuple[Path, Path]: # return path for model and labels
    
    # load datasets
    train_ds, val_ds = make_datasets(processed_dir, batch_size, validation_split, seed) 

    # build model
    model = build_model()

    # trains for epochs
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # gets validation loss and accuracy
    eval_metrics = model.evaluate(val_ds, verbose=0)

    artifacts_dir = Path(artifacts_dir) # model output path
    artifacts_dir.mkdir(parents=True, exist_ok=True) # make directories if needed
    #create filenames for outputs
    model_path = artifacts_dir / "drowsiness_cnn.keras"
    labels_path = artifacts_dir / "labels.json"
    metrics_path = artifacts_dir / "metrics.json"

    #save model
    model.save(model_path)
    # with this file open, do ...
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"class_names": CLASS_NAMES}, f, indent=2)

    history_json = {k: [float(v) for v in values] for k, values in history.history.items()} # format and store training metrics per epoch
    metrics_payload = {
        "eval_loss": float(eval_metrics[0]),
        "eval_accuracy": float(eval_metrics[1]),
        "history": history_json,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved metrics: {metrics_path}")
    return model_path, labels_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train drowsiness CNN from processed dataset.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_and_save(
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

