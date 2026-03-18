import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import CLASS_NAMES, IMG_SIZE, MODELS_DIR, PROCESSED_BASE_DIR

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

    # Validation split 20%.
    # shuffle=True with the same seed ensures the file list is shuffled identically
    # to the training split, so both classes appear in the validation set.
    # (shuffle=False sorts files alphabetically, causing the split to land entirely
    # in whichever class comes last alphabetically.)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        processed_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        shuffle=True,
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
    processed_dir: Path = PROCESSED_BASE_DIR,
    artifacts_dir: Path = MODELS_DIR,
    epochs: int = 20,
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
) -> tuple[Path, Path]: # return path for model and labels
    
    # load datasets
    train_ds, val_ds = make_datasets(processed_dir, batch_size, validation_split, seed)

    # Section 4 (DEVELOPMENT_GUIDE.md): Check for class imbalance and compute class weights.
    processed_dir = Path(processed_dir)
    counts = {name: len(list((processed_dir / name).iterdir())) for name in CLASS_NAMES}
    for name, count in counts.items():
        print(f"  {name}: {count} images")
    total_images = sum(counts.values())
    class_weight = {
        i: total_images / (len(counts) * count)
        for i, (name, count) in enumerate(counts.items())
    }
    print(f"Class weights: {class_weight}")

    # build model
    model = build_model()

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Section 4 (DEVELOPMENT_GUIDE.md): EarlyStopping + ModelCheckpoint — save best val_accuracy epoch.
    best_model_path = artifacts_dir / "drowsiness_cnn.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # trains for epochs, applying class weights and callbacks
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # gets validation loss and accuracy
    eval_metrics = model.evaluate(val_ds, verbose=0)

    # Section 4 (DEVELOPMENT_GUIDE.md): Confusion matrix and classification report.
    all_labels, all_preds = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        all_labels.extend(labels.numpy())
        all_preds.extend((preds.squeeze() >= 0.5).astype(int))
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    labels_path = artifacts_dir / "labels.json"
    metrics_path = artifacts_dir / "metrics.json"

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"class_names": CLASS_NAMES}, f, indent=2)

    history_json = {k: [float(v) for v in values] for k, values in history.history.items()}
    metrics_payload = {
        "eval_loss": float(eval_metrics[0]),
        "eval_accuracy": float(eval_metrics[1]),
        "history": history_json,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Saved model: {best_model_path}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved metrics: {metrics_path}")
    return best_model_path, labels_path


def _discover_datasets(base: Path) -> list[Path]:
    """Return sorted list of processed dataset directories under *base*."""
    base = Path(base)
    if not base.exists():
        return []
    candidates = sorted(
        [d for d in base.iterdir() if d.is_dir() and (d / CLASS_NAMES[0]).exists()],
        key=lambda p: p.name,
    )
    return candidates


def _prompt_dataset_selection(base: Path) -> Path:
    """List available preprocessed datasets and let the user pick one."""
    datasets = _discover_datasets(base)
    if not datasets:
        raise FileNotFoundError(
            f"No preprocessed datasets found under {base}. "
            "Run preprocess.py first to create one."
        )
    if len(datasets) == 1:
        print(f"Using the only available dataset: {datasets[0].name}")
        return datasets[0]

    print("\nAvailable preprocessed datasets:")
    for i, ds in enumerate(datasets, start=1):
        # Show image counts per class for quick reference
        counts = {
            name: len(list((ds / name).iterdir())) for name in CLASS_NAMES if (ds / name).exists()
        }
        summary = ", ".join(f"{name}: {c}" for name, c in counts.items())
        print(f"  [{i}] {ds.name}  ({summary})")

    while True:
        choice = input(f"\nSelect dataset [1-{len(datasets)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(datasets):
            selected = datasets[int(choice) - 1]
            print(f"Selected: {selected.name}\n")
            return selected
        print("Invalid selection, try again.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train drowsiness CNN from processed dataset.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Path to a specific preprocessed dataset folder. If omitted you will be prompted to choose.",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.processed_dir:
        processed_dir = args.processed_dir
    else:
        processed_dir = _prompt_dataset_selection(PROCESSED_BASE_DIR)

    train_and_save(
        processed_dir=processed_dir,
        artifacts_dir=args.artifacts_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

