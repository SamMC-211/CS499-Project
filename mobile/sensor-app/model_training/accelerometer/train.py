"""Leave-one-driver-out CV + final model training."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from config import ARTIFACTS_DIR, CHANNEL_NAMES, CLASS_NAMES, WINDOW_SAMPLES
from load_dataset import Session, load_sessions, summarize
from model import build_model
from preprocess import apply_norm, compute_norm_stats, window_signal


def _build_windows(
    sessions: list[Session],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (windows, labels, driver_ids) stacked across sessions."""
    X, y, drv = [], [], []
    for s in sessions:
        w = window_signal(s.signal)
        if w.shape[0] == 0:
            continue
        X.append(w)
        y.append(np.full(w.shape[0], s.label, dtype=np.int32))
        drv.append(np.array([s.driver] * w.shape[0]))
    return np.concatenate(X), np.concatenate(y), np.concatenate(drv)


def _class_weights(labels: np.ndarray) -> dict[int, float]:
    # Inverse-frequency weighting -- drowsy class is the minority here.
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    total = counts.sum()
    return {i: float(total / (2 * c)) if c > 0 else 1.0 for i, c in enumerate(counts)}


def _train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict:
    tf.keras.utils.set_random_seed(seed)
    mean, std = compute_norm_stats(X_train)
    Xtr = apply_norm(X_train, mean, std)
    Xte = apply_norm(X_test, mean, std)

    model = build_model()
    cw = _class_weights(y_train)
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True, verbose=0,
    )
    history = model.fit(
        Xtr,
        y_train,
        validation_data=(Xte, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=[es],
        verbose=0,
    )
    probs = model.predict(Xte, verbose=0).ravel()
    preds = (probs >= 0.5).astype(np.int32)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else float("nan")
    cm = confusion_matrix(y_test, preds, labels=[0, 1]).tolist()
    return {
        "accuracy": float(acc),
        "auc": float(auc),
        "confusion_matrix": cm,
        "n_train_windows": int(len(y_train)),
        "n_test_windows": int(len(y_test)),
        "epochs_ran": len(history.history["loss"]),
    }


def leave_one_driver_out(
    sessions: list[Session],
    epochs: int = 30,
    batch_size: int = 32,
    seed: int = 42,
) -> dict:
    drivers = sorted({s.driver for s in sessions})
    fold_results: dict[str, dict] = {}
    for held_out in drivers:
        train_sessions = [s for s in sessions if s.driver != held_out]
        test_sessions = [s for s in sessions if s.driver == held_out]
        X_train, y_train, _ = _build_windows(train_sessions)
        X_test, y_test, _ = _build_windows(test_sessions)
        if X_test.shape[0] == 0:
            print(f"[{held_out}] no test windows, skipping")
            continue
        if len(set(y_test)) < 2:
            # Driver has only one class represented -- still report accuracy
            # but AUC will be NaN.
            print(f"[{held_out}] warning: test set has only one class")
        print(
            f"[{held_out}] train_windows={X_train.shape[0]} "
            f"test_windows={X_test.shape[0]}"
        )
        result = _train_fold(X_train, y_train, X_test, y_test, epochs, batch_size, seed)
        fold_results[held_out] = result
        print(
            f"[{held_out}] acc={result['accuracy']:.3f} "
            f"auc={result['auc']:.3f} cm={result['confusion_matrix']}"
        )

    accs = [r["accuracy"] for r in fold_results.values()]
    aucs = [r["auc"] for r in fold_results.values() if not np.isnan(r["auc"])]
    summary = {
        "mean_accuracy": float(np.mean(accs)) if accs else float("nan"),
        "std_accuracy": float(np.std(accs)) if accs else float("nan"),
        "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "std_auc": float(np.std(aucs)) if aucs else float("nan"),
        "n_folds": len(fold_results),
    }
    return {"folds": fold_results, "summary": summary}


def train_final(
    sessions: list[Session],
    artifacts_dir: Path = ARTIFACTS_DIR,
    epochs: int = 30,
    batch_size: int = 32,
    seed: int = 42,
) -> Path:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X, y, _ = _build_windows(sessions)
    tf.keras.utils.set_random_seed(seed)
    mean, std = compute_norm_stats(X)
    Xn = apply_norm(X, mean, std)

    model = build_model()
    cw = _class_weights(y)
    history = model.fit(
        Xn, y,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", mode="max", patience=5,
                restore_best_weights=True, verbose=1,
            )
        ],
        verbose=1,
    )

    model_path = artifacts_dir / "accel_drowsiness_cnn.keras"
    model.save(model_path)

    # Save normalization stats so the mobile app can apply the same z-scoring.
    norm_path = artifacts_dir / "normalization.json"
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "channel_names": list(CHANNEL_NAMES),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "window_samples": int(WINDOW_SAMPLES),
            },
            f,
            indent=2,
        )

    labels_path = artifacts_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"class_names": list(CLASS_NAMES)}, f, indent=2)

    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train accelerometer drowsiness CNN.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip leave-one-driver-out CV; only train the final model.",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Run CV only; don't train the final all-data model.",
    )
    args = parser.parse_args()

    sessions = load_sessions()
    summarize(sessions)

    metrics_payload: dict = {}

    if not args.skip_cv:
        print("\n=== Leave-one-driver-out CV ===")
        cv = leave_one_driver_out(
            sessions, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        )
        print("\nSummary:", cv["summary"])
        metrics_payload["cv"] = cv

    if not args.skip_final:
        print("\n=== Final model (all drivers) ===")
        model_path = train_final(
            sessions,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        print(f"Saved: {model_path}")

    if metrics_payload:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = ARTIFACTS_DIR / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
