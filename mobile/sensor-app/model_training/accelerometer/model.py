"""Small 1D-CNN over (window_samples, channels) inputs."""
from __future__ import annotations

import tensorflow as tf

from config import WINDOW_SAMPLES, CHANNEL_NAMES


def build_model(
    window_samples: int = WINDOW_SAMPLES,
    channels: int = len(CHANNEL_NAMES),
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(window_samples, channels), name="accel_window")

    x = tf.keras.layers.Conv1D(32, kernel_size=7, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(64, kernel_size=5, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="accel_drowsiness_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model
