# Drowsiness Model Pipeline

This folder is organized as a script pipeline instead of a single notebook-style file.

## Files

- `config.py`: Shared paths and constants.
- `preprocess.py`: Detects faces, draws landmarks, writes class-separated processed images.
- `train.py`: Loads processed images, trains CNN, saves model + labels + metrics.
- `export_tflite.py`: Converts trained Keras model to TensorFlow Lite.
- `train_model.py`: CLI runner for `preprocess`, `train`, `export`, or `all`.

## Dataset Layout

Expected input structure:

- `input/drowsiness-prediction-dataset/0 FaceImages/Fatigue Subjects/`
- `input/drowsiness-prediction-dataset/0 FaceImages/Active Subjects/`
- `input/drowsiness-prediction-dataset/prediction-images/haarcascade_frontalface_default.xml`
- `input/face_landmarker.task` (MediaPipe Tasks model file)

## Run Steps

From `mobile/sensor-app/model_training`:

```powershell
python train_model.py preprocess
python train_model.py train --epochs 20
python train_model.py export
```

Or run everything:

```powershell
python train_model.py all --epochs 20
```

## Outputs

- Processed images: `processed/Fatigue Subjects/`, `processed/Active Subjects/`
- Trained model: `artifacts/drowsiness_cnn.keras`
- Labels: `artifacts/labels.json`
- Metrics/history: `artifacts/metrics.json`
- TFLite model: `artifacts/drowsiness_cnn.tflite`

## Notes

- `preprocess.py` now uses MediaPipe Tasks Face Landmarker (not `mp.solutions`).
- If your task model is in a different location, run:

```powershell
python preprocess.py --task-model "<path-to-face_landmarker.task>"
```

## Expo-specific note
For local inference in Expo, you should use the .tflite model (not .h5/.keras). You’ll also need your app preprocessing to match training input exactly: RGB, 145x145, scaled to [0,1].

Next step I can do:

Add an Expo-side inference helper that loads drowsiness_cnn.tflite, preprocesses camera frames, and maps output to Fatigue Subjects / Active Subjects.

Possibly consider landmark bbased training instead of givning the  whole picture to keras to make model more mobile efficient!
