from pathlib import Path

# UAH-DRIVESET RAW_ACCELEROMETERS.txt column layout (1-indexed in dataset docs):
#   1: timestamp (s)
#   2: status flag (always 0 in our copies)
#   3-5:   raw accelerometer X, Y, Z (g)
#   6-8:   KF-filtered vehicle-frame accelerations X, Y, Z (g) -- gravity removed
#   9-11:  roll, pitch, yaw estimates (rad)
#
# We train on columns 6-8 (indices 5,6,7) -- the gravity-removed vehicle-frame
# acceleration. This is the closest analogue to what the mobile app will produce
# after the calibration step (subtract gravity, rotate device-frame into
# vehicle-frame). Using these channels keeps train/inference frames consistent.
CHANNEL_INDICES = (5, 6, 7)
CHANNEL_NAMES = ("accel_x", "accel_y", "accel_z")

# Dataset is sampled at ~10 Hz. A 30 s window gives the model enough context to
# pick up sustained driving patterns (slow drift, weaving) without making the
# per-window count too small.
SAMPLE_RATE_HZ = 10
WINDOW_SECONDS = 30
WINDOW_SAMPLES = SAMPLE_RATE_HZ * WINDOW_SECONDS  # 300
WINDOW_STRIDE_SAMPLES = WINDOW_SAMPLES // 2       # 50% overlap

# class_names[0] is the "drowsy" class -- sigmoid output near 0 means drowsy,
# near 1 means normal. Matches the camera pipeline's convention of putting the
# alarming class at index 0.
CLASS_NAMES = ("Drowsy", "Normal")

# Behaviors in the dataset folder names. AGGRESSIVE is excluded per the
# binary-classification scope decided for this first iteration.
LABEL_KEYWORDS = {
    "DROWSY": 0,
    "NORMAL": 1,
}

BASE_DIR = Path(__file__).resolve().parent
INPUT_ROOT = BASE_DIR / "input"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
