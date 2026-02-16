from pathlib import Path

IMG_SIZE = 145 # Default for preprocessing and training
CLASS_NAMES = ["Fatigue Subjects", "Active Subjects"]

BASE_DIR = Path(__file__).resolve().parent # Anchor paths to script folder so code can be ran from wherever
INPUT_ROOT = BASE_DIR / "input" 
RAW_IMAGES_DIR = INPUT_ROOT / "drowsiness-prediction-dataset" / "0 FaceImages"
CASCADE_PATH = INPUT_ROOT / "prediction-images" / "haarcascade_frontalface_default.xml"
FACE_LANDMARKER_TASK_PATH = BASE_DIR / "input" / "face_landmarker.task"

PROCESSED_DIR = BASE_DIR / "processed"
MODELS_DIR = BASE_DIR / "artifacts"
