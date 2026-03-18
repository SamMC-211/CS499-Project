import argparse
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from config import (
    CASCADE_PATH,
    CLASS_NAMES,
    FACE_LANDMARKER_TASK_PATH,
    IMG_SIZE,
    PROCESSED_BASE_DIR,
    RAW_IMAGES_DIR,
)

# Section 3 (DEVELOPMENT_GUIDE.md): MLKit contour index subset.
# Training now draws ONLY the landmark indices that react-native-vision-camera-face-detector
# returns on mobile, so training preprocessing and mobile inference stay in sync.
#
# The mobile app (drowsiness.tsx) iterates over these contour keys:
#   FACE, LEFT_EYEBROW_TOP, LEFT_EYEBROW_BOTTOM, RIGHT_EYEBROW_TOP, RIGHT_EYEBROW_BOTTOM,
#   LEFT_EYE, RIGHT_EYE, UPPER_LIP_TOP, UPPER_LIP_BOTTOM, LOWER_LIP_TOP, LOWER_LIP_BOTTOM,
#   NOSE_BRIDGE, NOSE_BOTTOM, LEFT_CHEEK, RIGHT_CHEEK
#
# Each set below maps a contour key to the approximate MediaPipe Face Mesh 468 indices.
# Eye indices get radius=2 (matching the mobile EYE_CONTOUR_RADIUS constant).
# All other indices get radius=1 (matching LANDMARK_DOT_RADIUS).

# --- Eyes (radius=2 on mobile) ---
MLKIT_LEFT_EYE_IDXS = {
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
}
MLKIT_RIGHT_EYE_IDXS = {
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
}
MLKIT_EYE_IDXS = MLKIT_LEFT_EYE_IDXS | MLKIT_RIGHT_EYE_IDXS

# --- Face oval (radius=1 on mobile) ---
MLKIT_FACE_OVAL_IDXS = {
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
}

# --- Eyebrows (radius=1 on mobile) ---
MLKIT_LEFT_EYEBROW_TOP_IDXS = {276, 283, 282, 295, 285}
MLKIT_LEFT_EYEBROW_BOTTOM_IDXS = {300, 293, 334, 296, 336}
MLKIT_RIGHT_EYEBROW_TOP_IDXS = {46, 53, 52, 65, 55}
MLKIT_RIGHT_EYEBROW_BOTTOM_IDXS = {70, 63, 105, 66, 107}

# --- Lips / mouth (radius=1 on mobile) ---
MLKIT_UPPER_LIP_TOP_IDXS = {61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291}
MLKIT_UPPER_LIP_BOTTOM_IDXS = {78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308}
MLKIT_LOWER_LIP_TOP_IDXS = {78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308}
MLKIT_LOWER_LIP_BOTTOM_IDXS = {61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291}

# --- Nose (radius=1 on mobile) ---
MLKIT_NOSE_BRIDGE_IDXS = {168, 6, 197, 195, 5}
MLKIT_NOSE_BOTTOM_IDXS = {48, 115, 220, 45, 4, 275, 440, 344, 278}

# --- Cheeks (radius=1 on mobile) ---
MLKIT_LEFT_CHEEK_IDXS = {330}
MLKIT_RIGHT_CHEEK_IDXS = {101}

# Combined set of ALL contour indices drawn on mobile
MLKIT_APPROX_INDICES = (
    MLKIT_EYE_IDXS
    | MLKIT_FACE_OVAL_IDXS
    | MLKIT_LEFT_EYEBROW_TOP_IDXS
    | MLKIT_LEFT_EYEBROW_BOTTOM_IDXS
    | MLKIT_RIGHT_EYEBROW_TOP_IDXS
    | MLKIT_RIGHT_EYEBROW_BOTTOM_IDXS
    | MLKIT_UPPER_LIP_TOP_IDXS
    | MLKIT_UPPER_LIP_BOTTOM_IDXS
    | MLKIT_LOWER_LIP_TOP_IDXS
    | MLKIT_LOWER_LIP_BOTTOM_IDXS
    | MLKIT_NOSE_BRIDGE_IDXS
    | MLKIT_NOSE_BOTTOM_IDXS
    | MLKIT_LEFT_CHEEK_IDXS
    | MLKIT_RIGHT_CHEEK_IDXS
)


def _normalized_to_pixel_coordinates(
    normalized_x: float,
    normalized_y: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int] | None:
    if normalized_x < 0 or normalized_x > 1 or normalized_y < 0 or normalized_y > 1:
        return None
    x_px = min(int(normalized_x * image_width), image_width - 1)
    y_px = min(int(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def draw_and_save_face_mesh(
    image_bgr: np.ndarray,
    face_landmarks,
    output_path: Path,
) -> np.ndarray:
    image_drawing_tool = image_bgr.copy()
    img_h, img_w, _ = image_bgr.shape

    # Draw only MLKit-matching contour landmark indices (Section 3 of DEVELOPMENT_GUIDE.md).
    # Skipping indices not in MLKIT_APPROX_INDICES keeps training images visually identical
    # to the dot-stamped buffers the mobile app sends to the model at inference time.
    for i, landmark in enumerate(face_landmarks):
        if i not in MLKIT_APPROX_INDICES:
            continue
        point = _normalized_to_pixel_coordinates(landmark.x, landmark.y, img_w, img_h)
        if point is None:
            continue
        radius = 2 if i in MLKIT_EYE_IDXS else 1
        cv2.circle(image_drawing_tool, point, radius, (255, 255, 255), -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_drawing_tool)
    return cv2.resize(image_drawing_tool, (IMG_SIZE, IMG_SIZE))

# Create tasks director, loads .task model via base options
def _create_face_landmarker(task_model_path: Path):
    base_options = mp.tasks.BaseOptions(model_asset_path=str(task_model_path))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def preprocess_dataset(
    # arg_name: TypeHint = Default_Value
    raw_images_dir: Path = RAW_IMAGES_DIR,
    cascade_path: Path = CASCADE_PATH,
    output_dir: Path = PROCESSED_BASE_DIR,
    task_model_path: Path = FACE_LANDMARKER_TASK_PATH,
) -> int: # return integer
    raw_images_dir = Path(raw_images_dir)
    cascade_path = Path(cascade_path)
    output_dir = Path(output_dir)
    task_model_path = Path(task_model_path)

    if not raw_images_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {raw_images_dir}")
    if not cascade_path.exists():
        raise FileNotFoundError(f"Haar cascade not found: {cascade_path}")
    if not task_model_path.exists():
        raise FileNotFoundError(
            f"Face Landmarker task model not found: {task_model_path}. "
            "Download 'face_landmarker.task' and pass --task-model or place it at the default path."
        )

    # Load OpenCVs pre-trained face detector
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load cascade classifier: {cascade_path}")

    total_written = 0
    landmarker = _create_face_landmarker(task_model_path) # initialize tasks model
    try:
        for class_name in CLASS_NAMES: #for both input image folders
            class_input_dir = raw_images_dir / class_name # path for either folder of images
            if not class_input_dir.exists():
                raise FileNotFoundError(f"Missing category folder: {class_input_dir}")

            class_output_dir = output_dir / class_name
            class_output_dir.mkdir(parents=True, exist_ok=True) #create output folder if needed

            index = 1
            for image_path in class_input_dir.iterdir(): #for every image
                if not image_path.is_file(): #check if exists
                    continue

                image_array = cv2.imread(str(image_path), cv2.IMREAD_COLOR) # load image
                if image_array is None: #skip if corrupt
                    continue

                gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) #convert to greyscale (for Haar cascade)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5) # list of face bounding boxes
                if len(faces) == 0: # skip if no face detected
                    continue

                x, y, w, h = faces[0] # take first detected face
                roi_color = image_array[y : y + h, x : x + w] # crop to region of interest
                if roi_color.size == 0: # skip if no region of interest
                    continue

                roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB) # convert to rgb (for Mediapipe)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb) # build mp.Image
                result = landmarker.detect(mp_image) # call landmarker.detect()
                if not result.face_landmarks: # skip if no lardmarks
                    continue

                output_path = class_output_dir / f"{index}.jpg" # create image output filename using index
                draw_and_save_face_mesh(roi_color, result.face_landmarks[0], output_path) # pass image to function, landmarks will be drawn 
                index += 1
                total_written += 1
    finally:
        landmarker.close()

    return total_written


def _make_dated_output_dir(base: Path) -> Path:
    """Create a date-stamped subfolder under the processed base directory."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dated = base / f"processed_{stamp}"
    dated.mkdir(parents=True, exist_ok=True)
    return dated


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess drowsiness dataset images.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_IMAGES_DIR)
    parser.add_argument("--cascade", type=Path, default=CASCADE_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit output directory. If omitted a date-stamped folder is created under processed/.",
    )
    parser.add_argument("--task-model", type=Path, default=FACE_LANDMARKER_TASK_PATH)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else _make_dated_output_dir(PROCESSED_BASE_DIR)

    total = preprocess_dataset(
        raw_images_dir=args.raw_dir,
        cascade_path=args.cascade,
        output_dir=output_dir,
        task_model_path=args.task_model,
    )
    print(f"Preprocessing complete. Wrote {total} images to {output_dir}")


if __name__ == "__main__":
    main()

