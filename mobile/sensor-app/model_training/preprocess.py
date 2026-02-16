import argparse
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
    PROCESSED_DIR,
    RAW_IMAGES_DIR,
)

# Index groups for optional eye landmark highlighting.
chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
all_chosen_idxs = set(chosen_left_eye_idxs + chosen_right_eye_idxs)


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

    # Draw all landmarks as mesh points.
    for i, landmark in enumerate(face_landmarks):
        point = _normalized_to_pixel_coordinates(landmark.x, landmark.y, img_w, img_h)
        if point is None:
            continue
        radius = 2 if i in all_chosen_idxs else 1
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
    output_dir: Path = PROCESSED_DIR,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess drowsiness dataset images.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_IMAGES_DIR)
    parser.add_argument("--cascade", type=Path, default=CASCADE_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--task-model", type=Path, default=FACE_LANDMARKER_TASK_PATH)
    args = parser.parse_args()

    total = preprocess_dataset(
        raw_images_dir=args.raw_dir,
        cascade_path=args.cascade,
        output_dir=args.output_dir,
        task_model_path=args.task_model,
    )
    print(f"Preprocessing complete. Wrote {total} images to {args.output_dir}")


if __name__ == "__main__":
    main()

