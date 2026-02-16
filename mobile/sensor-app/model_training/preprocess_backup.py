import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from config import CASCADE_PATH, CLASS_NAMES, IMG_SIZE, PROCESSED_DIR, RAW_IMAGES_DIR

# Setup mediapipe landmark indicies 
mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalize_to_pixel_coordinates

all_left_eye_idxs = set(np.ravel(list(mp_facemesh.FACEMESH_LEFT_EYE)))
all_right_eye_idxs = set(np.ravel(list(mp_facemesh.FACEMESH_RIGHT_EYE)))
all_idxs = all_left_eye_idxs.union(all_right_eye_idxs) # all of the eye landmark indices
chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs # all of the chosen eye landmark indices

# draws face mesh landmarks on cropped face images, saves resulting image as index.jpg, resize to 145 by 145
def draw_and_save_face_mesh(
    image_bgr: np.ndarray,
    face_landmarks,
    output_path: Path,
) -> np.ndarray: # outputs an numpy array
    # 3 image copies, for each drawing
    image_drawing_tool = image_bgr.copy() # full face mesh
    # 2 Other drawings
    # image_eye_lmks = image_bgr.copy() # all eye landmarks
    # img_eye_lmks_chosen = image_bgr.copy() # only selected eye points

    # Get width, height, and color channels to convert landmark into pixel coordinates
    img_h, img_w, _ = image_bgr.shape

    # defines how lines are drawn (for Mediapipe)
    connections_drawing_spec = mp_drawing.DrawingSpec(
        thickness=1,
        circle_radius=2,
        color=(255, 255, 255),
    )

    # draw full face mesh onto image_drawing_tool
    mp_drawing.draw_landmarks(
        image=image_drawing_tool,
        landmark_list=face_landmarks,
        connections=mp_facemesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=connections_drawing_spec,
    )

    # Draw landmakrs for 2 other drawings (not saved, so commented out)
    # landmark = raw landmark list of coords (z , y ,z)
    # landmarks = face_landmarks.landmark
    # # Iterate through all mesh indicies
    # for landmark_idx, landmark in enumerate(landmarks):
    #     # if landmark is not in chosen, convert normalized coords to pixel coords (Mediapipe: 0.5 -> CV2: 72.5 px)
    #     if landmark_idx in all_idxs:
    #         pred_cord = denormalize_coordinates(landmark.x, landmark.y, img_w, img_h)
    #         if pred_cord is not None:
    #             cv2.circle(image_eye_lmks, pred_cord, 3, (255, 255, 255), -1) # draw landmark onto image_eye_lmks

    #     # same for chosen landmarks, to img_eye_lmks_chosen
    #     if landmark_idx in all_chosen_idxs:
    #         pred_cord = denormalize_coordinates(landmark.x, landmark.y, img_w, img_h)
    #         if pred_cord is not None:
    #             cv2.circle(img_eye_lmks_chosen, pred_cord, 3, (255, 255, 255), -1)

    output_path.parent.mkdir(parents=True, exist_ok=True) #create output dir if needed
    cv2.imwrite(str(output_path), image_drawing_tool) #write full face mesh drawn image to output folder
    return cv2.resize(image_drawing_tool, (IMG_SIZE, IMG_SIZE)) # return resized version of face mesh image

# (raw dataset images, the haarcascade xml for cv2, output folders for processed images)
def preprocess_dataset(
    # arg_name: TypeHint = Default_Value
    raw_images_dir: Path = RAW_IMAGES_DIR,
    cascade_path: Path = CASCADE_PATH,
    output_dir: Path = PROCESSED_DIR,
) -> int:  # return integer
    # if a string is passed as an arg, convert to path object
    raw_images_dir = Path(raw_images_dir)
    cascade_path = Path(cascade_path)
    output_dir = Path(output_dir)

    # Error check input data dirs exist
    if not raw_images_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {raw_images_dir}")
    if not cascade_path.exists():
        raise FileNotFoundError(f"Haar cascade not found: {cascade_path}")

    # Load OpenCVs pre-trained face detector
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load cascade classifier: {cascade_path}")

    total_written = 0 #track number of images written
    with mp_facemesh.FaceMesh( # create MediaPipe facemesh instance
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh: # alias instance
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
                results = face_mesh.process(roi_rgb) # Run MediaPipe face mesh on CV2 processed image, results = 468 face landmarks
                if not results.multi_face_landmarks: # if no landmarks skip image
                    continue

                output_path = class_output_dir / f"{index}.jpg" # create image output filename using index
                draw_and_save_face_mesh(roi_color, results.multi_face_landmarks[0], output_path) # pass image to function, landmarks will be drawn 
                index += 1
                total_written += 1

    return total_written


def main() -> None:
    # Parser stuff allows us to pass arguments for these paths when running the python script from the terminal
    parser = argparse.ArgumentParser(description="Preprocess drowsiness dataset images.") #create command line arg parser
    parser.add_argument("--raw-dir", type=Path, default=RAW_IMAGES_DIR)
    parser.add_argument("--cascade", type=Path, default=CASCADE_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    total = preprocess_dataset(args.raw_dir, args.cascade, args.output_dir) 
    print(f"Preprocessing complete. Wrote {total} images to {args.output_dir}")

# If file is ran directly, run from main. Otherwise main() does not run automatically when preprocess.py is imported
if __name__ == "__main__":
    main()

