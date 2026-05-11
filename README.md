![Vigilare](/assets/Project%20Banner.png)

# CS499-Project | Vigilare 

This project is to explore how to build a mobile application using **Expo with React Native** that can collect sensor data and apply machine learning–based activity recognition techniques. The core idea is to leverage mobile device sensors to capture user behavioral signals and use machine learning models to classify certain states or activities. This 

## Project Structure

```
CS499-Project/
├── mobile/sensor-app/               # Main Expo/React Native app
│   │ 
│   ├── app/                          # Expo Router screens
│   │   ├── _layout.tsx               # Root stack navigator
│   │   ├── index.tsx                 # Home screen (navigation menu)
│   │   └── sensors/
│   │       ├── drowsiness.tsx        # Main drowsiness detection screen (782 lines)
│   │       ├── accelerometer.tsx     # Accelerometer demo
│   │       ├── camera.tsx            # Basic camera test
│   │       ├── debug-gallery.tsx     # Debug image gallery
│   │       └── accelerometer-drowsiness.tsx # Accelerometer-based drowsiness prediction screen
│   │ 
│   ├── components/                   # React Native Components
│   │ 
│   ├── assets/
│   │   ├── ml/
│   │   │   ├── drowsiness_cnn.tflite # TFLite model (5.8 MB)
│   │   │   └── labels.json           # Class labels ("Fatigue Subjects", "Active Subjects")
│   │   └── images/                   # Icons, splash screen, favicon
│   │
│   ├── model_training/               # Python ML training pipeline
│   │   │
│   │   ├── config.py                 # Training config (IMG_SIZE=145, class names)
│   │   ├── preprocess.py             # MediaPipe facial landmark preprocessing
│   │   ├── train.py                  # CNN model training
│   │   ├── export_tflite.py          # Keras -> TFLite conversion
│   │   ├── train_model.py            # CLI orchestrator
│   │   ├── artifacts/                # Model outputs (.tflite, .keras, metrics)
│   │   ├── input/                    # Training dataset + MediaPipe task file
│   │   ├── processed/               # Preprocessed training images
│   │   ├── debugging/                # Debug utilities
│   │   │
│   │   └── accelerometer/            # Accelerometer drowsiness test pipeline (UAH-DRIVESET)
│   │       │
│   │       ├── config.py             # Window size, sample rate, channel indices, paths
│   │       ├── load_dataset.py       # Parse input/ session folders into Session records
│   │       ├── preprocess.py         # Windowing + per-channel z-score normalization
│   │       ├── model.py              # 1D-CNN definition (Conv1D x3 -> GAP -> Dense -> sigmoid)
│   │       ├── train.py              # Leave-one-driver-out CV + final all-data training
│   │       ├── export_tflite.py      # Keras -> TFLite conversion
│   │       ├── train_model.py        # CLI orchestrator (train/export/all)
│   │       ├── input/                # UAH RAW_ACCELEROMETERS.txt files (D1..D6, DROWSY/NORMAL only)
│   │       └── artifacts/            # Trained model outputs (.keras, .tflite, normalization.json, metrics.json)
│   │
│   ├── android/                      # Native Android project (Expo Prebuild output)
│   ├── app.json                      # Expo configuration
│   ├── package.json                  # JS dependencies
│   ├── metro.config.js               # Metro bundler config (registers .tflite extension)
│   ├── babel.config.js               # Babel config (worklets plugin)
│   └── tsconfig.json                 # TypeScript config
│  
├── venv/                             # Python virtual environment (not in build)
├── ACCELEROMETER_PIPELINE.md         # Detailed implementation notes for the accelerometer pipeline
└── README.md
```

## Libraries & Dependencies

### JavaScript/NPM Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `expo` | ~54.0.32 | React Native framework / build system |
| `react` | 19.1.0 | UI library |
| `react-native` | 0.81.5 | Native runtime |
| **Camera & Vision** | | |
| `react-native-vision-camera` | ^4.7.3 | Camera access with frame processor support |
| `vision-camera-resize-plugin` | ^3.2.0 | Fast frame resizing for ML input |
| `react-native-vision-camera-face-detector` | ^1.10.1 | Google MLKit face + landmark detection |
| `expo-camera` | ~17.0.10 | Basic camera API (used in camera test screen) |
| **ML / Inference** | | |
| `react-native-fast-tflite` | ^2.0.0 | TensorFlow Lite inference engine |
| **Worklets / Performance** | | |
| `react-native-worklets-core` | ^1.6.3 | Shared worklet runtime for frame processors |
| `react-native-worklets` | 0.5.1 | Worklet scheduling utilities |

### Python Dependencies (model training pipeline)

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | 2.20.0 | Model training & export |
| `mediapipe` | 0.10.32 | Facial landmark extraction for preprocessing |
| `opencv-python` | 4.13.0.92 | Image I/O and processing |
| `numpy` | 2.4.2 | Numerical operations |
| `pandas` | 2.3.2 | Data manipulation |
| `matplotlib` | 3.10.6 | Training visualization/plots |
| `scikit-learn` | 1.8.0 | Metrics, train/test splitting |



# Project Meeting Logs

## Week 3 | 2/9/26 | Aggregating Possible Training Datasets

This week was spent looking for test data to be able to train the model on, as well as figuring out what the tech stack was going to look like for building this application. Below are the collections of possible datasets I could use to train the model for this project, as well as a sample tech stack for the project. The tech stack is subject to change once I start developing the project.


**Dataset Candidates** 
- [Driver Drowsiness](https://huggingface.co/datasets/ckcl/driver-safety-dataset) (Labled Images)
- [Lateral Acceleration](https://github.com/commaai/comma-steering-control?tab=readme-ov-file) (openpilot driver assistance system)
- [Drowsy Detection](https://www.kaggle.com/datasets/yasharjebraeily/drowsy-detection-dataset) (Greyscale face images)
- [Drowsiness Detection System](https://www.kaggle.com/code/mohamedkhaledelsafty/drowsiness-detection-system/n) (Python processing imports given, Colored full face)
- [Driver drowsiness using keras](https://www.kaggle.com/code/adinishad/driver-drowsiness-using-keras/notebook) (Eyes closed/Yawn)

**Proposed Tech Stack**
- Frontend/Mobile
    - Expo - React Native
        - Expo-Camera
        - Expo-Accelerometer
- Local Model Processing
    - TensorFlowLite (Lightweight model exported for mobile)
    - Mediapipe/tasks-vision (For reading face data for mobile build)
- Model Training
    - Python 3.12
        - numpy
        Mat plotlib
    - MediaPipe 0.10.32 > 0.10.14 (For mp.solutions)
    - TensorFlow (Training -> TFLite for Local running)
    - OpenCV-Python (Useful for reading video and images to train model)

**Code References**
- [Available MediaPipe facial detection models](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector#models)
- [MediaPipe Sample Code](https://github.com/google-ai-edge/mediapipe-samples)
- [MediaPipe Tasks Python API](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_detector/python/face_detector.ipynb#scrollTo=L_cQX8dWu4Dv)
- [MediaPipe Tasks Android Native Demo](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/face_detector/android)

## Week 4 | 2/16/26 | Adapting Model Training Code

This week I spent some time attempting to get a model created using the example code and training data from [Drowsiness Detection System](https://www.kaggle.com/code/mohamedkhaledelsafty/drowsiness-detection-system/n). It took me a while and some adjustments were made as I went. 

**The adjustments made are as follows:**
- Adapted example code into python files for each step of the model creation with a main file that allows you to run the entire model training process using flags. 
- **Suite of Python scripts for training model:**
    - **train_model.py** : Main file for streamlining entire training process, with flags to control flow
    - **Proprocess.py** : Preprocess raw images by stamping facial landmarks onto images and creating a processed dataset. 
    - **Train.py** : Uses preprocessed image set to train model running several training cycles/ 
    - **Export.py** : Exports the trained model to a tflite format tailored to run on a mobile device.  
    - **Config.py** : Config setting for the model training files and process. 
- Refactored MediaPipe code to use version 0.10.32 instead of 0.8.11 which is what the project used. (This was due to a dependency conflict between TensorFlow and MediaPipe) 
    - Instead of MediaPipe.Solutions I am now using MediaPipe.tasks, solutions was outdated
    - model_training/input/ includes "face_landmarker.task"
- Eliminated some redundant code that was drawing eye landmarks on unsaved image copies
- Ran Preprocessing, Training, and Export processes on my computer 
    - Outputs (in /mobile/sensor-app/model_training/artifacts/): 
        - drowsiness_cnn.keras
        - drowsiness_cnn.tflite
        - labels.json
        - metrics.json

Additionally there were some changes to the tech stack as I progressed within the project

**Current Tech Stack**
- Frontend/Mobile
    - Expo - React Native
        - Expo-Accelerometer
        - `React-native-vision-camera` (camera + frame processors)
        - `Vision-camera-resize-plugin` (fast frame resize + tensor input formatting)
- Local Model Processing
    - `React-native-fast-tflite` (native TFLite runtime)
    - Mediapipe/tasks-vision
- Model Training
    - Python 3.12
    - numpy 2.4.2
    - MediaPipe 0.10.32
        - `Tasks Face Landmarker`
    - TensorFlow 2.20.0
    - OpenCV-Python 4.13.0.92

## Week 5 | 2/23/26 | Figuring out Preprocessing on Mobile

I spent this week familiarizing myself with and troubleshooting my implementation of the preprocessing steps needed before I start feeding faces to my model. This was quite difficult as I was running into issues with mapping points from the react-native-vision-camera-face-detector. Heres a list of some of the things I accomplished during this week.
- **Implemented**
    - frameProcessor
    - mapLandmarkToCrop
    - stampDot
- Troubleshot the process of mapping contour points from the face-detector to the user camera view
    - Scaling conversion issues
    - View bounds issue


## Week 6 | 3/2/26 | Troubleshooting Frame processing on Mobile

This week I worked on getting the model up and running on mobile, I toiled a lot over this issue and while I didn't gain much ground in terms of app side development I did realize some changes to my approach I may need to make for this mobile application. Specificially addressing the disconnect between the preprocessing and training steps for the model vs the capabilities of mobile. 
- Familiarized myself with react-native-fast-tflite
- Create a Python script to render images from the Float32Array format that I am preprocessing frame inputs into for debugging
- Spent time troubleshooting the image stamping of inputs for the model and mapping points from the face detector to the frame
- Began feeding preprocessed images to my model and created UI to display driver state prediction
- Build Issues, had to set minSdkVersion to 26 in 'android/app/build.gradle'

## Week 7 | 3/9/26 | NA

## Week 8 | 3/16/26 | Improving Model Accuracy 

These past two weeks I've spent attempting to retrain the model to be accurate on mobile.

- Retrained model with MLKit landmarks, tested...
    - Was missing a few landmark groups, model was less accurate than before
    - Fixed preprocessing to include missing landmark groups and timestamp processed folders
- Retrained model after including missing MLKit landmark groups

**Model Versions (In Artifacts)**
- Drowsiness (Base model using MediaPipe facial landmarking)   
    - Most accurate so far, fluctuating reading
- Drowsiness 1.0 (Model using MLKit facial landmarking)
    - Missing landmarks innacurate on mobile
- Drowsiness 2.0 (Model using MLKit including previously missing landmark groups)
    - Updated preprocessing to include all landmark groups
    - Still highly innacurate on mobile
- Created serve.py to download apk remotely after building

**Example image of what is being fed to the model on mobile: **

![Rotated Image](/assets/Mobile_Inputs%20(Rotated%20image).png)

Not sure if the image being rotated when being fed to the model is messing with the accuracy but I will have to play around a little more with it.

## Week 9 | 3/23/26 | More Model Fine Tunring

This week I worked on fixing model accuracy on mobile and adding prediction smoothing, and correcting issues with how the model was being fed images on mobile. While correcting the rotation and stamping issue from the previous week I ran into an issue that I was stumped on for a while. Eventually I figured out what was causing the images fed to the model on mobile to be completely black.

- **Fixed input value range mismatch** — `vision-camera-resize-plugin` returns float32 in [0, 1] but the model's Rescaling(1/255) layer expects [0, 255]. Added a `* 255.0` scaling step after rotation, (`lines 148-158 in drowsiness.tsx`) before dot stamping. Before that the model was seeing a nearly black image.
    - **Example:**

![Black Image](/assets/Mobile_Inputs(Black%20image).png)

- **Added rolling average prediction smoothing** — The raw scores are now averaged over the last 10 frames (SMOOTHING_WINDOW_SIZE) before making a drowsy/active decision. Prevents single frame flickers from swapping the label back and forth
- **Added tunable drowsy threshold** — (DROWSY_THRESHOLD) constant replaces the hard 0.5 sigmoid midpoint. I adjusted this because the predictions were skewed towards Drowsy.
- **Score buffer clears on face loss** — when no face is detected the rolling average resets so stale scores dont carry over
- **Computer Specs** 
    - **Processor:** AMD AMD RYZEN 7 9800X3D
    - **RAM:** 32gb G.SKILL 2X D5 6000 C36 FX B
    - **Graphics Card:** ASUS PRIME RX9070XT
    - **MotherBoard:** 	ASUS TUF GAMING B650E-E WF
- **Dataset Size:** processed_2026-03-16_110640  (Fatigue Subjects: 3677, Active Subjects: 3861)
- **Training Time**
    - Preprocessing: 4:05
    - Model Training (20 Epochs): 4:44
- **Output from Last Training Step:**
![Output Image](/assets/Model%20Training%20Output.png)

**Notes for "week 9" (Actually week 10)**
- Look into using 1 additional sensor
- Look into using driving time to advise drowsiness alert/prediction
- Add driver alert functionality
- Add Screenshots to meeting document
- Work on cleaning up code 
- Attach the project APK 
- Create Notes for adapting project to IOS

## Week 10 | 3/30/26 | Looking into Additional Sensors

This week I looked into using an additional sensor for detecting drowsiness. I looked into what other similar applications used. The most common of which being accelerometer & gyroscope, GPS/Location services (to detect speed and route), with less common sensors used being the microphone, or compass. I was able to find a publicly available dataset that was also used to develop the "DriveSafe" app which was made to monitor, score, and alert drivers. In their case they used the rear camera to scan the road as well as several other sensors to monitor their drivers.

- **E. Romera, L.M. Bergasa and R. Arroyo, "Need Data for Driving Behavior Analysis? Presenting the Public UAH-DriveSet", IEEE International Conference on Intelligent Transportation Systems (ITSC), pp. 387-392, Rio de Janeiro (Brazil), November 2016**
- Link to their project and dataset [**here.**](http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/) 

I created the **accelerometer/** directory in the **model_training/** directory and copied only the RAW_ACCELEROMETERS.txt from DROWSY and NORMAL sessions. AGGRESSIVE sessions were intentionally excluded.
- 20151110175712-16km-D1-NORMAL1-SECONDARY/RAW_ACCELEROMETERS.txt
- 20151110180824-16km-D1-NORMAL2-SECONDARY/RAW_ACCELEROMETERS.txt
- 20151111123124-25km-D1-NORMAL-MOTORWAY/RAW_ACCELEROMETERS.txt
- 20151111132348-25km-D1-DROWSY-MOTORWAY/RAW_ACCELEROMETERS.txt
- 20151111135612-13km-D1-DROWSY-SECONDARY/RAW_ACCELEROMETERS.txt

The formatting for these raw data files is whitespace-separated floats, no header, 11 columns per row which are as follows:

| Column | Meaning |
|---|---|
| 1 | Timestamp (seconds from trip start) |
| 2 | Status flag (always 0 in our copies) |
| 3–5 | Raw accelerometer X, Y, Z (g) — device frame, gravity included |
| 6–8 | KF-filtered vehicle-frame accelerations X, Y, Z (g) — gravity removed |
| 9–11 | Roll, pitch, yaw estimates (rad) |

After figuring out how the data files are formatted I started working on creating another python pipeline for training this model based on the training process of the other model. All files live in `mobile/sensor-app/model_training/accelerometer/`. The
pipeline mirrors the existing camera pipeline's structure. I used Claude to help me adapt the existing code from the other model training pipeline to create one for this dataset. It helped with a lot of the data interpretation, and training approach since it would've taken me a long time to work through figuring out how to work through a new data format and normalize/interpret it.

- `config.py` Constants only. Defines:
    - CHANNEL_INDICES = (5, 6, 7) — 0-indexed selection of UAH columns 6–8.
    - SAMPLE_RATE_HZ = 10, WINDOW_SECONDS = 30 → WINDOW_SAMPLES = 300.
    - WINDOW_STRIDE_SAMPLES = 150 (50 % overlap).
    - CLASS_NAMES = ("Drowsy", "Normal").
    - LABEL_KEYWORDS = {"DROWSY": 0, "NORMAL": 1}.
- `load_dataset.py`
    - Session dataclass: driver, behavior, road, label, session_dir,
    signal (np.ndarray (N, 3) float32).
    - load_signal(path) — np.loadtxt + column selection.
    - load_sessions(root) — walks D{1..6}/{session-name}/RAW_ACCELEROMETERS.txt,
    parses folder name with SESSION_NAME_RE, filters by LABEL_KEYWORDS, returns
    a list of Session.
    - summarize(sessions) — prints per-driver class counts. Used by the CLI and
    for manual sanity checking.
- `preprocess.py` Critical detail: normalization stats are **computed from training windows only**.
    - window_signal(signal, window, stride) → (num_windows, window, channels).
    - compute_norm_stats(windows) → per-channel (mean, std), with a 1e-6
    floor on std to guard against zero-variance channels.
    - apply_norm(windows, mean, std) → z-scored windows in float32.
- `model.py` Small 1D-CNN:

```
Input(WINDOW_SAMPLES=300, channels=3)
 ├─ Conv1D(32, k=7, padding=same) → BN → ReLU → MaxPool(2)
 ├─ Conv1D(64, k=5, padding=same) → BN → ReLU → MaxPool(2)
 ├─ Conv1D(128, k=3, padding=same) → BN → ReLU → GlobalAveragePooling1D
 ├─ Dropout(0.3) → Dense(64, ReLU) → Dropout(0.2)
 └─ Dense(1, sigmoid)
```

Optimizer Adam (1e-3), loss BCE, metrics `accuracy` + `AUC`. Kept
deliberately small: TFLite export needs to be mobile-friendly and the
training set is tiny (~1500 windows total), so capacity is bounded to keep
overfitting in check.

Receptive-field rationale: kernel 7 at the bottom over 300 samples (30 s)
gives the early filters a ~700 ms view — enough to see one micro-event like
a sharp lane correction. Stacked Conv1D + pool stages widen the effective
receptive field to most of the window before global pooling.

- `train.py` Two routines:

    - **`leave_one_driver_out(sessions, ...)`** — 6 folds. For each held-out
    driver:

        1. Split sessions by `driver` (no window leakage — the split happens
        *before* windowing).
        2. Window the remaining sessions and the held-out sessions independently.
        3. Compute normalization stats from the training windows only.
        4. Train a fresh model with class-weighted BCE, early-stopping on `val_auc`
        (patience 5, restore best weights).
        5. Evaluate window-level accuracy, AUC, confusion matrix on the held-out
        driver.
        - Returns `{"folds": {...}, "summary": {mean_accuracy, std_accuracy, mean_auc, std_auc, n_folds}}`. Folds with single-class test sets log a warning but still report accuracy.
    - `train_final(sessions, ...)` — trains one model on all 29 sessions with a 10 % internal validation split for early stopping. Saves:

        - artifacts/accel_drowsiness_cnn.keras — Keras SavedModel.
        - artifacts/normalization.json — {channel_names, mean, std, window_samples}.
        The mobile app needs the mean/std to z-score live samples identically.
        - artifacts/labels.json — {class_names: ["Drowsy", "Normal"]}.
- `export_tflite.py`
    - Standard tf.lite.TFLiteConverter.from_keras_model → bytes → file.
    - Output: `artifacts/accel_drowsiness_cnn.tflite`.
- `train_model.py` Same purpose as the camera pipeline's `train_model.py`:
    - python train_model.py train [--epochs N] [--skip-cv]
    - python train_model.py export [--quantize]
    - python train_model.py all





## Week 11 Work Notes | 4/6/26 | Continued Work on Secondary Accelerometer Prediction Model

This week I continued working on trying to implement processing pipeline for Accelerometer model. I did not include the trained model in the final APK.

## Week 12 Work Notes | 4/13/26 | Notes for IOS Development

This week I spent some time looking into what changes or additional developments needs to be done to get this project running on IOS. Since I only had access to an android device I did not spend time attempting to develop this project to be functioning on IOS That being said, this project was build using Expo which boasts a multi-modal framework so it shouldn't be too much of a stretch to get this project working on IOS. From what I've found many of the react-native libraries that I am using in this project support IOS with pretty minimal setup. I created a markdown file for these notes called `NOTES_FOR_IOS.md`

## Week 13 Work Notes | 4/20/26 | NA

## Week 14 Work Notes | 4/27/26 | Running the Application in the Background

This week I worked on getting the application to run in the background, the goal being to have the user enable this app on their phone and allow it to run while they're driving without having to have their phone on. 

I ran into issues with being able to allow the application to run while the phone is locked or while the application is minimized. The issue is that react-native-vision-camera's amera component internally uses Android's CameraX, and CameraX requires a LifecycleOwner (the app itself is the host activity tied to the lifecycle) [**See Android Activity Lifecycle**](https://developer.android.com/guide/components/activities/activity-lifecycle). CameraX registers a lifecycle observer that automatically tears down the camera session whenever the lifecycle drops below RESUMED. When the application is unfocused the lifecycle drops to PAUSED. There is no way to disable this from the JS side; it's a structural choice in CameraX itself and by extension the react-native-vision-camera, not something that I can reasonably configure at this point in the project.

The compromise that I decided to pursue was to allow the app to sit in the Picture-In-Picture display when the user unfocused the application. In the "plugins" directory within "sensor-app/app/" I created "withBackgroundDetection.js" which is a local Expo config plugin that installs the background-detection foreground service plus its React Native bridge, AND the MainActivity changes required to keep VisionCamera's Camera component alive while the app is backgrounded or the device is locked. 

I used Claude to help me generate this file as I wasn't familiar with what changes needed to be made to MainActivity to keep the Activity and the frame processor running. These were the files it created
| File | Purpose |
|---|---|
| `mobile/sensor-app/plugins/withBackgroundDetection.js` | Local Expo config plugin — adds permissions + `<service>`, patches `MainActivity` manifest entry with `supportsPictureInPicture`/`resizeableActivity`/extra `configChanges`, writes Kotlin source files, patches `MainActivity.kt` for the PiP `onUserLeaveHint` override, and patches `MainApplication.kt` to register the package. Reproducible across `prebuild --clean`. |
| `mobile/sensor-app/native/BackgroundDetection.ts` | TS wrapper around `NativeModules.BackgroundDetection`. |
| `mobile/sensor-app/contexts/BackgroundContext.tsx` | React context — owns `enabled`, runs the permission walk-through, calls start/stop on the native module. |

Kotlin files emitted by the plugin (visible after `prebuild`):
- `android/.../sensorapp/BackgroundDetectionService.kt` — the foreground service.
- `android/.../sensorapp/BackgroundDetectionModule.kt` — the native module (now includes permission + window-flag methods).
- `android/.../sensorapp/BackgroundDetectionPackage.kt` — the `ReactPackage` registration.

With these, the program is now able to run in the PiP display. Not quite the ability to run while fully minimized or while the phone is locked but the best I could manage with the library limitations.

`NOTE:` In order to allow the application to have access to being able to have the overlay and PiP you'll need to manually allow the app to have permission. On android: Settings > Apps > All Apps > Select Vigilare > From "App Info" select "Permissions" > Select the 3 dots in the top right corner of the page and allow the sensitive permissions. Otherwise Android will prevent you from providing the unknown app with these permissions. 


## Week 15 Work Notes | 5/4/26 | Final Cleaning & Note Taking

This week I worked on expanding the README to have detailed notes, as well as doing some final code cleaning. 

## Miscellaneous Notes

### Useful Command List

This is a list of commands relevant for building and debugging this project. These commands may be referenced at other points in this README document.
- **Create Virtual Env For Python Dependencies**
    - `python -3.12 -m venv venv(can be whatever you want to name env)`
- **Import Python Dependencies** 
    - `pip install -r requirements.txt`
    - In Bash: `python -m pip install -r pyrequirements.txt`
- **To Activate Python (venv)**
    - `source venv/Scripts/activate`
- **Run Python Scripts**
    - `python script.py`
- **Build Expo to Android (Will build and install on connected android device, must be at /mobile/sensor-app/)**
    - `npx expo prebuild`
    - `npx expo run:android`
    - `npx expo run:android --variant release`
    - `npx expo run:android --variant debug`
- **Build APK Only (no device needed)**
    - `cd mobile/sensor-app/android && ./gradlew assembleRelease`
- **Pull Debug Files From Mobile** 
    - `adb shell run-as com.anonymous.sensorapp ls files`
    - `adb shell run-as com.anonymous.sensorapp ls files/debug_inputs`
    - `adb exec-out "run-as com.anonymous.sensorapp tar -C files -cf - debug_inputs" > debug_inputs.tar`
    - `tar -xf debug_inputs.tar`
- **Clear Debug Directory on Mobile**
    - `adb shell run-as com.anonymous.sensorapp rm -rf files/debug_inputs`
- **Android Debug**
    - `adb logcat | grep com.anonymous.sensorapp`
    - List adb devices: `adb devices`
        - Formatted: `<device_serial> <status>`
    - Remove adb device: `adb -s <device_serial> emu kill`
        - Might have to kill task: qemu-system-x86
- **Create Temp X Drive for Building**
    - `subst X: C:\Users\samsu\CS499-Project`   
- **For full model training pipeline**
    - `python train_model.py` 
- **After new training, before new app build**
    - Copy artifacts: `cp model_training/artifacts/drowsiness_cnn.tflite assets/ml/drowsiness_cnn.tflite`
    - Copy labels: `cp model_training/artifacts/labels.json assets/ml/labels.json`
    - Clean & rebuild: `npx expo prebuild --clean && npx expo run:android --variant release`
- **Rebuild Steps**


### Building App for Remote Development with TailScale
- **Build Release APK**
    ```bash
    cd mobile/sensor-app
    npx expo run:android --variant release
    ```
    The APK will be at: `mobile/sensor-app/android/app/build/outputs/apk/release/app-release.apk`

    If you change native config (app.json plugins, permissions, SDK version), run a clean prebuild first:
    ```bash
    npx expo prebuild --clean && npx expo run:android --variant release
    ```

- **Serve APK to Phone via Tailscale**
    1. Build the APK (see above)
    2. From the project root, serve with:
        ```bash
        python serve.py            # Serve the release APK (default)
        python serve.py --debug    # Serve the debug APK instead
        ```
    3. Get your desktop's Tailscale IP:
        ```bash
        tailscale ip -4
        ```
    4. On your phone (connected to the same tailnet), open a browser and go to:
        ```
        http://<desktop-tailscale-ip>:8080/app-release.apk   # if serving release
        http://<desktop-tailscale-ip>:8080/app-debug.apk     # if serving debug
        ```
    5. Android will prompt you to install. Enable "Install unknown apps" for your browser in **Settings > Apps > [browser] > Install unknown apps** if not already enabled.

    > **Tip:** If you get "App not installed" errors, uninstall the previous version first — release and debug APKs have different signing keys.

- **Pull Debug Images from Mobile**
    1. Enable debug mode in the Drowsiness Detection screen (tap "Debug: OFF" button at the bottom)
    2. Let the app capture some frames, then connect phone via USB
    3. Pull and convert to images:
        ```bash
        python serve.py pull          # Pull debug inputs and convert to images
        python serve.py pull --clean  # Same, but also clear debug files from device
        ```
    4. Images are saved to: `mobile/sensor-app/model_training/debugging/debug_pngs/`

### Build Process Breakdown

There are 3 layers to the build and you only need to redo the layers that are affected by your changes. Listed from heaviest to lightest:

### 1. `npx expo prebuild --clean` — Regenerate native project from scratch
**When you need it:** When you change anything in the native config layer
- Changed `app.json` (plugins, permissions, SDK version, package name)
- Added/removed/updated a native module in `package.json` (like react-native-vision-camera, react-native-fast-tflite, etc)
- Something in `android/` got corrupted or out of sync

**What it does:** Deletes the entire `android/` folder and regenerates it from `app.json` + expo plugins. This links all native modules, sets permissions, configures gradle, etc. Its basically a fresh native project scaffolded from your expo config.

**Command:**
```bash
cd mobile/sensor-app
npx expo prebuild --clean
npx expo run:android --variant release
```

**Note:** This will wipe any manual edits you made inside `android/` (like manually editing build.gradle). Those changes need to go in `app.json` plugins instead so prebuild can recreate them.

### 2. `npx expo run:android --variant release` — Compile native + bundle JS
**When you need it:** When you change TypeScript/JS code OR swap out asset files
- Changed any `.tsx`/`.ts`/`.js` file (like drowsiness.tsx)
- Replaced the `.tflite` model file in `assets/ml/`
- Replaced `labels.json` or other bundled assets
- Basically any time you change code or assets but NOT native config

**What it does:** Two things happen in sequence:
1. **Metro bundler** compiles all your TypeScript/JS into a single bundle and collects assets (including the .tflite file)
2. **Gradle** compiles the native Android code, packages the JS bundle + assets into an APK, and installs it

If native code hasnt changed, gradle reuses most of its cache so this is way faster on repeat runs (~25s vs ~2min). The APK ends up at `android/app/build/outputs/apk/release/app-release.apk`.

**Command:**
```bash
cd mobile/sensor-app
npx expo run:android --variant release
```

**Debug vs Release:** `--variant release` makes an optimized APK you can install on any phone. Without it you get a debug build that needs a dev server running (metro). For testing on your phone you almost always want release.

**Building APK only (no device connected):** `npx expo run:android` will try to install the APK on a connected device/emulator and fail if none is found. If you just want to generate the APK file without installing, use gradle directly:
```bash
cd mobile/sensor-app
npx expo prebuild --platform android --clean
cd mobile/sensor-app/android
./gradlew assembleRelease
```
This skips the device-install step and produces the same APK at `android/app/build/outputs/apk/release/app-release.apk`. Works on any machine with the Android SDK — no phone, USB cable, or emulator needed. You can then transfer the APK to your phone however you like (Tailscale serve, Google Drive, USB, etc).

### 3. Metro only (dev server) — Hot reload JS changes
**When you need it:** Only during active development when you have a debug build connected
- Changed TypeScript/JS code and want to see it instantly without rebuilding

**What it does:** Recompiles just the changed JS and pushes it to the running debug app over the network. Does NOT work with release builds.

**Command:**
```bash
cd mobile/sensor-app
npx expo start --clear   # --clear wipes metro cache if things are stale
```

### Quick reference — what changed → what to run

| What you changed | Command needed |
|---|---|
| `.tsx`/`.ts` code (like drowsiness.tsx) | `npx expo run:android --variant release` |
| `.tflite` model or `labels.json` | `npx expo run:android --variant release` |
| `app.json` (plugins, permissions, etc) | `npx expo prebuild --clean` then `run:android` |
| Added/removed npm package with native code | `npm install` then `npx expo prebuild --clean` then `run:android` |
| `android/` is broken or weird build errors | `npx expo prebuild --clean` then `run:android` |
| Python training code only (preprocess/train) | No app build needed, just retrain and copy artifacts |
| Any of the above, but no device connected | Use `./gradlew assembleRelease` from `android/` instead of `npx expo run:android` |

### Common gotcha — stale .tflite in cache
If you retrained and copied a new `.tflite` to `assets/ml/` but the app still uses the old model, metro or gradle might have cached the old one. Fix with:
```bash
npx expo start --clear    # ctrl+c after it starts, just wipes metro cache
npx expo run:android --variant release
```
Or the nuclear option: `npx expo prebuild --clean && npx expo run:android --variant release`

![Vigilare Logo](/assets/icon.png)
