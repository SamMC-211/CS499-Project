# CS499-Project
This project is to explore how to build a mobile application using **React Native** that can collect sensor data and apply machine learning–based activity recognition techniques.   The core idea is to leverage mobile device sensors to capture user behavioral signals and use machine learning models to classify certain states or activities.


# Command List

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
- **Build Expo to Android (Will build and install on connected android device)**
    - `npx expo prebuild`
    - `npx expo run:android`
    - `npx expo run:android --variant release`
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

# Building App for Remote Development with TailScale
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

## Build Process Breakdown

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

# Week 3 Meeting Notes
- Available MediaPipe facial detection models
    - https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector#models
    - Short-Range Blazeface
- MediaPipe Sample Code
    - https://github.com/google-ai-edge/mediapipe-samples
- MediaPipe Tasks Python API
    - https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_detector/python/face_detector.ipynb#scrollTo=L_cQX8dWu4Dv
- MediaPipe Tasks Android Native Demo
    - https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/face_detector/android

**Current Tech Stack**
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

**Notes** 
- Mediapipe
    - Face/landmark/head pose detection 
    - Required for your pipeline
- TensorFlow
    - Train/run custom neural network models on top of Mediapipe outputs
- OpenCV
    - Image preprocessing, video frame handling

**Test Data** 
- [Driver Drowsiness](https://huggingface.co/datasets/ckcl/driver-safety-dataset) (Labled Images)
- [Lateral Acceleration](https://github.com/commaai/comma-steering-control?tab=readme-ov-file) (openpilot driver assistance system)
- [Drowsy Detection](https://www.kaggle.com/datasets/yasharjebraeily/drowsy-detection-dataset) (Greyscale face images)
- [Drowsiness Detection System](https://www.kaggle.com/code/mohamedkhaledelsafty/drowsiness-detection-system/n) (Python processing imports given, Colored full face)
- [Driver drowsiness using keras](https://www.kaggle.com/code/adinishad/driver-drowsiness-using-keras/notebook) (Eyes closed/Yawn)


# Week 4 Meeting Notes

This week I spent some time attempting to get a model created using the example code and training data from [Drowsiness Detection System](https://www.kaggle.com/code/mohamedkhaledelsafty/drowsiness-detection-system/n). It took me a while and some adjustments were made as I went. 
- Adapted example code into python files for each step of the model creation
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

# Week 5 Meeting Notes

I spent this week familiarizing myself with and troubleshooting my implementation of the preprocessing steps needed before I start feeding faces to my model. This was quite difficult as I was running into issues with mapping points from the react-native-vision-camera-face-detector. Heres a list of some of the things I accomplished during this week.
- **Implemented **
    - frameProcessor
    - mapLandmarkToCrop
    - stampDot
- Troubleshot the process of mapping contour points from the face-detector to the user camera view
    - Scaling conversion issues
    - View bounds issue


# Week 6 Meeting Notes

This week I worked on getting the model up and running on mobile, I toiled a lot over this issue and while I didn't gain much ground in terms of app side development I did realize some changes to my approach I may need to make for this mobile application. Specificially addressing the disconnect between the preprocessing and training steps for the model vs the capabilities of mobile. 
- Familiarized myself with react-native-fast-tflite
- Create a Python script to render images from the Float32Array format that I am preprocessing frame inputs into for debugging
- Spent time troubleshooting the image stamping of inputs for the model and mapping points from the face detector to the frame
- Began feeding preprocessed images to my model and created UI to display driver state prediction
- Build Issues, had to set minSdkVersion to 26 in 'android/app/build.gradle'


# Week 7

This week I've spent attempting to retrain the model to be accurate on mobile.

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


# Week 8 Meeting Notes

This week I worked on fixing model accuracy on mobile and adding prediction smoothing

- **Fixed input value range mismatch** — `vision-camera-resize-plugin` returns float32 in [0, 1] but the model's Rescaling(1/255) layer expects [0, 255]. Added a `* 255.0` scaling step after rotation, before dot stamping. Before that the model was seeing a nearly black image. 
![Example Image](/)
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

**Notes for week 9**
- Look into using 1 additional sensor
- Look into using driving time to advise drowsiness alert/prediction
- Add driver alert functionality
- Add Screenshots to meeting document
- Work on cleaning up code 
- Attach the project APK 
- Create Notes for adapting project to IOS

# Week 9 Meeting Notes

NA

# Week 10 Meeting Notes
