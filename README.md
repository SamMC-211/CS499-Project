# CS499-Project
This project is to explore how to build a mobile application using **React Native** that can collect sensor data and apply machine learning–based activity recognition techniques.   The core idea is to leverage mobile device sensors to capture user behavioral signals and use machine learning models to classify certain states or activities.


**Expo Notes** 
https://docs.expo.dev/develop/tools/
- npm run android
- npm run ios # you need to use macOS to build the iOS project - use the Expo app if you need to do iOS development without a Mac
- npm run web
- npx expo start	Starts the development server (whether you are using a development build or Expo Go).
- npx expo prebuild	Generates native Android and iOS directories using Prebuild.
- npx expo run:android	Compiles native Android app locally.
- npx expo run:ios	Compiles native iOS app locally.
- npx expo install package-name	Used to install a new library or validate and update specific libraries in your project by adding --fix option to this command.
- npx expo lint	Setup and configures ESLint. If ESLint is already configured, this command will lint your project files.

# Expo Docs
---
**Detailed Camera Use** 
https://github.com/expo/examples/blob/master/with-camera/App.tsx
**Accelerometer**
https://docs.expo.dev/versions/latest/sdk/accelerometer/


# Command List
- **Create Virtual Env For Python Dependencies**
    - `python -3.12 -m venv venv(can be whatever you want to name env)`
- **Import Python Dependencies** 
    - `pip install -r requirements.txt`
    - In Bash: `python -m pip install -r pyrequirements.txt`
- **To Activate Python (venv)**
    - `source venv/Scripts/activate`
- **Run Python Scripts**
    - python script.py
- **Build Expo to Android**
    - `npx expo prebuild`
    - `npx expo run:android`
    - `npx expo run:android --variant release`
- **Pull Debug Files From Mobile** 
    - `adb shell run-as com.anonymous.sensorapp ls files`
    - `adb shell run-as com.anonymous.sensorapp ls files/debug_inputs`
    - `adb exec-out "run-as com.anonymous.sensorapp tar -C files -cf - debug_inputs" > debug_inputs.tar`
    - `tar -xf debug_inputs.tar`
- **Clear Debug Directory on Mobile**
    - `adb shell run-as com.anonymous.sensorapp rm -rf files/debug_inputs`
- **Android Debug**
    - `adb logcat | grep com.anonymous.sensorapp`
- **Create Temp X Drive for Building**
    - `subst X: C:\Users\samsu\CS499-Project`   
- **For full model training pipeline**
    - `python train_model.py` 
- **After new training, before new app build**
    - Copy artifacts: `cp model_training/artifacts/drowsiness_cnn.tflite assets/ml/drowsiness_cnn.tflite`
    - Copy labels: `cp model_training/artifacts/labels.json assets/ml/labels.json`
    - Clean & rebuild: `npx expo prebuild --clean && npx expo run:android --variant release`
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

- **Serve Release APK to Phone via Tailscale**
    1. Build the release APK (see above)
    2. From the project root, serve with:
        ```bash
        python serve.py
        ```
    3. Get your desktop's Tailscale IP:
        ```bash
        tailscale ip -4
        ```
    4. On your phone (connected to the same tailnet), open a browser and go to:
        ```
        http://<desktop-tailscale-ip>:8080/app-release.apk
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

Model Versions (In Artifacts)
- Drowsiness (Base model using MediaPipe facial landmarking)   
    - Most accurate so far, fluctuating reading
- Drowsiness 1.0 (Model using MLKit facial landmarking)
    - Missing landmarks innacurate on mobile
- Drowsiness 2.0 (Model using MLKit including previously missing landmark groups)
    - Updated preprocessing to include all landmark groups
    - Still highly innacurate on mobile

- Created serve.py to download apk remotely after building
- Drow
        