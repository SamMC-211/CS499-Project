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


# Commands Ran
**Create Virtual Env For Python Dependencies**
- .../CS499-Project/mobile/sensor-app (main)
    - $ py -3.12 --version
    - Python 3.12.10
    - $ py -3.12 -m venv venv(can be whatever you want to name env)
    - *Activate on Linux*
        - source venv/bin/activate
- **From venv** 
    - python
    - import mediapipe as mp
    - print(mp.__version__)
- **To Activate Python (venv)??**
    - source venv/Scripts/activate

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