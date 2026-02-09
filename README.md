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
    - $ py -3.12 -m venv venv
    - *Activate on Linux*
        - source venv/bin/activate
- **From venv** 
    - python
    - import mediapipe as mp
    - print(mp.__version__)

# Week 3 Meeting Notes
---
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
    - MediaPipe 0.10.32
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