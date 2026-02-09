# CS499-Project
This project is to explore how to build a mobile application using **React Native** that can collect sensor data and apply machine learning–based activity recognition techniques.   The core idea is to leverage mobile device sensors to capture user behavioral signals and use machine learning models to classify certain states or activities.


**Expo Notes** 
https://docs.expo.dev/develop/tools/

- cd sensor-app
- npm run android
- npm run ios # you need to use macOS to build the iOS project - use the Expo app if you need to do iOS development without a Mac
- npm run web

**Command Description**
- npx expo start	Starts the development server (whether you are using a development build or Expo Go).
- npx expo prebuild	Generates native Android and iOS directories using Prebuild.
- npx expo run:android	Compiles native Android app locally.
- npx expo run:ios	Compiles native iOS app locally.
- npx expo install package-name	Used to install a new library or validate and update specific libraries in your project by adding --fix option to this command.
- npx expo lint	Setup and configures ESLint. If ESLint is already configured, this command will lint your project files.

# Links 
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
**From venv** 
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

**Current Tech Stack**
- Frontend/Mobile
    - React Native
- Local Facial Processing
    - MediaPipe 0.10.32
        - Python 3.12
            - numpy
    - TensorFlow (Training -> TFLite for Local running)
    - OpenCV-Python (Useful for reading video and images to train model)
    - MediaPipe/tasks-vision (Replace OpenCV for mobile runtime to read face data)

**Notes** 
- Mediapipe
    - Face/landmark/head pose detection 
    - Required for your pipeline
- TensorFlow
    - Train/run custom neural network models on top of Mediapipe outputs
- OpenCV
    - Image preprocessing, video frame handling