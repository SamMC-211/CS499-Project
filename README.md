# CS499-Project
This project is to explore how to build a mobile application using **React Native** that can collect sensor data and apply machine learning–based activity recognition techniques.   The core idea is to leverage mobile device sensors to capture user behavioral signals and use machine learning models to classify certain states or activities.


✅ Your project is ready!

To run your project, navigate to the directory and run one of the following npm commands.

- cd sensor-app
- npm run android
- npm run ios # you need to use macOS to build the iOS project - use the Expo app if you need to do iOS development without a Mac
- npm run web


sensor-app/
├── app/              ← Expo Router (THIS is your screens)
├── app-example/      ← demo / starter content (safe to delete)
├── assets/           ← images, fonts
├── app.json
├── expo-env.d.ts
├── eslint.config.js
├── node_modules/
├── package.json
├── tsconfig.json
└── README.md


Key takeaway

app/ is ONLY for routes/screens

Don’t dump sensor logic, state, or data handling directly into app/

If you do, things get messy fast.