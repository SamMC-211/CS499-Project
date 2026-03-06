# Drowsiness Detection App — Development Guide

This document is a living reference for the next stages of development on this project. It covers known issues, structural improvements, mobile-specific concerns, UI/UX tooling, and learning context for each topic.

---

## Table of Contents

1. [Project Architecture Summary](#1-project-architecture-summary)
2. [Critical Bug: Label Mapping Is Inverted](#2-critical-bug-label-mapping-is-inverted)
3. [The Core Problem: Preprocessing Parity](#3-the-core-problem-preprocessing-parity)
4. [Model Accuracy Improvements](#4-model-accuracy-improvements)
5. [Mobile App Architecture](#5-mobile-app-architecture)
6. [Navigation & Screen Structure](#6-navigation--screen-structure)
7. [Sensor Fusion: Combining Camera + Accelerometer](#7-sensor-fusion-combining-camera--accelerometer)
8. [Alert System Design](#8-alert-system-design)
9. [Mobile Requirements, Restrictions & User Expectations](#9-mobile-requirements-restrictions--user-expectations)
10. [Performance & Battery Optimization](#10-performance--battery-optimization)
11. [UI/UX Tooling & Approach](#11-uiux-tooling--approach)
12. [What You Are Learning (Student Context)](#12-what-you-are-learning-student-context)

---

## 1. Project Architecture Summary

The project has two major components that must stay in sync:

```
CS499-Project/
├── mobile/sensor-app/
│   ├── app/                        # Expo Router screens
│   │   ├── _layout.tsx             # Root Stack navigator
│   │   └── sensors/
│   │       ├── camera.tsx          # Camera screen (expo-camera, no inference yet)
│   │       └── accelerometer.tsx   # Accelerometer debug screen
│   ├── components/
│   │   └── Nav_Button.tsx          # Reusable navigation button
│   ├── assets/ml/
│   │   ├── drowsiness_cnn.tflite   # Bundled model (copied from artifacts)
│   │   └── labels.json
│   └── model_training/
│       ├── config.py               # Shared paths and constants
│       ├── preprocess.py           # Current: MediaPipe Tasks + dots
│       ├── preprocess_backup.py    # Old: MediaPipe Solutions + tesselation mesh
│       ├── train.py                # CNN definition + training loop
│       ├── export_tflite.py        # Keras -> TFLite conversion
│       ├── train_model.py          # End-to-end pipeline runner
│       └── artifacts/              # Trained model outputs
```

**Pipeline flow:**

```
Raw dataset images
    -> preprocess.py  (Haar cascade face crop + MediaPipe landmark dots)
    -> processed/     (145x145 face images with dots drawn)
    -> train.py       (CNN trained on processed images)
    -> artifacts/drowsiness_cnn.keras
    -> export_tflite.py
    -> artifacts/drowsiness_cnn.tflite
    -> copied to assets/ml/
    -> Mobile app loads and runs inference per-frame
```

The critical constraint: **whatever drawing step happens in preprocessing must be reproduced exactly on mobile before each frame is fed to the model.**

---

## 2. Critical Bug: Label Mapping Is Inverted

This is a correctness issue that will cause the model to display the wrong label every single time.

### The Problem

In `config.py`:
```python
CLASS_NAMES = ["Fatigue Subjects", "Active Subjects"]
```

When `image_dataset_from_directory` is called with an explicit `class_names` list, TensorFlow assigns integer labels in the order you provide:
- Index `0` → `"Fatigue Subjects"`
- Index `1` → `"Active Subjects"`

The model's final layer is `Dense(1, activation="sigmoid")`. Sigmoid output approaches `1.0` when the model predicts class `1` (Active), and approaches `0.0` when it predicts class `0` (Fatigue).

### The Bug in EXPO_REALTIME_INTEGRATION.md

The skeleton code in that document has this line:
```tsx
setStateText(score >= 0.5 ? "Fatigue Subjects" : "Active Subjects");
```

This is **backwards**. A score >= 0.5 means the model is predicting class 1, which is "Active Subjects."

### The Fix

```tsx
// score >= 0.5 means the model predicts class 1 = Active
setStateText(score >= 0.5 ? "Active Subjects" : "Fatigue Subjects");
```

Or even better, load the labels from `labels.json` dynamically so this never gets out of sync:

```tsx
import labels from '../../assets/ml/labels.json';

// labels.class_names[0] = "Fatigue Subjects" (score near 0)
// labels.class_names[1] = "Active Subjects"  (score near 1)
const label = score >= threshold
  ? labels.class_names[1]  // Active
  : labels.class_names[0]; // Fatigue
```

### Why This Matters (Learning Note)

This is one of the most common bugs in ML deployment. The model does not know what "Fatigue" means — it just learns to output a number. The meaning of that number is determined entirely by how you encoded labels during training. Always trace your class index assignments from `image_dataset_from_directory` all the way through to your inference display code.

---

## 3. The Core Problem: Preprocessing Parity

This is the most important technical challenge in the project right now.

### What the Model Learned

During preprocessing, every image that gets fed into training goes through:
1. Haar cascade face detection (crop to face bounding box)
2. MediaPipe `FaceLandmarker` runs on the cropped face
3. White dots are drawn at each of the 478 landmark positions (radius 2 for eye landmarks, radius 1 for others)
4. Resized to 145x145, RGB, normalized to [0,1]

The model has never seen a "clean" face image. It has only ever seen faces with white dots painted on them. This means on mobile, you **must** reproduce steps 1–3 before calling the model.

### What Needs to Happen on Mobile (per frame)

```
Camera frame
    -> react-native-vision-camera-face-detector: get face bounding box
    -> crop frame to face bounding box
    -> run MediaPipe landmark detection on the crop
       (either via @mediapipe/tasks-vision in JS, or the face-detector plugin's landmark output)
    -> draw white dots at each landmark position onto the cropped image buffer
    -> resize to 145x145
    -> convert to float32 RGB normalized [0, 1]
    -> feed to model via react-native-fast-tflite
```

### Why MLKit Is Relevant Here

Google MLKit's Face Detection API (available via the `react-native-vision-camera-face-detector` plugin) provides face bounding boxes and facial contour points. The contour points are a subset of full 478-point landmarks, but they cover the key regions (eyes, eyebrows, lips, face outline). The trade-off:

| Option | Points | Speed | Parity with Training |
|---|---|---|---|
| Full MediaPipe 478-point (`@mediapipe/tasks-vision`) | 478 | Slower (runs in JS thread) | Exact match to training |
| MLKit / VisionCamera face-detector contours | ~130 | Fast (runs native) | Approximate |

**If you retrain the model using only MLKit-style contour points**, the training preprocessing and mobile inference will be a much closer match. This is the approach you mentioned — it is the right call.

### How to Retrain with MLKit-Compatible Points

1. In `preprocess.py`, instead of drawing all 478 MediaPipe landmarks, only draw the subset of points that MLKit's contour detection returns on mobile.
2. The `react-native-vision-camera-face-detector` gives you `contours` — these map to specific MediaPipe landmark index groups (left eye, right eye, face oval, etc.).
3. Identify which landmark indices the plugin returns, and restrict your training preprocessing to draw only those same indices.

```python
# In preprocess.py — conceptual change to match MLKit contour output
# MLKit face contours roughly correspond to these MediaPipe index groups:
MLKIT_APPROX_INDICES = {
    # Left eye contour
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    # Right eye contour
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    # Face oval
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
}

# Then in draw_and_save_face_mesh, only draw dots for indices in this set
for i, landmark in enumerate(face_landmarks):
    if i not in MLKIT_APPROX_INDICES:
        continue
    point = _normalized_to_pixel_coordinates(landmark.x, landmark.y, img_w, img_h)
    if point:
        cv2.circle(image_drawing_tool, point, 2, (255, 255, 255), -1)
```

The goal is: **the image buffer you draw dots on in mobile should look as close as possible to the images the model trained on.**

---

## 4. Model Accuracy Improvements

### Current Training Metrics

From `artifacts/metrics.json`, final eval accuracy is **~95%** and final eval loss is **~0.117**. However, look at the validation accuracy across epochs:

```
epoch 1:  val_acc = 1.00    (suspiciously perfect)
epoch 2:  val_acc = 0.55    (random-chance level)
epoch 6:  val_acc = 0.57
epoch 8:  val_acc = 0.74
epoch 10: val_acc = 0.81
epoch 20: val_acc = 0.95
```

This oscillation is a warning sign. It could mean:
- The validation set is small (high variance in each estimate)
- The model is inconsistently generalizing between epochs
- The dataset may have a class imbalance

### What to Improve

**Add Early Stopping and Model Checkpointing**

Don't just save the last epoch. Save the epoch with the best validation accuracy:

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    ModelCheckpoint(
        filepath=str(artifacts_dir / "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,           # stop if val_acc doesn't improve for 5 epochs
        restore_best_weights=True,
        verbose=1,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks,   # <-- add this
)
```

**Why this matters:** You trained for 20 epochs, but epoch 12 had `val_accuracy = 0.981`. If you had saved the best checkpoint, you would deploy a better model.

**Check for Class Imbalance**

If your dataset has significantly more "Active" images than "Fatigue" images (or vice versa), the model learns to just predict the majority class. Add a check:

```python
import os
for class_name in CLASS_NAMES:
    count = len(list((processed_dir / class_name).iterdir()))
    print(f"{class_name}: {count} images")
```

If one class has more than ~1.5x the other, add class weights to training:

```python
total = sum(counts.values())
class_weight = {
    i: total / (len(counts) * count)
    for i, (name, count) in enumerate(counts.items())
}
history = model.fit(..., class_weight=class_weight)
```

**Add a Confusion Matrix After Training**

Accuracy alone doesn't tell you if the model is better at detecting drowsiness vs. alertness. Add this to `train.py` after `model.evaluate()`:

```python
import numpy as np

all_labels, all_preds = [], []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    all_labels.extend(labels.numpy())
    all_preds.extend((preds.squeeze() >= 0.5).astype(int))

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(all_labels, all_preds)
print(cm)
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
```

This tells you: "Of all actual drowsy frames, how many did the model correctly flag?" That metric (recall for the drowsy class) is more important for a safety app than raw accuracy.

**Consider Transfer Learning**

Your current CNN is trained from scratch on a relatively small dataset. A pre-trained model like MobileNetV2 has already learned general image features (edges, textures, shapes) on millions of images. You can fine-tune it on your data:

```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(145, 145, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze base weights initially

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
```

MobileNetV2 is specifically designed to be small and fast — it exports to TFLite cleanly and runs well on mobile.

---

## 5. Mobile App Architecture

### Current State

The app currently has:
- `app/_layout.tsx` — Stack navigator root (minimal, no headers configured)
- `app/sensors/camera.tsx` — Shows camera preview with `expo-camera`, no inference
- `app/sensors/accelerometer.tsx` — Working accelerometer debug display

### What Needs to Change

**camera.tsx needs to be completely rebuilt** around the inference pipeline. The current `expo-camera` implementation is a placeholder. The real implementation requires `react-native-vision-camera` with frame processors. You already have all the right packages installed (`react-native-vision-camera`, `react-native-fast-tflite`, `vision-camera-resize-plugin`).

The recommended file structure for the app going forward:

```
app/
├── _layout.tsx                  # Root layout (keep as Stack or convert to Tabs)
├── index.tsx                    # Home/landing screen
├── drowsiness/
│   ├── _layout.tsx              # Optional nested layout
│   └── index.tsx                # Main drowsiness detection screen (camera + inference)
└── debug/
    ├── accelerometer.tsx        # Keep for development/testing
    └── model-info.tsx           # Display model metrics, labels, version
```

**Separate concerns into services/hooks:**

Right now all logic would be crammed into one screen component. Instead, break it up:

```
hooks/
├── useDrowsinessModel.ts    # loads tflite, runs inference, returns prediction
├── useFaceDetector.ts       # wraps vision-camera face detection
└── usePredictionSmoothing.ts # smooths raw per-frame scores

services/
└── alertService.ts          # decides when to trigger alerts
```

This makes each piece independently testable and keeps your screen components clean.

**Example: `useDrowsinessModel.ts` concept**

```typescript
import { useTensorflowModel } from 'react-native-fast-tflite';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useRef } from 'react';

const MODEL_INPUT_SIZE = 145;

export function useDrowsinessModel() {
  const { model, state } = useTensorflowModel(
    require('../assets/ml/drowsiness_cnn.tflite')
  );
  const { resize } = useResizePlugin();

  // Returns raw sigmoid score 0.0-1.0
  // score < 0.5 = Fatigue, score >= 0.5 = Active
  const runInference = (frame: Frame): number | null => {
    if (!model || state !== 'loaded') return null;
    const input = resize(frame, {
      scale: { width: MODEL_INPUT_SIZE, height: MODEL_INPUT_SIZE },
      pixelFormat: 'rgb',
      dataType: 'float32',
    });
    const output = model.runSync([input]) as unknown[];
    return Number((output[0] as number[])[0]);
  };

  return { runInference, isReady: state === 'loaded' };
}
```

---

## 6. Navigation & Screen Structure

### Current Navigation

The app uses `expo-router` with a single `<Stack />` in `_layout.tsx`. This works but is minimal — there are no screen titles, no header buttons, and no bottom tab bar.

### Recommended: Add a Bottom Tab Navigator

For a drowsiness detection app, a simple two or three tab layout is the clearest structure for users:

```
Tab 1: Detection    (camera + live inference — the main feature)
Tab 2: Settings     (sensitivity threshold, alert type, etc.)
Tab 3: About/Info   (model accuracy info, how it works)
```

**How to set this up with Expo Router:**

```
app/
├── _layout.tsx          # Root (can stay as Stack wrapping tabs)
└── (tabs)/
    ├── _layout.tsx      # Tab bar definition
    ├── index.tsx        # Detection tab (main screen)
    ├── settings.tsx     # Settings tab
    └── info.tsx         # About tab
```

```tsx
// app/(tabs)/_layout.tsx
import { Tabs } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';

export default function TabLayout() {
  return (
    <Tabs screenOptions={{ tabBarActiveTintColor: '#1162cc' }}>
      <Tabs.Screen
        name="index"
        options={{
          title: 'Detection',
          tabBarIcon: ({ color }) => (
            <Ionicons name="eye" size={24} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: 'Settings',
          tabBarIcon: ({ color }) => (
            <Ionicons name="settings-outline" size={24} color={color} />
          ),
        }}
      />
    </Tabs>
  );
}
```

### Understanding Expo Router File-Based Routing (Learning Note)

Expo Router maps file paths directly to URL/navigation paths — this is the same idea as Next.js for the web. A file at `app/(tabs)/index.tsx` becomes the `/` route inside a tab group. The parentheses `(tabs)` in a folder name mean "this is a layout group — don't include the folder name in the URL path." This is a powerful pattern worth understanding because it is increasingly standard in both web and mobile React development.

---

## 7. Sensor Fusion: Combining Camera + Accelerometer

The accelerometer screen is currently just a debug display. It has real potential as a supplementary drowsiness signal:

- A drowsy driver's head may nod (sudden tilt on the X/Y axis)
- The phone sitting on a dashboard will pick up erratic lane changes (lateral Z spikes)
- Long periods of very low accelerometer variance can indicate the driver has stopped moving entirely (another fatigue indicator)

### Simple Fusion Approach

Don't replace the camera model — add the accelerometer as a secondary signal to reduce false positives.

```typescript
// Conceptual fusion logic
function computeAlertLevel(
  visionScore: number,     // 0.0 = definitely drowsy, 1.0 = definitely alert
  accelVariance: number,   // low variance = very still = potential fatigue
  headNodDetected: boolean
): 'safe' | 'warning' | 'alert' {
  const visionDrowsy = visionScore < 0.4;
  const motionDrowsy = accelVariance < 0.02 || headNodDetected;

  if (visionDrowsy && motionDrowsy) return 'alert';
  if (visionDrowsy || motionDrowsy) return 'warning';
  return 'safe';
}
```

This "AND" logic means both signals must agree before a full alert fires — this reduces false alarms significantly, which is critical for user trust.

---

## 8. Alert System Design

Currently there is no alert system at all. This is the core user-facing safety feature. Here is how to approach it:

### Prediction Smoothing First

Never fire an alert on a single frame. Smooth the raw per-frame scores:

```typescript
// hooks/usePredictionSmoothing.ts
import { useRef } from 'react';

export function usePredictionSmoothing(windowSize = 10) {
  const buffer = useRef<number[]>([]);

  const addScore = (score: number): number => {
    buffer.current.push(score);
    if (buffer.current.length > windowSize) {
      buffer.current.shift(); // remove oldest
    }
    // Return the moving average
    const sum = buffer.current.reduce((a, b) => a + b, 0);
    return sum / buffer.current.length;
  };

  return { addScore };
}
```

### Alert Levels

A good drowsiness app uses escalating alerts rather than a sudden alarm:

| Level | Trigger | Response |
|---|---|---|
| Warning | Smoothed score < 0.4 for 3+ seconds | Subtle visual indicator changes color |
| Caution | Smoothed score < 0.35 for 5+ seconds | Screen pulses / vibration |
| Alert | Smoothed score < 0.3 for 8+ seconds | Loud audio alarm + full-screen overlay |

### Haptic Feedback

```tsx
import * as Haptics from 'expo-haptics';

// In your alert handler:
await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
// or for a stronger alert:
await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
```

### Audio Alert

```tsx
import { Audio } from 'expo-av';

const soundRef = useRef<Audio.Sound | null>(null);

const playAlert = async () => {
  const { sound } = await Audio.Sound.createAsync(
    require('../assets/audio/alert.mp3')
  );
  soundRef.current = sound;
  await sound.playAsync();
};
```

> You will need `expo-av` installed (`npx expo install expo-av`) and an alert sound file in `assets/audio/`.

---

## 9. Mobile Requirements, Restrictions & User Expectations

### Permissions

Android requires explicit permissions declared in `AndroidManifest.xml` AND requested at runtime in code. You already handle camera permission. If you add audio alerts or background detection, you will also need:

- `VIBRATE` permission (for haptics in some cases — usually granted by default)
- `FOREGROUND_SERVICE` if you want detection to run while the screen is locked or another app is open
- `WAKE_LOCK` to keep the CPU running while the screen is on during detection

```xml
<!-- android/app/src/main/AndroidManifest.xml -->
<uses-permission android:name="android.permission.VIBRATE" />
<uses-permission android:name="android.permission.WAKE_LOCK" />
```

### Screen Always-On

During active detection, the screen should never dim or lock. Without this, the camera feed stops and detection fails.

```tsx
import { activateKeepAwakeAsync, deactivateKeepAwake } from 'expo-keep-awake';

useEffect(() => {
  activateKeepAwakeAsync();
  return () => deactivateKeepAwake();
}, []);
```

Install: `npx expo install expo-keep-awake`

### Background Execution Limits

Android aggressively kills background processes to save battery. If the user switches apps, your detection will stop. This is an important UX consideration — let the user know detection is only active when the app is in the foreground. You can detect this with `AppState`:

```tsx
import { AppState } from 'react-native';

useEffect(() => {
  const subscription = AppState.addEventListener('change', (state) => {
    if (state !== 'active') {
      // Detection paused — show a re-open reminder
    }
  });
  return () => subscription.remove();
}, []);
```

### Safe Area / Notch Handling

Always wrap your main screen content in a `SafeAreaView` to avoid camera notch and navigation bar overlap:

```tsx
import { SafeAreaView } from 'react-native-safe-area-context';

export default function DetectionScreen() {
  return (
    <SafeAreaView style={{ flex: 1 }}>
      {/* camera + overlay */}
    </SafeAreaView>
  );
}
```

### Portrait vs. Landscape

A front-facing driver camera should be locked to portrait. In `app.json`:

```json
{
  "expo": {
    "orientation": "portrait"
  }
}
```

### First-Launch User Onboarding

Users will not know how to position the phone for best detection. A one-time onboarding screen (shown only on first launch) explaining "Mount phone on dashboard facing driver" with a diagram goes a long way. Use `AsyncStorage` to remember if onboarding was shown:

```tsx
import AsyncStorage from '@react-native-async-storage/async-storage';

const hasOnboarded = await AsyncStorage.getItem('onboarded');
if (!hasOnboarded) {
  router.push('/onboarding');
  await AsyncStorage.setItem('onboarded', 'true');
}
```

Install: `npx expo install @react-native-async-storage/async-storage`

---

## 10. Performance & Battery Optimization

### Frame Rate Throttling

Running inference on every camera frame (up to 60 FPS) is wasteful and will drain the battery fast. You only need 3–5 inferences per second for drowsiness detection.

```tsx
const lastInferenceTime = useRef(0);
const INFERENCE_INTERVAL_MS = 200; // ~5 FPS

const frameProcessor = useFrameProcessor((frame) => {
  'worklet';
  const now = Date.now();
  if (now - lastInferenceTime.value < INFERENCE_INTERVAL_MS) return;
  lastInferenceTime.value = now;

  // ... run inference
}, [model]);
```

### TFLite Quantization

When exporting the model, use the `--quantize` flag:

```bash
python train_model.py export --quantize
```

This applies post-training quantization which reduces model size by ~75% and speeds up inference by converting float32 weights to int8. The trade-off is a small accuracy drop (usually < 1%) that is worth it on mobile.

You can go further with full integer quantization by providing a representative dataset during conversion — this maximizes compatibility with mobile hardware accelerators (GPU delegate, NNAPI on Android).

```python
# In export_tflite.py — full integer quantization example
def representative_data_gen():
    for image_batch, _ in sample_dataset.take(100):
        for image in image_batch:
            yield [tf.expand_dims(image, 0)]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
```

### Camera Resolution

`react-native-vision-camera` lets you specify the camera format. Running inference on a full 4K frame before resizing it is wasteful. Request a lower resolution format:

```tsx
const format = useCameraFormat(device, [
  { videoResolution: { width: 640, height: 480 } },
  { fps: 30 },
]);

<Camera format={format} ... />
```

---

## 11. UI/UX Tooling & Approach

This section covers the standard tools and patterns for building the visual side of a React Native app.

### Core Styling: `StyleSheet`

React Native does not use CSS. Styles are JavaScript objects. The `StyleSheet.create()` call is a performance optimization that validates styles at build time:

```tsx
import { StyleSheet, View, Text } from 'react-native';

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  label: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
});
```

The layout system is Flexbox — the same as CSS Flexbox, but `flexDirection` defaults to `column` instead of `row` in React Native.

### Icon Library: `@expo/vector-icons`

Already installed. Gives you access to icon sets like Ionicons, MaterialIcons, FontAwesome:

```tsx
import { Ionicons } from '@expo/vector-icons';

<Ionicons name="eye-off-outline" size={32} color="red" />
```

Browse all available icons at: `icons.expo.fyi`

### Animations: `react-native-reanimated`

Already installed (`react-native-reanimated ~4.1.1`). This is the standard library for smooth, performant animations in React Native. A key concept is that animations run on the **UI thread** rather than the JavaScript thread, which prevents jank even when JS is busy with inference.

**Example: Pulsing warning indicator**

```tsx
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
} from 'react-native-reanimated';

function DrowsinessIndicator({ isDrowsy }: { isDrowsy: boolean }) {
  const opacity = useSharedValue(1);

  useEffect(() => {
    if (isDrowsy) {
      opacity.value = withRepeat(withTiming(0.2, { duration: 500 }), -1, true);
    } else {
      opacity.value = 1;
    }
  }, [isDrowsy]);

  const animatedStyle = useAnimatedStyle(() => ({
    opacity: opacity.value,
  }));

  return (
    <Animated.View style={[styles.indicator, animatedStyle,
      { backgroundColor: isDrowsy ? 'red' : 'green' }
    ]} />
  );
}
```

### Drawing Overlays: `@shopify/react-native-skia`

For drawing face landmark dots and bounding boxes on top of the camera preview in real time. Skia integrates directly with VisionCamera frame processors so drawing happens on the UI thread alongside the camera preview.

```tsx
import { Canvas, Circle, Group } from '@shopify/react-native-skia';

// Draw a dot at each landmark position
<Canvas style={StyleSheet.absoluteFill}>
  <Group>
    {landmarks.map((pt, i) => (
      <Circle key={i} cx={pt.x} cy={pt.y} r={3} color="white" />
    ))}
  </Group>
</Canvas>
```

Install: `npx expo install @shopify/react-native-skia`

### Status / Alert Banner Component

A reusable status banner that shows the current driver state is the key UI element of this app:

```tsx
// components/DriverStatusBanner.tsx
import { View, Text, StyleSheet } from 'react-native';

type Status = 'active' | 'warning' | 'alert' | 'loading';

const STATUS_CONFIG: Record<Status, { label: string; color: string }> = {
  active:  { label: 'Alert',     color: '#22c55e' }, // green
  warning: { label: 'Warning',   color: '#f59e0b' }, // amber
  alert:   { label: 'DROWSY',    color: '#ef4444' }, // red
  loading: { label: 'Starting...', color: '#6b7280' }, // gray
};

export function DriverStatusBanner({ status }: { status: Status }) {
  const { label, color } = STATUS_CONFIG[status];
  return (
    <View style={[styles.banner, { backgroundColor: color + 'CC' }]}>
      <Text style={styles.text}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    position: 'absolute',
    top: 60,
    alignSelf: 'center',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 12,
  },
  text: {
    color: 'white',
    fontSize: 20,
    fontWeight: '800',
    letterSpacing: 1,
  },
});
```

### Dark Mode / Theming

The `app-example/` folder contains a complete theming setup with `useColorScheme` and `ThemedText`/`ThemedView` components. You can either adopt this pattern from the example into your main `app/`, or keep it simple with a single dark theme since this is a night-driving safety app.

---

## 12. What You Are Learning (Student Context)

This project touches a wide range of important computer science and software engineering concepts. Here is a map of what each part of the project teaches:

### Machine Learning Pipeline
- **Data preprocessing**: You are learning that raw data is never directly usable. The gap between "raw images" and "model-ready tensors" is where most of the real ML engineering work lives.
- **Convolutional Neural Networks (CNNs)**: You built one from scratch. Each layer type (Conv2D, MaxPooling, BatchNorm, Dropout) serves a specific purpose — understanding *why* each layer is there is more valuable than memorizing the architecture.
- **Train/validation split**: The separation of training data from validation data is foundational. Your oscillating `val_accuracy` is a real signal worth investigating.
- **Sigmoid vs. Softmax**: You used sigmoid for binary classification. For more than 2 classes you would switch to a softmax output layer and categorical crossentropy loss.
- **Model deployment gap**: The preprocessing parity problem you are solving is one of the most common (and painful) real-world ML engineering issues. Solving it yourself at this stage is genuinely valuable.

### Mobile Development
- **Frame processors and worklets**: The `'worklet'` directive in VisionCamera tells the React Native runtime to run that function on the UI thread, not the JS thread. This is a key concept in React Native performance.
- **Native modules vs. JS**: `react-native-fast-tflite` runs TFLite natively (C++), not in JavaScript. You are learning how React Native bridges between JS and native code, and why that bridge is expensive (which is why you should avoid crossing it every frame).
- **Permissions model**: Android and iOS require you to explicitly request sensitive permissions at runtime. This is a security/privacy design decision by the OS, not a framework quirk.
- **File-based routing**: Expo Router's file-system routing pattern is the same model used by Next.js, Remix, and SvelteKit — understanding it transfers to web development.

### Software Engineering
- **Separation of concerns**: Breaking logic into hooks (`useDrowsinessModel`, `usePredictionSmoothing`) versus keeping it all in one component file. Smaller, focused modules are easier to test and debug.
- **Configuration management**: Your `config.py` pattern of centralizing all paths and constants is a good software engineering habit. The same principle applies to the mobile side (a `constants/model.ts` file for `INPUT_SIZE`, threshold values, etc.).
- **Debugging across the stack**: You are debugging a pipeline that spans Python, C++ (TFLite runtime), and TypeScript. The `adb logcat` commands in your README and the debug image export system are examples of instrumentation — adding visibility into a system that would otherwise be a black box.

---

*Last updated: March 2026. Update this document as the project evolves.*
