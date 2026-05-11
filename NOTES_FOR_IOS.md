# iOS Support — What You Need to Change

Currently the app is **Android-only in practice** — no `ios/` directory exists, and the project has never been prebuilt or tested on iOS. However, all of your core native dependencies (VisionCamera, fast-tflite, worklets) do support iOS. Here's the full process:

## Prerequisites

- **A Mac** — iOS builds require macOS. There is no way around this.
- **Xcode** (latest stable, currently 16.x) — install from the Mac App Store
- **CocoaPods** — `sudo gem install cocoapods` or `brew install cocoapods`
- **Apple Developer Account** — Free for simulator testing, $99/year for device testing and App Store distribution
- **A physical iOS device** (optional but recommended) — drowsiness detection needs a real camera

## Step 1: Generate the iOS Native Project

```bash
cd mobile/sensor-app
npx expo prebuild --platform ios
```

This will:
- Create the `ios/` directory with an Xcode project
- Read your `app.json` plugins and configure native modules
- Generate a `Podfile` for CocoaPods dependency management
- Apply the camera permission from your existing `app.json` iOS config

## Step 2: Install CocoaPods Dependencies

```bash
cd ios
pod install
cd ..
```

This downloads and links the native iOS libraries for VisionCamera, TFLite, MLKit, worklets, etc.

## Step 3: Files You Need to Change

### `app.json` — Add iOS-specific configuration

Your current iOS config is minimal. You'll want to expand it:

```json
"ios": {
  "supportsTablet": true,
  "bundleIdentifier": "com.yourname.vigilare",
  "infoPlist": {
    "NSCameraUsageDescription": "Allow $(PRODUCT_NAME) to access your camera to detect drowsiness",
    "NSMicrophoneUsageDescription": "Allow $(PRODUCT_NAME) to access your microphone"
  }
}
```

**Required changes:**
- Add a `bundleIdentifier` (e.g., `com.yourname.vigilare`) — this is the iOS equivalent of Android's package name. Required for device builds and App Store.
- VisionCamera may require microphone permission even if you don't use audio — add `NSMicrophoneUsageDescription` to be safe.

### `app.json` plugins — Add iOS build properties

```json
[
  "expo-build-properties",
  {
    "android": {
      "minSdkVersion": 26
    },
    "ios": {
      "deploymentTarget": "15.0"
    }
  }
]
```

The `react-native-fast-tflite` and `react-native-vision-camera` libraries require iOS 13+ minimum; setting 15.0 gives comfortable margin.

### `metro.config.js` — No changes needed

Your `.tflite` asset extension registration already works cross-platform.

### `app/sensors/drowsiness.tsx` — Likely no changes needed

VisionCamera, the face detector plugin, fast-tflite, and the resize plugin all use the same JavaScript API on both platforms. The frame processor code should work as-is. However, you should test for:
- Camera orientation differences (the 90-degree rotation you apply may behave differently on iOS)
- Performance differences in the worklet frame processor
- Haptic feedback API differences (expo-haptics works on both but feels different)

### Potential platform-specific adjustments

If you encounter issues, you may need platform checks in `drowsiness.tsx`:

```typescript
import { Platform } from 'react-native';

// Example: if rotation differs between platforms
const rotation = Platform.OS === 'ios' ? 0 : -90;
```

## Step 4: Build and Run on iOS

**Simulator:**
```bash
npx expo run:ios
```

**Physical device:**
```bash
npx expo run:ios --device
```

For physical devices, you need to:
1. Open `ios/sensorapp.xcworkspace` in Xcode
2. Go to **Signing & Capabilities**
3. Select your Apple Developer team
4. Select your connected device as the build target
5. Build and run (or use `npx expo run:ios --device`)

## Summary of Files to Touch for iOS

| File | Change |
|------|--------|
| `app.json` | Add `bundleIdentifier`, microphone permission, iOS deployment target |
| `ios/` (generated) | Created by `npx expo prebuild --platform ios` |
| `ios/Podfile` (generated) | Auto-generated; may need manual tweaks for TFLite pod config |
| `drowsiness.tsx` | Test and adjust if camera orientation or haptics differ |
| No other source files should need changes | The JS/TS code is cross-platform |

---

## Step 5: iOS Deployment (App Store)

### Prerequisites

- **Mac with Xcode** — Required for all iOS builds
- **Apple Developer Program membership** — $99/year at [developer.apple.com](https://developer.apple.com)
- **Apple Developer certificates and provisioning profiles**

### Step 1: Generate the iOS Project (if not done already)

```bash
npx expo prebuild --platform ios
cd ios && pod install && cd ..
```

### Step 2: Configure Signing in Xcode

1. Open `ios/sensorapp.xcworkspace` in Xcode
2. Select the project in the navigator -> select your target
3. Go to **Signing & Capabilities** tab
4. Check **Automatically manage signing**
5. Select your **Team** (your Apple Developer account)
6. Xcode will create provisioning profiles and certificates automatically

### Step 3: Set the Bundle Identifier

Must match what you set in `app.json` under `ios.bundleIdentifier` (e.g., `com.yourname.vigilare`).

### Step 4: Create an App Store Connect Listing

1. Go to [App Store Connect](https://appstoreconnect.apple.com)
2. Click **My Apps** -> **+** -> **New App**
3. Fill in:
   - Platform: iOS
   - App name
   - Bundle ID (must match your Xcode project)
   - SKU (any unique string)
   - Primary language
4. Fill in the app information:
   - Description, keywords, screenshots (6.7" and 5.5" required minimum)
   - Privacy policy URL (required)
   - App category
   - Age rating questionnaire

### Step 5: Build and Upload

**Option A: Xcode**
1. In Xcode, set the scheme to **Release**
2. Select **Any iOS Device** as the destination
3. **Product** -> **Archive**
4. In the **Organizer** window, click **Distribute App**
5. Choose **App Store Connect** -> **Upload**
6. Follow the prompts (Xcode handles signing)

**Option B: Command line**
```bash
npx expo run:ios --configuration Release --device
# Or for archive:
cd ios
xcodebuild -workspace sensorapp.xcworkspace -scheme sensorapp -configuration Release -archivePath build/sensorapp.xcarchive archive
xcodebuild -exportArchive -archivePath build/sensorapp.xcarchive -exportOptionsPlist ExportOptions.plist -exportPath build/output
```

**Option C: EAS Build (Recommended)**
```bash
eas build --platform ios --profile production
eas submit --platform ios
```

EAS handles certificates, provisioning, and uploading. It builds in the cloud so you technically don't need a Mac for the build step (but you still need one for debugging).

### Step 6: TestFlight

After uploading, your build appears in App Store Connect under **TestFlight**:
- Add internal testers (up to 100, no review needed)
- Add external testers (up to 10,000, requires brief Apple review)
- Test thoroughly before submitting for App Store review

### Step 7: App Store Review

1. In App Store Connect, go to your app -> **App Store** tab
2. Select the build from TestFlight
3. Fill in all required metadata
4. Click **Submit for Review**
5. Apple reviews the app (typically 24-48 hours, can take longer)
6. Common rejection reasons to watch for:
   - Camera permission usage must clearly explain **why** (drowsiness detection)
   - App must work as described
   - Privacy policy must be accurate
   - No placeholder content

### iOS Build Size

iOS app sizes are typically similar to Android. Using App Store distribution, Apple applies **App Thinning** automatically — only the relevant device architecture (arm64) is delivered. Your iOS app will likely be 60-90 MB on-device (compared to the 186 MB universal Android APK).