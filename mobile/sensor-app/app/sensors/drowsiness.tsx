import { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { Camera, useCameraDevice, useCameraPermission, useFrameProcessor } from 'react-native-vision-camera';
import { Face, useFaceDetector } from 'react-native-vision-camera-face-detector';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { Worklets } from 'react-native-worklets-core';

const MODEL_INPUT_SIZE = 145;
const LANDMARK_DOT_RADIUS = 2;
const PROCESS_EVERY_N_FRAMES = 3;

type UiPrediction = {
  label: string;
  score: number;
  hasFace: boolean;
};

function clamp(v: number, min: number, max: number): number {
  'worklet';
  return Math.max(min, Math.min(max, v));
}

function stampDot(
  input: Float32Array,
  x: number,
  y: number,
  radius: number,
  size: number
): void {
  'worklet';
  const minY = clamp(y - radius, 0, size - 1);
  const maxY = clamp(y + radius, 0, size - 1);
  const minX = clamp(x - radius, 0, size - 1);
  const maxX = clamp(x + radius, 0, size - 1);

  for (let yy = minY; yy <= maxY; yy++) {
    for (let xx = minX; xx <= maxX; xx++) {
      const offset = (yy * size + xx) * 3;
      input[offset] = 1.0;
      input[offset + 1] = 1.0;
      input[offset + 2] = 1.0;
    }
  }
}

export default function DrowsinessScreen() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');
  const modelPlugin = useTensorflowModel(
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    require('../../assets/ml/drowsiness_cnn.tflite')
  );
  const { model, state: modelState } = modelPlugin;
  const { resize } = useResizePlugin();

  const { detectFaces, stopListeners } = useFaceDetector({
    performanceMode: 'fast',
    landmarkMode: 'all',
    contourMode: 'all',
    classificationMode: 'none',
    minFaceSize: 0.15,
    trackingEnabled: false,
    cameraFacing: 'front',
    autoMode: false,
  });

  const [prediction, setPrediction] = useState<UiPrediction>({
    label: 'Initializing...',
    score: 0,
    hasFace: false,
  });

  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission, requestPermission]);

  useEffect(() => {
    return () => {
      stopListeners();
    };
  }, [stopListeners]);

  const updatePredictionOnJs = useMemo(
    () =>
      Worklets.createRunOnJS((next: UiPrediction) => {
        setPrediction(next);
      }),
    []
  );

  const frameCounter = useMemo(() => Worklets.createSharedValue(0), []);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';

      if (model == null) return;

      frameCounter.value += 1;
      if (frameCounter.value % PROCESS_EVERY_N_FRAMES !== 0) {
        return;
      }

      const faces = detectFaces(frame) as Face[];
      if (faces.length === 0) {
        updatePredictionOnJs({ label: 'No Face', score: 0, hasFace: false });
        return;
      }

      const face = faces[0];
      const bx = clamp(face.bounds.x, 0, frame.width - 1);
      const by = clamp(face.bounds.y, 0, frame.height - 1);
      const bw = clamp(face.bounds.width, 1, frame.width - bx);
      const bh = clamp(face.bounds.height, 1, frame.height - by);

      const input = resize(frame, {
        crop: {
          x: bx,
          y: by,
          width: bw,
          height: bh,
        },
        scale: {
          width: MODEL_INPUT_SIZE,
          height: MODEL_INPUT_SIZE,
        },
        pixelFormat: 'rgb',
        dataType: 'float32',
      });

      // Stamp detected facial landmarks onto the cropped tensor to match the training
      // pattern where landmark points are drawn on the face crop.
      const landmarks = face.landmarks;
      if (landmarks) {
        const points = [
          landmarks.LEFT_EYE,
          landmarks.RIGHT_EYE,
          landmarks.NOSE_BASE,
          landmarks.MOUTH_LEFT,
          landmarks.MOUTH_RIGHT,
          landmarks.MOUTH_BOTTOM,
          landmarks.LEFT_CHEEK,
          landmarks.RIGHT_CHEEK,
          landmarks.LEFT_EAR,
          landmarks.RIGHT_EAR,
        ];

        for (const p of points) {
          if (!p) continue;
          const lx = ((p.x - bx) / bw) * MODEL_INPUT_SIZE;
          const ly = ((p.y - by) / bh) * MODEL_INPUT_SIZE;
          const px = clamp(Math.round(lx), 0, MODEL_INPUT_SIZE - 1);
          const py = clamp(Math.round(ly), 0, MODEL_INPUT_SIZE - 1);
          stampDot(input, px, py, LANDMARK_DOT_RADIUS, MODEL_INPUT_SIZE);
        }
      }

      const outputs = model.runSync([input]);
      const score = Number(outputs[0][0] ?? 0);

      // Current class mapping in training:
      // 0 = Fatigue Subjects, 1 = Active Subjects
      const label = score >= 0.5 ? 'Active Subjects' : 'Fatigue Subjects';
      updatePredictionOnJs({ label, score, hasFace: true });
    },
    [model, detectFaces, resize, updatePredictionOnJs, frameCounter]
  );

  if (!hasPermission) {
    return (
      <View style={styles.centered}>
        <Text style={styles.statusText}>Camera permission required.</Text>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.centered}>
        <Text style={styles.statusText}>No front camera device found.</Text>
      </View>
    );
  }

  if (modelState === 'loading') {
    return (
      <View style={styles.centered}>
        <ActivityIndicator />
        <Text style={styles.statusText}>Loading TensorFlow Lite model...</Text>
      </View>
    );
  }

  if (modelState === 'error') {
    return (
      <View style={styles.centered}>
        <Text style={styles.statusText}>Model load failed.</Text>
        <Text style={styles.errorText}>
          {String(modelPlugin.state === 'error' ? modelPlugin.error?.message : 'Unknown error')}
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        pixelFormat="yuv"
        frameProcessor={frameProcessor}
      />

      <View style={styles.badge}>
        <Text style={styles.badgeTitle}>Driver State</Text>
        <Text style={styles.badgeLabel}>{prediction.label}</Text>
        <Text style={styles.badgeScore}>score: {prediction.score.toFixed(3)}</Text>
        <Text style={styles.badgeMeta}>{prediction.hasFace ? 'face: detected' : 'face: none'}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#111',
    padding: 16,
  },
  statusText: {
    color: '#FFF',
    marginTop: 8,
    textAlign: 'center',
  },
  errorText: {
    color: '#FCA5A5',
    marginTop: 8,
    textAlign: 'center',
  },
  badge: {
    position: 'absolute',
    top: 56,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 12,
    borderRadius: 10,
  },
  badgeTitle: {
    color: '#9CA3AF',
    fontSize: 12,
    marginBottom: 2,
  },
  badgeLabel: {
    color: '#FFF',
    fontSize: 20,
    fontWeight: '700',
  },
  badgeScore: {
    color: '#D1D5DB',
    fontSize: 13,
    marginTop: 2,
  },
  badgeMeta: {
    color: '#D1D5DB',
    fontSize: 13,
  },
});
