import React, { useEffect, useMemo, useState, useRef, useCallback } from 'react';
import {
    ActivityIndicator,
    Pressable,
    StyleSheet,
    Text,
    View,
    Dimensions,
    useWindowDimensions,
    Animated,
} from 'react-native';
import * as Haptics from 'expo-haptics';
import {
    loadTensorflowModel,
    useTensorflowModel,
} from 'react-native-fast-tflite';
import {
    Camera,
    useCameraDevice,
    useCameraPermission,
    useFrameProcessor,
} from 'react-native-vision-camera';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { Worklets } from 'react-native-worklets-core';
import {
    Face,
    useFaceDetector,
} from 'react-native-vision-camera-face-detector';
import Svg, { Circle } from 'react-native-svg';
import { scheduleOnRN } from 'react-native-worklets';

// Item 2: Load class names from labels.json so mobile stays in sync with training output.
// labels.class_names[0] = 'Fatigue Subjects' (sigmoid score near 0)
// labels.class_names[1] = 'Active Subjects'  (sigmoid score near 1)
import labels from '../../assets/ml/labels.json';

// DEBUG
import { Directory, File, Paths } from 'expo-file-system';

type UiPrediction = {
    label: string;
    score: number;
    hasFace: boolean;
};
const PROCESS_EVERY_N_FRAMES = 5;
const MODEL_INPUT_SIZE = 145;

// how many recent scores to average together before making a drowsy/active decision
// higher = smoother but slower to react, lower = more responsive but flickery
const SMOOTHING_WINDOW_SIZE = 10;

// decision boundary for the smoothed score (sigmoid output averaged over window)
// scores below this = Fatigue, above = Active
// default sigmoid midpoint is 0.5 but can be tuned if model leans one way
const DROWSY_THRESHOLD = 0.45;

// Alert escalation tiers — trigger after sustained drowsiness (Fatigue label) for N seconds
// These only affect visual/haptic UI and do not touch inference logic.
const ALERT_WARNING_SEC = 3;   // tier 1: badge turns red
const ALERT_CAUTION_SEC = 5;   // tier 2: haptic vibration pulses
const ALERT_DANGER_SEC = 8;    // tier 3: full-screen overlay + heavy vibration

// Item 3: Dot radii matched to training preprocess.py draw_and_save_face_mesh().
// Training draws radius=2 for eye landmark indices and radius=1 for all others.
// We mirror that split here using the face detector's contour key as a proxy.
const EYE_CONTOUR_RADIUS = 2; // left/right eye outlines
const LANDMARK_DOT_RADIUS = 1; // all other contour groups (brows, lips, nose, oval)

// Item 3: Which contour keys map to eye regions — these get the larger radius.
function isEyeContour(key: string): boolean {
    'worklet';
    return key === 'LEFT_EYE' || key === 'RIGHT_EYE';
}

// DEBUG
const DEBUG_SAVE_EVERY_N = 60; // save 1 every 60 processed frames

// //Map landmark points to cropped image
function mapLandmarkToCrop(
    px: number,
    py: number,
    bx: number,
    by: number,
    bw: number,
    bh: number,
) {
    'worklet';

    // px/py and bx/by/bw/bh must already be in the same rotated frame space.
    const x = ((px - bx) / bw) * (MODEL_INPUT_SIZE - 1);
    const y = ((py - by) / bh) * (MODEL_INPUT_SIZE - 1);

    const cx = Math.max(0, Math.min(MODEL_INPUT_SIZE - 1, Math.round(x)));
    const cy = Math.max(0, Math.min(MODEL_INPUT_SIZE - 1, Math.round(y)));
    return { x: cx, y: cy };
}

// Rotate a square RGB Float32Array buffer 90° counter-clockwise in-place so that
// the landscape-right camera crop becomes an upright face matching the training data.
function rotateBuffer90CCW(input: Float32Array, size: number): Float32Array {
    'worklet';
    const out = new Float32Array(input.length);
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            // CCW: newPixel(x, y) = oldPixel(size-1-y, x)
            const srcOffset = (x * size + (size - 1 - y)) * 3;
            const dstOffset = (y * size + x) * 3;
            out[dstOffset] = input[srcOffset];
            out[dstOffset + 1] = input[srcOffset + 1];
            out[dstOffset + 2] = input[srcOffset + 2];
        }
    }
    return out;
}

//Draw dot at landmark position
function stampDot(
    input: Float32Array,
    x: number,
    y: number,
    radius: number,
    size: number,
) {
    'worklet'; // needed if inside frame processor

    const minY = Math.max(y - radius, 0);
    const maxY = Math.min(y + radius, size - 1);
    const minX = Math.max(x - radius, 0);
    const maxX = Math.min(x + radius, size - 1);

    for (let yy = minY; yy <= maxY; yy++) {
        for (let xx = minX; xx <= maxX; xx++) {
            const offset = (yy * size + xx) * 3; // 3 channels: RGB

            // Item 3: 255.0 is correct here — the TFLite model includes a Rescaling(1/255)
            // layer as its first layer (see train.py), so it expects raw [0, 255] float32
            // input, not [0, 1]. Setting 255.0 produces a white dot after internal rescaling.
            input[offset] = 255.0; // R
            input[offset + 1] = 255.0; // G
            input[offset + 2] = 255.0; // B
        }
    }
}

export default function DrowsinessScreen() {
    const frameCounter = useMemo(() => Worklets.createSharedValue(0), []); //shared value across frames
    const [landmarks, setLandmarks] = useState<{ x: number; y: number }[]>([]); //points from face detector
    const [overlaySize, setOverlaySize] = useState({ width: 0, height: 0 }); //View dimensions
    const [debugEnabled, setDebugEnabled] = useState(false);
    const debugEnabledShared = useMemo(() => Worklets.createSharedValue(false), []);

    const modelPlugin = useTensorflowModel(
        require('../../assets/ml/drowsiness_cnn.tflite'),
    );
    const model =
        modelPlugin.state === 'loaded' ? modelPlugin.model : undefined; //Only run inference when state is loaded

    const [prediction, setPrediction] = useState<UiPrediction>({
        label: 'Initializing...',
        score: 0,
        hasFace: false,
    });

    const { hasPermission, requestPermission } = useCameraPermission();
    const device = useCameraDevice('front');
    const { resize } = useResizePlugin(); //function for resizing vision frame

    //Declare/define face detector
    const { detectFaces } = useFaceDetector({
        performanceMode: 'fast', // can be 'accurate' for better detection
        landmarkMode: 'all', // we need landmarks
        contourMode: 'all', // optional, if using full face mesh
        classificationMode: 'none',
        minFaceSize: 0.15, // ignore tiny faces
        trackingEnabled: false,
        // windowHeight:
        // windowWidth:
    });

    // Sync debug toggle to worklet shared value
    useEffect(() => {
        debugEnabledShared.value = debugEnabled;
    }, [debugEnabled]);

    // If hasPermission is False, requestPermission
    useEffect(() => {
        if (!hasPermission) {
            requestPermission();
        }
    }, [hasPermission, requestPermission]);

    //use instead of scheduleOnRN
    const updateLandmarks = Worklets.createRunOnJS(
        (points: { x: number; y: number }[]) => {
            setLandmarks(points);
            // console.log(points);
        },
    );

    // ring buffer holding the last N raw sigmoid scores from the model
    // used to compute a rolling average so one weird frame cant flip the label
    const scoreBuffer = useRef<number[]>([]);

    // --- Alert escalation state ---
    // Tracks when continuous drowsiness started so we can escalate alerts over time
    const drowsyStartRef = useRef<number | null>(null);
    const [alertTier, setAlertTier] = useState<0 | 1 | 2 | 3>(0);
    const dangerOpacity = useRef(new Animated.Value(0)).current;
    const hapticIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

    // Pulsing animation for the danger overlay (tier 3)
    useEffect(() => {
        if (alertTier === 3) {
            const pulse = Animated.loop(
                Animated.sequence([
                    Animated.timing(dangerOpacity, { toValue: 0.45, duration: 600, useNativeDriver: true }),
                    Animated.timing(dangerOpacity, { toValue: 0.15, duration: 600, useNativeDriver: true }),
                ]),
            );
            pulse.start();
            return () => pulse.stop();
        } else {
            dangerOpacity.setValue(0);
        }
    }, [alertTier]);

    // Haptic feedback for tiers 2 and 3
    useEffect(() => {
        if (hapticIntervalRef.current) {
            clearInterval(hapticIntervalRef.current);
            hapticIntervalRef.current = null;
        }

        if (alertTier === 2) {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
            hapticIntervalRef.current = setInterval(() => {
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
            }, 2000);
        } else if (alertTier === 3) {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
            hapticIntervalRef.current = setInterval(() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
            }, 1000);
        }

        return () => {
            if (hapticIntervalRef.current) {
                clearInterval(hapticIntervalRef.current);
                hapticIntervalRef.current = null;
            }
        };
    }, [alertTier]);

    // Evaluate alert tier based on how long prediction has been "Fatigue"
    const evaluateAlertTier = useCallback((label: string) => {
        const now = Date.now();

        if (label === labels.class_names[0]) {
            // Fatigue detected
            if (drowsyStartRef.current === null) {
                drowsyStartRef.current = now;
            }
            const elapsed = (now - drowsyStartRef.current) / 1000;

            if (elapsed >= ALERT_DANGER_SEC) {
                setAlertTier(3);
            } else if (elapsed >= ALERT_CAUTION_SEC) {
                setAlertTier(2);
            } else if (elapsed >= ALERT_WARNING_SEC) {
                setAlertTier(1);
            } else {
                setAlertTier(0);
            }
        } else {
            // Active or no face — reset
            drowsyStartRef.current = null;
            setAlertTier(0);
        }
    }, []);

    const updatePredictionOnJs = useMemo(
        () =>
            Worklets.createRunOnJS((rawScore: number, hasFace: boolean) => {
                if (!hasFace) {
                    // no face = clear the buffer so stale scores dont carry over
                    scoreBuffer.current = [];
                    setPrediction({ label: 'No Face', score: 0, hasFace: false });
                    evaluateAlertTier('No Face');
                    return;
                }

                // push new score into ring buffer, drop oldest if full
                const buf = scoreBuffer.current;
                buf.push(rawScore);
                if (buf.length > SMOOTHING_WINDOW_SIZE) {
                    buf.shift(); // drop oldest score
                }

                // average all scores in the buffer for a smoothed prediction
                const smoothed = buf.reduce((sum, s) => sum + s, 0) / buf.length;

                // classify using the tunable threshold instead of hard 0.5
                const classIndex = smoothed >= DROWSY_THRESHOLD ? 1 : 0;
                const label = labels.class_names[classIndex] ?? 'Unknown';

                setPrediction({ label, score: smoothed, hasFace: true });
                evaluateAlertTier(label);
            }),
        [evaluateAlertTier],
    );

    // DEBUG
    const savedCountRef = useRef(0);
    const saveInputOnJs = useMemo(
        () =>
            Worklets.createRunOnJS(async (flat: number[]) => {
                savedCountRef.current += 1;

                const debugDir = new Directory(Paths.document, 'debug_inputs');
                debugDir.create({ idempotent: true, intermediates: true });

                // const dir = `${FileSystem.documentDirectory}debug_inputs/`;
                // await FileSystem.makeDirectoryAsync(dir, { intermediates: true });

                const payload = {
                    width: 145,
                    height: 145,
                    channels: 3,
                    data: flat,
                };

                const outFile = new File(debugDir, `input_${Date.now()}.json`);
                outFile.create({ overwrite: true });
                outFile.write(JSON.stringify(payload));
                console.log(savedCountRef);
                // const path = `${dir}input_${Date.now()}.json`;
                // await FileSystem.writeAsStringAsync(path, JSON.stringify(payload));
                // console.log('saved tensor:', path);
            }),
        [],
    );

    const aFaceW = useMemo(() => Worklets.createSharedValue(0), []); //useMemo doesn't refresh value during re-renders
    const aFaceH = useMemo(() => Worklets.createSharedValue(0), []);
    const aFaceX = useMemo(() => Worklets.createSharedValue(0), []);
    const aFaceY = useMemo(() => Worklets.createSharedValue(0), []);
    // const aRot = useMemo(() => Worklets.createSharedValue(0), []);

    //declare frame processor function passed to camera
    const frameProcessor = useFrameProcessor(
        // Item 3: This processor mirrors the training preprocessing pipeline from preprocess.py:
        //   Step 1 — detect face bounding box (Haar cascade in training, MLKit here)
        //   Step 2 — crop frame to face region
        //   Step 3 — draw white dots at contour landmark positions onto the cropped buffer
        //   Step 4 — resize to MODEL_INPUT_SIZE x MODEL_INPUT_SIZE
        //   Step 5 — feed float32 buffer to TFLite model
        // {"height": 480, "width": 640} frame size
        (frame) => {
            'worklet';

            //If model is null exit
            if (model == null) return;

            //Skip frames to make faster
            frameCounter.value += 1;
            if (frameCounter.value % PROCESS_EVERY_N_FRAMES !== 0) return;

            // Step 1: detect face bounding box using MLKit via face detector plugin
            const faces = detectFaces(frame) as Face[];

            // no face found, clear smoothing buffer and update UI
            if (!faces || faces.length === 0) {
                updateLandmarks([]);
                updatePredictionOnJs(0, false);
                return;
            }

            // DEBUG
            // const fullInput = resize(frame, {
            //     scale: { width: 145, height: 145 },
            //     pixelFormat: 'rgb',
            //     dataType: 'float32',
            // });
            // if (frameCounter.value % DEBUG_SAVE_EVERY_N === 0) {
            //     saveInputOnJs(Array.from(fullInput)); // input is Float32Array
            // }

            // each face has bounds(x, y, w, h) and landmarks(LEFT_EYE, ...)
            const face = faces[0];

            // Step 2: Convert face bounds from portrait detector space to landscape-right
            // frame buffer space (640x480). The front camera frame arrives rotated 90°
            // so width/height and x/y axes are swapped relative to the screen.
            aFaceW.value = face.bounds.height;
            aFaceH.value = face.bounds.width;
            aFaceX.value = Math.max(
                0,
                Math.min(
                    frame.width - 1,
                    frame.width - (face.bounds.y + face.bounds.height),
                ),
            );
            aFaceY.value = Math.max(
                0,
                Math.min(frame.height - 1, face.bounds.x),
            );

            // Step 4: Crop and resize frame to MODEL_INPUT_SIZE using the resize plugin.
            // pixelFormat 'rgb' and dataType 'float32' give us a Float32Array in [0, 1] range.
            // We scale to [0, 255] after rotation so the model's Rescaling(1/255) layer
            // receives the same value distribution as training data.
            const cropped = resize(frame, {
                crop: {
                    x: aFaceX.value,
                    y: aFaceY.value,
                    width: aFaceW.value,
                    height: aFaceH.value,
                },
                scale: { width: MODEL_INPUT_SIZE, height: MODEL_INPUT_SIZE },
                pixelFormat: 'rgb',
                dataType: 'float32',
            });

            // Rotate 90° CCW so the landscape-right crop becomes an upright face
            // matching the orientation of the training images.
            const input = rotateBuffer90CCW(cropped, MODEL_INPUT_SIZE);

            // resize plugin gives us floats in [0, 1] but the model has a Rescaling(1/255)
            // layer baked in (see train.py) so it expects [0, 255] input like training data.
            // without this the model was getting near-zero values and always predicting drowsy
            for (let i = 0; i < input.length; i++) {
                input[i] = input[i] * 255.0;
            }

            // PREDICT FIX ATTEMOT
            // const face = faces[0];
            // const mirrorX = true; // front cam
            // const o = String(frame.orientation);

            // // 1) Convert face bounds to frame coords by transforming all 4 corners
            // const b = face.bounds;
            // const c1 = toFramePoint({ x: b.x, y: b.y }, frame.width, frame.height, o, mirrorX);
            // const c2 = toFramePoint({ x: b.x + b.width, y: b.y }, frame.width, frame.height, o, mirrorX);
            // const c3 = toFramePoint({ x: b.x, y: b.y + b.height }, frame.width, frame.height, o, mirrorX);
            // const c4 = toFramePoint({ x: b.x + b.width, y: b.y + b.height }, frame.width, frame.height, o, mirrorX);
            // const crop = rectFromCorners([c1, c2, c3, c4], frame.width, frame.height);

            // // 2) Crop frame in frame-buffer space
            // const input = resize(frame, {
            //     crop,
            //     scale: { width: MODEL_INPUT_SIZE, height: MODEL_INPUT_SIZE },
            //     pixelFormat: 'rgb',
            //     dataType: 'float32',
            // });

            // DEBUG
            // if (frameCounter.value % DEBUG_SAVE_EVERY_N === 0) {
            //     // console.log(aFaceW.value);
            //     // console.log(aFaceH.value);
            //     // console.log(aFaceX.value);
            //     // console.log(aFaceY.value);
            //     saveInputOnJs(Array.from(input)); // input is Float32Array
            // }

            // Step 3: Draw white dots at each contour landmark position onto the input buffer.
            // This mirrors preprocess.py's draw_and_save_face_mesh() which uses white cv2 circles.
            // The face detector returns MLKit-style contours (a subset of the full 478-point mesh)
            // which is why we retrain with the same subset — see DEVELOPMENT_GUIDE.md §3.
            const overlayPoints: { x: number; y: number }[] = [];
            const contours = face.contours as any;

            const contourKeys = [
                'FACE',
                'LEFT_EYEBROW_TOP',
                'LEFT_EYEBROW_BOTTOM',
                'RIGHT_EYEBROW_TOP',
                'RIGHT_EYEBROW_BOTTOM',
                'LEFT_EYE',
                'RIGHT_EYE',
                'UPPER_LIP_TOP',
                'UPPER_LIP_BOTTOM',
                'LOWER_LIP_TOP',
                'LOWER_LIP_BOTTOM',
                'NOSE_BRIDGE',
                'NOSE_BOTTOM',
                'LEFT_CHEEK',
                'RIGHT_CHEEK',
            ];

            for (const key of contourKeys) {
                const contour = contours[key];
                if (!contour) continue;

                // Item 3: Match training radii — preprocess.py uses radius=2 for eye indices
                // and radius=1 for all others. We use contour key as the eye proxy.
                const dotRadius = isEyeContour(key)
                    ? EYE_CONTOUR_RADIUS
                    : LANDMARK_DOT_RADIUS;

                for (const p of contour) {
                    const screenX = p.x;
                    const screenY = p.y;

                    // Rotate landmark into the same landscape-right frame space used by the crop.
                    const pFrameX = Math.max(
                        0,
                        Math.min(frame.width - 1, frame.width - screenY),
                    );
                    const pFrameY = Math.max(
                        0,
                        Math.min(frame.height - 1, screenX),
                    );

                    const mapped = mapLandmarkToCrop(
                        pFrameX,
                        pFrameY,
                        aFaceX.value,
                        aFaceY.value,
                        aFaceW.value,
                        aFaceH.value,
                    );
                    // Apply the same 90° CCW rotation to landmark coordinates
                    const x = mapped.y;
                    const y = MODEL_INPUT_SIZE - 1 - mapped.x;
                    stampDot(input, x, y, dotRadius, MODEL_INPUT_SIZE);

                    overlayPoints.push({
                        x: -(screenX * 1.4) + overlaySize.width + 125,
                        y: screenY * 1.25 + 5,
                        // {"height": 480, "width": 640} frame size
                    });
                }
            }

            updateLandmarks(overlayPoints);

            // Save debug input when toggle is enabled
            if (debugEnabledShared.value && frameCounter.value % DEBUG_SAVE_EVERY_N === 0) {
                saveInputOnJs(Array.from(input));
            }

            // Step 5: Run TFLite inference on the cropped, dot-stamped input buffer
            const outputs = model.runSync([input]) as unknown[];
            const out0 = outputs?.[0] as number[] | undefined;
            const score = Number(out0?.[0] ?? 0);

            // send the raw score to JS side where it gets averaged with recent scores
            // smoothing + threshold logic lives in updatePredictionOnJs callback above
            updatePredictionOnJs(score, true);
        },
        [detectFaces, model, debugEnabledShared],
    );

    /*
     * ERROR STATE RENDERING
     */
    if (!hasPermission) {
        return (
            <View style={styles.centered}>
                <Text style={styles.statusText}>
                    Camera permission required.
                </Text>
            </View>
        );
    }

    if (!device) {
        return (
            <View style={styles.centered}>
                <Text style={styles.statusText}>No front camera found.</Text>
            </View>
        );
    }

    if (modelPlugin.state === 'loading') {
        return (
            <View style={styles.centered}>
                <ActivityIndicator color="#93C5FD" size="large" />
                <Text style={styles.statusText}>
                    Loading model...
                </Text>
            </View>
        );
    }

    if (modelPlugin.state === 'error') {
        return (
            <View style={styles.centered}>
                <Text style={styles.statusText}>Model load failed.</Text>
                <Text style={styles.errorText}>
                    {String(modelPlugin.error?.message ?? 'Unknown error')}
                </Text>
            </View>
        );
    }

    return (
        // <View style={StyleSheet.absoluteFill}>
        <>
            <View
                style={StyleSheet.absoluteFill}
                onLayout={(e) => {
                    const { width, height } = e.nativeEvent.layout;
                    setOverlaySize({ width, height });
                }}
            >
                <Camera
                    frameProcessor={frameProcessor}
                    style={StyleSheet.absoluteFill}
                    device={device}
                    isActive={true}
                />

                <Svg style={StyleSheet.absoluteFill} width="100%" height="100%">
                    {landmarks.map((p, i) => (
                        <Circle key={i} cx={p.x} cy={p.y} r={1.5} fill="rgba(173,216,230,0.85)" />
                    ))}
                </Svg>

                {/* Tier 3 danger: full-screen pulsing red overlay */}
                {alertTier === 3 && (
                    <Animated.View
                        pointerEvents="none"
                        style={[StyleSheet.absoluteFill, styles.dangerOverlay, { opacity: dangerOpacity }]}
                    />
                )}

                <View style={[
                    styles.badge,
                    alertTier >= 1 && styles.badgeWarning,
                    alertTier >= 2 && styles.badgeCaution,
                    alertTier >= 3 && styles.badgeDanger,
                ]}>
                    <Text style={styles.badgeTitle}>Driver State</Text>
                    <Text style={[
                        styles.badgeLabel,
                        { color: prediction.label === 'Fatigue Subjects'
                            ? (alertTier >= 2 ? '#FEF08A' : '#F87171')
                            : '#93C5FD' }
                    ]}>{prediction.label}</Text>
                    <View style={styles.badgeDivider} />
                    <Text style={styles.badgeScore}>
                        {prediction.score.toFixed(3)}
                    </Text>
                    <Text style={styles.badgeMeta}>
                        {prediction.hasFace ? 'Face Detected (smoothed)' : 'No Face'}
                    </Text>
                    {alertTier >= 1 && (
                        <Text style={styles.alertText}>
                            {alertTier === 1 && '⚠ Drowsiness Warning'}
                            {alertTier === 2 && '⚠ Caution — Stay Alert!'}
                            {alertTier === 3 && '🚨 DANGER — Pull Over!'}
                        </Text>
                    )}
                </View>

                <Pressable
                    onPress={() => setDebugEnabled((v) => !v)}
                    android_ripple={{ color: 'rgba(147, 197, 253, 0.3)' }}
                    style={[styles.debugButton, debugEnabled && styles.debugButtonActive]}
                >
                    <Text style={[styles.debugButtonText, debugEnabled && styles.debugButtonTextActive]}>
                        {debugEnabled ? 'Debug: ON' : 'Debug: OFF'}
                    </Text>
                </Pressable>
            </View>
        </>
    );
}

const styles = StyleSheet.create({
    centered: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#0F172A',
        padding: 24,
    },
    statusText: {
        color: '#BFDBFE',
        marginTop: 12,
        textAlign: 'center',
        fontSize: 15,
        fontWeight: '500',
    },
    errorText: {
        color: '#FCA5A5',
        marginTop: 8,
        textAlign: 'center',
        fontSize: 13,
    },
    badge: {
        position: 'absolute',
        top: 56,
        left: 16,
        right: 16,
        backgroundColor: 'rgba(15, 23, 42, 0.82)',
        padding: 16,
        borderRadius: 16,
        borderWidth: 1,
        borderColor: 'rgba(147, 197, 253, 0.2)',
    },
    badgeTitle: {
        color: '#94A3B8',
        fontSize: 11,
        fontWeight: '600',
        letterSpacing: 1.2,
        textTransform: 'uppercase',
        marginBottom: 4,
    },
    badgeLabel: {
        color: '#93C5FD',
        fontSize: 22,
        fontWeight: '700',
    },
    badgeDivider: {
        height: 1,
        backgroundColor: 'rgba(147, 197, 253, 0.15)',
        marginVertical: 8,
    },
    badgeScore: {
        color: '#CBD5E1',
        fontSize: 28,
        fontWeight: '300',
        fontVariant: ['tabular-nums'],
    },
    badgeMeta: {
        color: '#64748B',
        fontSize: 12,
        fontWeight: '500',
        marginTop: 4,
    },
    debugButton: {
        position: 'absolute',
        bottom: 32,
        alignSelf: 'center',
        paddingVertical: 10,
        paddingHorizontal: 20,
        backgroundColor: 'rgba(15, 23, 42, 0.82)',
        borderRadius: 10,
        borderWidth: 1,
        borderColor: 'rgba(147, 197, 253, 0.25)',
    },
    debugButtonActive: {
        backgroundColor: 'rgba(59, 130, 246, 0.25)',
        borderColor: '#3B82F6',
    },
    debugButtonText: {
        color: '#64748B',
        fontSize: 13,
        fontWeight: '600',
    },
    debugButtonTextActive: {
        color: '#93C5FD',
    },
    // Alert escalation styles
    dangerOverlay: {
        backgroundColor: '#DC2626',
    },
    badgeWarning: {
        borderColor: 'rgba(248, 113, 113, 0.6)',
        backgroundColor: 'rgba(127, 29, 29, 0.85)',
    },
    badgeCaution: {
        borderColor: '#FBBF24',
        backgroundColor: 'rgba(146, 64, 14, 0.88)',
        borderWidth: 2,
    },
    badgeDanger: {
        borderColor: '#EF4444',
        backgroundColor: 'rgba(153, 27, 27, 0.92)',
        borderWidth: 2,
    },
    alertText: {
        color: '#FEF08A',
        fontSize: 16,
        fontWeight: '700',
        textAlign: 'center',
        marginTop: 8,
        letterSpacing: 0.5,
    },
});
