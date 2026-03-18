import React, { useEffect, useMemo, useState, useRef } from 'react';
import {
    ActivityIndicator,
    StyleSheet,
    Text,
    View,
    Dimensions,
    useWindowDimensions,
} from 'react-native';
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

    const updatePredictionOnJs = useMemo(
        () =>
            Worklets.createRunOnJS((next: UiPrediction) => {
                setPrediction(next);
            }),
        [],
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

            // no face found, update UI or skip
            if (!faces || faces.length === 0) {
                updateLandmarks([]);
                updatePredictionOnJs({
                    label: 'No Face',
                    score: 0,
                    hasFace: false,
                });
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
            // pixelFormat 'rgb' and dataType 'float32' give us a Float32Array in [0, 255] range.
            // (The Rescaling layer inside the TFLite model handles the /255 normalization.)
            const input = resize(frame, {
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

                    const { x, y } = mapLandmarkToCrop(
                        pFrameX,
                        pFrameY,
                        aFaceX.value,
                        aFaceY.value,
                        aFaceW.value,
                        aFaceH.value,
                    );
                    stampDot(input, x, y, dotRadius, MODEL_INPUT_SIZE);

                    overlayPoints.push({
                        x: -(screenX * 1.4) + overlaySize.width + 125,
                        y: screenY * 1.25 + 5,
                        // {"height": 480, "width": 640} frame size
                    });
                }
            }

            updateLandmarks(overlayPoints);

            // DEBUG
            // if (frameCounter.value % DEBUG_SAVE_EVERY_N === 0) {
            //     saveInputOnJs(Array.from(input)); // input is Float32Array
            // }

            // Step 5: Run TFLite inference on the cropped, dot-stamped input buffer
            const outputs = model.runSync([input]) as unknown[];
            const out0 = outputs?.[0] as number[] | undefined;
            const score = Number(out0?.[0] ?? 0);

            // Item 2: The model's final layer is Dense(1, sigmoid).
            // Sigmoid output approaches 1.0 for class index 1 (Active Subjects)
            // and approaches 0.0 for class index 0 (Fatigue Subjects).
            // Threshold 0.5 is the natural decision boundary for a sigmoid classifier.
            const classIndex = score >= 0.5 ? 1 : 0;
            const label = labels.class_names[classIndex] ?? 'Unknown';

            updatePredictionOnJs({ label, score, hasFace: true });
        },
        [detectFaces, model],
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
                <ActivityIndicator />
                <Text style={styles.statusText}>
                    Loading TensorFlow Lite model...
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
                        <Circle key={i} cx={p.x} cy={p.y} r={1} fill="white" />
                    ))}
                    <Circle
                        cx={overlaySize.width}
                        cy={overlaySize.height}
                        r={50}
                        fill="green"
                    />
                    <Circle cx={0} cy={0} r={4} fill="red" />
                    {/* <Circle cx={frameSize.width} cy={frameSize.height} r={400} fill='red' /> */}
                    <Circle cx={0} cy={0} r={4} fill="blue" />
                    <Circle cx={0} cy={0} r={4} fill="blue" />
                    <Circle cx={0} cy={0} r={4} fill="blue" />
                    <Circle cx={0} cy={0} r={4} fill="blue" />
                </Svg>

                <View style={styles.badge}>
                    <Text style={styles.badgeTitle}>Driver State</Text>
                    <Text style={styles.badgeLabel}>{prediction.label}</Text>
                    <Text style={styles.badgeScore}>
                        score: {prediction.score.toFixed(3)}
                    </Text>
                    <Text style={styles.badgeMeta}>
                        {prediction.hasFace ? 'face: detected' : 'face: none'}
                    </Text>
                </View>
            </View>
        </>
    );
}

const styles = StyleSheet.create({
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
