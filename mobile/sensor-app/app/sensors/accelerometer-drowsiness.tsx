import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
    ActivityIndicator,
    Pressable,
    StyleSheet,
    Text,
    View,
} from 'react-native';
import { router } from 'expo-router';
import { Accelerometer } from 'expo-sensors';
import { EventSubscription } from 'expo-modules-core';

// ---------------------------------------------------------------------------
// Model bundling
// ---------------------------------------------------------------------------
// This screen runs the accelerometer-based drowsiness CNN once the trained
// artifacts have been produced and copied into assets/ml/. Until then, this
// screen still works as a live-sensor + buffer-fill harness so the UI can be
// tested before training is finished.
//
// To enable inference after training:
//   1. From mobile/sensor-app/model_training/accelerometer/, run
//        python train_model.py all
//   2. Copy:
//        artifacts/accel_drowsiness_cnn.tflite -> assets/ml/
//        artifacts/normalization.json          -> assets/ml/accel_normalization.json
//        artifacts/labels.json                 -> assets/ml/accel_labels.json
//   3. Set MODEL_AVAILABLE = true and uncomment the three require() lines.
// ---------------------------------------------------------------------------
const MODEL_AVAILABLE = false;
// import { useTensorflowModel } from 'react-native-fast-tflite';
// const MODEL_REQUIRE = require('../../assets/ml/accel_drowsiness_cnn.tflite');
// const NORM_REQUIRE = require('../../assets/ml/accel_normalization.json');
// const LABELS_REQUIRE = require('../../assets/ml/accel_labels.json');

// Must match config.py in the training pipeline.
const SAMPLE_INTERVAL_MS = 100; // 10 Hz
const WINDOW_SAMPLES = 300; // 30 s
const CHANNELS = 3;
const INFERENCE_PERIOD_MS = 2000; // run inference every 2 s once the buffer is full
const CALIBRATION_SAMPLES = 20; // ~2 s of stationary readings to estimate gravity
const DROWSY_THRESHOLD = 0.5; // sigmoid midpoint; tune after evaluating CV results

const DEFAULT_LABELS = ['Drowsy', 'Normal'] as const;

type Vec3 = { x: number; y: number; z: number };

type UiPrediction = {
    label: string;
    score: number;
    ready: boolean;
};

export default function AccelerometerDrowsinessScreen() {
    // Live accelerometer state
    const [current, setCurrent] = useState<Vec3>({ x: 0, y: 0, z: 0 });
    const [bufferFill, setBufferFill] = useState(0);
    const [calibrated, setCalibrated] = useState(false);
    const [calibrating, setCalibrating] = useState(false);
    const [prediction, setPrediction] = useState<UiPrediction>({
        label: MODEL_AVAILABLE ? 'Initializing...' : 'Model not bundled',
        score: 0,
        ready: false,
    });

    // Subscription handle
    const subRef = useRef<EventSubscription | null>(null);

    // Ring buffer of gravity-removed samples: (WINDOW_SAMPLES, CHANNELS) flattened.
    const bufferRef = useRef<Float32Array>(
        new Float32Array(WINDOW_SAMPLES * CHANNELS),
    );
    const writeIdxRef = useRef(0);
    const filledRef = useRef(0);

    // Gravity estimate built during calibration; subtracted from every sample.
    const gravityRef = useRef<Vec3>({ x: 0, y: 0, z: 0 });
    const calibBufferRef = useRef<Vec3[]>([]);

    // Model + normalization stats (only loaded when MODEL_AVAILABLE).
    // const modelPlugin = MODEL_AVAILABLE ? useTensorflowModel(MODEL_REQUIRE) : undefined;
    // const norm = MODEL_AVAILABLE ? NORM_REQUIRE : undefined;
    // const labels = MODEL_AVAILABLE ? LABELS_REQUIRE.class_names : DEFAULT_LABELS;
    const labels = DEFAULT_LABELS;

    // Subscribe at 10 Hz on mount.
    useEffect(() => {
        Accelerometer.setUpdateInterval(SAMPLE_INTERVAL_MS);
        subRef.current = Accelerometer.addListener((sample) => {
            handleSample(sample);
        });
        return () => {
            subRef.current?.remove();
            subRef.current = null;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Inference timer: kicks off every INFERENCE_PERIOD_MS once buffer is full.
    useEffect(() => {
        if (!MODEL_AVAILABLE) return;
        const id = setInterval(runInference, INFERENCE_PERIOD_MS);
        return () => clearInterval(id);
    }, []);

    function handleSample(s: Vec3) {
        setCurrent(s);

        if (calibrating) {
            calibBufferRef.current.push(s);
            if (calibBufferRef.current.length >= CALIBRATION_SAMPLES) {
                const n = calibBufferRef.current.length;
                const sum = calibBufferRef.current.reduce(
                    (acc, v) => ({
                        x: acc.x + v.x,
                        y: acc.y + v.y,
                        z: acc.z + v.z,
                    }),
                    { x: 0, y: 0, z: 0 },
                );
                gravityRef.current = {
                    x: sum.x / n,
                    y: sum.y / n,
                    z: sum.z / n,
                };
                calibBufferRef.current = [];
                setCalibrating(false);
                setCalibrated(true);
                // Reset the rolling buffer so post-calibration samples populate cleanly.
                writeIdxRef.current = 0;
                filledRef.current = 0;
                setBufferFill(0);
            }
            return;
        }

        if (!calibrated) return;

        // Gravity-remove and push into ring buffer.
        const g = gravityRef.current;
        const ax = s.x - g.x;
        const ay = s.y - g.y;
        const az = s.z - g.z;

        const idx = writeIdxRef.current;
        const buf = bufferRef.current;
        buf[idx * CHANNELS + 0] = ax;
        buf[idx * CHANNELS + 1] = ay;
        buf[idx * CHANNELS + 2] = az;

        writeIdxRef.current = (idx + 1) % WINDOW_SAMPLES;
        if (filledRef.current < WINDOW_SAMPLES) {
            filledRef.current += 1;
            setBufferFill(filledRef.current);
        }
    }

    function runInference() {
        if (!MODEL_AVAILABLE) return;
        if (filledRef.current < WINDOW_SAMPLES) return;
        // Placeholder for the real inference path -- left commented so the file
        // still type-checks without the bundled model. When you flip
        // MODEL_AVAILABLE on, replace this body with:
        //
        //   const ordered = orderedWindow();              // (WINDOW_SAMPLES * CHANNELS) Float32Array
        //   const normalized = zscore(ordered, norm.mean, norm.std);
        //   const out = modelPlugin?.model?.runSync([normalized]) as unknown[];
        //   const score = Number((out?.[0] as number[])?.[0] ?? 0);
        //   const idx = score >= DROWSY_THRESHOLD ? 1 : 0;
        //   setPrediction({ label: labels[idx], score, ready: true });
    }

    function onCalibrate() {
        calibBufferRef.current = [];
        gravityRef.current = { x: 0, y: 0, z: 0 };
        writeIdxRef.current = 0;
        filledRef.current = 0;
        setBufferFill(0);
        setCalibrated(false);
        setCalibrating(true);
    }

    const fillPct = useMemo(
        () => Math.round((bufferFill / WINDOW_SAMPLES) * 100),
        [bufferFill],
    );

    // Tier coloring mirrors the camera screen's styling for visual consistency.
    const drowsy =
        prediction.ready && prediction.label === labels[0]; // labels[0] = "Drowsy"

    return (
        <View style={styles.root}>
            <View style={styles.topBar}>
                <Pressable
                    onPress={() => router.replace('/sensors/drowsiness')}
                    android_ripple={{ color: 'rgba(147, 197, 253, 0.3)' }}
                    style={styles.topButton}
                >
                    <Text style={styles.topButtonText}>‹ Camera Detection</Text>
                </Pressable>
                <Text style={styles.topTitle}>Accelerometer</Text>
                <View style={styles.topSpacer} />
            </View>

            <View style={styles.body}>
                <View style={[styles.badge, drowsy && styles.badgeDanger]}>
                    <Text style={styles.badgeTitle}>Driver State</Text>
                    <Text
                        style={[
                            styles.badgeLabel,
                            drowsy ? styles.labelDrowsy : styles.labelActive,
                        ]}
                    >
                        {prediction.ready
                            ? prediction.label
                            : MODEL_AVAILABLE
                              ? calibrated
                                  ? `Collecting (${fillPct}%)`
                                  : 'Awaiting Calibration'
                              : 'Model Not Bundled'}
                    </Text>
                    <View style={styles.badgeDivider} />
                    <Text style={styles.badgeScore}>
                        {prediction.ready
                            ? prediction.score.toFixed(3)
                            : '—'}
                    </Text>
                    <Text style={styles.badgeMeta}>
                        Buffer: {bufferFill} / {WINDOW_SAMPLES} samples ({fillPct}%)
                    </Text>
                </View>

                <View style={styles.card}>
                    <Text style={styles.cardTitle}>Live Sensor (g)</Text>
                    <View style={styles.row}>
                        <Text style={styles.axisLabel}>x</Text>
                        <Text style={styles.axisValue}>{current.x.toFixed(4)}</Text>
                    </View>
                    <View style={styles.row}>
                        <Text style={styles.axisLabel}>y</Text>
                        <Text style={styles.axisValue}>{current.y.toFixed(4)}</Text>
                    </View>
                    <View style={styles.row}>
                        <Text style={styles.axisLabel}>z</Text>
                        <Text style={styles.axisValue}>{current.z.toFixed(4)}</Text>
                    </View>
                    {calibrated && (
                        <>
                            <View style={styles.divider} />
                            <Text style={styles.cardSub}>
                                Gravity offset: x={gravityRef.current.x.toFixed(3)}, y=
                                {gravityRef.current.y.toFixed(3)}, z=
                                {gravityRef.current.z.toFixed(3)}
                            </Text>
                        </>
                    )}
                </View>

                <Pressable
                    onPress={onCalibrate}
                    disabled={calibrating}
                    android_ripple={{ color: 'rgba(147, 197, 253, 0.3)' }}
                    style={[
                        styles.calibrateBtn,
                        calibrating && styles.calibrateBtnActive,
                    ]}
                >
                    {calibrating ? (
                        <View style={styles.calibrateRow}>
                            <ActivityIndicator color="#93C5FD" />
                            <Text style={styles.calibrateText}>
                                Hold still… ({calibBufferRef.current.length}/{CALIBRATION_SAMPLES})
                            </Text>
                        </View>
                    ) : (
                        <Text style={styles.calibrateText}>
                            {calibrated ? 'Recalibrate' : 'Calibrate (hold phone still)'}
                        </Text>
                    )}
                </Pressable>

                {!MODEL_AVAILABLE && (
                    <Text style={styles.hint}>
                        First iteration: TFLite model not yet bundled. Train it via{' '}
                        <Text style={styles.code}>
                            python train_model.py all
                        </Text>{' '}
                        and follow the instructions at the top of this file.
                    </Text>
                )}
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    root: { flex: 1, backgroundColor: '#0F172A' },
    topBar: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingTop: 48,
        paddingHorizontal: 12,
        paddingBottom: 12,
        borderBottomWidth: 1,
        borderBottomColor: 'rgba(147, 197, 253, 0.12)',
    },
    topButton: {
        paddingVertical: 8,
        paddingHorizontal: 12,
        borderRadius: 8,
        backgroundColor: 'rgba(30, 58, 138, 0.35)',
        borderWidth: 1,
        borderColor: 'rgba(147, 197, 253, 0.25)',
    },
    topButtonText: { color: '#93C5FD', fontWeight: '600', fontSize: 13 },
    topTitle: {
        flex: 1,
        textAlign: 'center',
        color: '#BFDBFE',
        fontSize: 15,
        fontWeight: '700',
        letterSpacing: 1.5,
        textTransform: 'uppercase',
    },
    topSpacer: { width: 130 }, // matches topButton width to center the title
    body: { flex: 1, padding: 16, gap: 16 },
    badge: {
        backgroundColor: 'rgba(15, 23, 42, 0.82)',
        padding: 16,
        borderRadius: 16,
        borderWidth: 1,
        borderColor: 'rgba(147, 197, 253, 0.2)',
    },
    badgeDanger: {
        borderColor: '#EF4444',
        backgroundColor: 'rgba(153, 27, 27, 0.92)',
        borderWidth: 2,
    },
    badgeTitle: {
        color: '#94A3B8',
        fontSize: 11,
        fontWeight: '600',
        letterSpacing: 1.2,
        textTransform: 'uppercase',
        marginBottom: 4,
    },
    badgeLabel: { fontSize: 22, fontWeight: '700' },
    labelDrowsy: { color: '#F87171' },
    labelActive: { color: '#93C5FD' },
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
    card: {
        backgroundColor: 'rgba(15, 23, 42, 0.82)',
        padding: 16,
        borderRadius: 16,
        borderWidth: 1,
        borderColor: 'rgba(147, 197, 253, 0.2)',
    },
    cardTitle: {
        color: '#94A3B8',
        fontSize: 11,
        fontWeight: '600',
        letterSpacing: 1.2,
        textTransform: 'uppercase',
        marginBottom: 12,
    },
    row: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginVertical: 2,
    },
    axisLabel: {
        color: '#64748B',
        fontSize: 14,
        fontWeight: '600',
        width: 24,
    },
    axisValue: {
        color: '#CBD5E1',
        fontSize: 14,
        fontWeight: '400',
        fontVariant: ['tabular-nums'],
    },
    divider: {
        height: 1,
        backgroundColor: 'rgba(147, 197, 253, 0.15)',
        marginVertical: 10,
    },
    cardSub: {
        color: '#64748B',
        fontSize: 11,
        fontVariant: ['tabular-nums'],
    },
    calibrateBtn: {
        paddingVertical: 14,
        paddingHorizontal: 16,
        borderRadius: 12,
        backgroundColor: 'rgba(30, 58, 138, 0.35)',
        borderWidth: 1,
        borderColor: 'rgba(147, 197, 253, 0.25)',
        alignItems: 'center',
    },
    calibrateBtnActive: {
        backgroundColor: 'rgba(59, 130, 246, 0.25)',
        borderColor: '#3B82F6',
    },
    calibrateRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
    calibrateText: { color: '#93C5FD', fontWeight: '600', fontSize: 14 },
    hint: {
        color: '#64748B',
        fontSize: 12,
        textAlign: 'center',
        fontStyle: 'italic',
        marginTop: 4,
        lineHeight: 18,
    },
    code: {
        fontFamily: 'monospace',
        color: '#93C5FD',
    },
});
