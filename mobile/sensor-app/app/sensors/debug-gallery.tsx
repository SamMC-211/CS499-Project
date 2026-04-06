import React, { useEffect, useState, useCallback } from 'react';
import {
    ActivityIndicator,
    FlatList,
    Image,
    Pressable,
    StyleSheet,
    Text,
    View,
} from 'react-native';
import { Directory, File, Paths } from 'expo-file-system';

type DebugEntry = {
    filename: string;
    timestamp: string; // human-readable
    epochMs: number;   // for sorting
};

/**
 * Convert a 145x145x3 float32 JSON payload into a BMP, write it to a temp
 * file, and return the file URI.  Writing to disk avoids two Hermes pitfalls:
 *   1. btoa() can choke on binary strings with bytes > 127
 *   2. React Native's <Image> on Android doesn't reliably handle BMP data URIs
 *
 * BMP is used instead of PNG because it requires no compression library —
 * just a fixed header followed by raw pixel rows (top-down, 4-byte padded).
 */
function buildBmpAndWriteFile(
    data: number[],
    width: number,
    height: number,
    filename: string,
): string {
    // Detect range like render_mobile_input.py: use 99th-percentile to ignore white dots
    const sorted = [...data].sort((a, b) => a - b);
    const p99 = sorted[Math.floor(sorted.length * 0.99)];
    const scale = p99 <= 1.5 ? 255.0 : 1.0;

    // BMP rows must be padded to a multiple of 4 bytes
    const rowBytes = width * 3;
    const paddedRowBytes = (rowBytes + 3) & ~3;
    const pixelDataSize = paddedRowBytes * height;
    const fileSize = 54 + pixelDataSize; // 14-byte file header + 40-byte DIB header

    const buf = new Uint8Array(fileSize);
    const view = new DataView(buf.buffer);

    // --- BMP File Header (14 bytes) ---
    buf[0] = 0x42; buf[1] = 0x4d; // 'BM'
    view.setUint32(2, fileSize, true);
    view.setUint32(10, 54, true); // pixel data offset

    // --- DIB Header (BITMAPINFOHEADER, 40 bytes) ---
    view.setUint32(14, 40, true);               // header size
    view.setInt32(18, width, true);              // width
    view.setInt32(22, -height, true);            // negative = top-down row order
    view.setUint16(26, 1, true);                 // color planes
    view.setUint16(28, 24, true);                // bits per pixel (24 = RGB)
    view.setUint32(30, 0, true);                 // no compression
    view.setUint32(34, pixelDataSize, true);     // pixel data size

    // --- Pixel data ---
    let offset = 54;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const srcIdx = (y * width + x) * 3;
            const r = Math.max(0, Math.min(255, Math.round(data[srcIdx] * scale)));
            const g = Math.max(0, Math.min(255, Math.round(data[srcIdx + 1] * scale)));
            const b = Math.max(0, Math.min(255, Math.round(data[srcIdx + 2] * scale)));
            // BMP stores pixels as BGR
            buf[offset++] = b;
            buf[offset++] = g;
            buf[offset++] = r;
        }
        // Pad row to 4-byte boundary
        const pad = paddedRowBytes - rowBytes;
        for (let p = 0; p < pad; p++) {
            buf[offset++] = 0;
        }
    }

    // Write BMP bytes to a cache file so <Image> can load it via file:// URI
    const bmpName = filename.replace('.json', '.bmp');
    const cacheDir = new Directory(Paths.cache, 'debug_previews');
    if (!cacheDir.exists) {
        cacheDir.create({ intermediates: true });
    }
    const bmpFile = new File(cacheDir, bmpName);
    bmpFile.write(buf);
    return bmpFile.uri;
}

/**
 * Extract a human-readable timestamp from filenames like "input_1711809432123.json"
 */
function filenameToTimestamp(filename: string): { display: string; epoch: number } {
    const match = filename.match(/input_(\d+)\.json/);
    if (!match) return { display: 'Unknown', epoch: 0 };
    const epoch = parseInt(match[1], 10);
    const date = new Date(epoch);
    return {
        display: date.toLocaleString(),
        epoch,
    };
}

export default function DebugGalleryScreen() {
    const [entries, setEntries] = useState<DebugEntry[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedFile, setSelectedFile] = useState<string | null>(null);
    const [imageUri, setImageUri] = useState<string | null>(null);
    const [rendering, setRendering] = useState(false);

    // Load the list of debug JSON files
    useEffect(() => {
        const debugDir = new Directory(Paths.document, 'debug_inputs');
        if (!debugDir.exists) {
            setLoading(false);
            return;
        }

        try {
            const files = debugDir.list();
            const jsonFiles = files
                .filter((f): f is File => f instanceof File && f.name.endsWith('.json'))
                .map((f) => {
                    const { display, epoch } = filenameToTimestamp(f.name);
                    return { filename: f.name, timestamp: display, epochMs: epoch };
                })
                .sort((a, b) => b.epochMs - a.epochMs); // newest first

            setEntries(jsonFiles);
        } catch (e) {
            console.warn('Failed to list debug_inputs:', e);
        }
        setLoading(false);
    }, []);

    // When a file is selected, read it and convert to an image
    const handleSelect = useCallback(async (filename: string) => {
        if (selectedFile === filename) {
            // Deselect
            setSelectedFile(null);
            setImageUri(null);
            return;
        }

        setSelectedFile(filename);
        setImageUri(null);
        setRendering(true);

        try {
            const debugDir = new Directory(Paths.document, 'debug_inputs');
            const file = new File(debugDir, filename);
            const text = await file.text();
            const payload = JSON.parse(text) as {
                width: number;
                height: number;
                channels: number;
                data: number[];
            };

            const uri = buildBmpAndWriteFile(payload.data, payload.width, payload.height, filename);
            setImageUri(uri);
        } catch (e) {
            console.warn('Failed to render debug frame:', e);
            setImageUri(null);
        }
        setRendering(false);
    }, [selectedFile]);

    if (loading) {
        return (
            <View style={styles.centered}>
                <ActivityIndicator color="#93C5FD" size="large" />
                <Text style={styles.statusText}>Loading debug frames...</Text>
            </View>
        );
    }

    if (entries.length === 0) {
        return (
            <View style={styles.centered}>
                <Text style={styles.statusText}>No debug frames captured yet.</Text>
                <Text style={styles.hintText}>
                    Enable the Debug toggle on the Detection screen to start capturing frames.
                </Text>
            </View>
        );
    }

    return (
        <View style={styles.container}>
            {/* Image preview area */}
            <View style={styles.previewArea}>
                {rendering && (
                    <ActivityIndicator color="#93C5FD" size="large" />
                )}
                {imageUri && !rendering && (
                    <Image
                        source={{ uri: imageUri }}
                        style={styles.previewImage}
                        resizeMode="contain"
                    />
                )}
                {!imageUri && !rendering && (
                    <Text style={styles.previewPlaceholder}>
                        Select a frame below to preview
                    </Text>
                )}
            </View>

            {/* Frame list */}
            <View style={styles.listContainer}>
                <Text style={styles.listHeader}>
                    {entries.length} captured frame{entries.length !== 1 ? 's' : ''}
                </Text>
                <FlatList
                    data={entries}
                    keyExtractor={(item) => item.filename}
                    renderItem={({ item }) => (
                        <Pressable
                            onPress={() => handleSelect(item.filename)}
                            android_ripple={{ color: 'rgba(147, 197, 253, 0.3)' }}
                            style={[
                                styles.listItem,
                                selectedFile === item.filename && styles.listItemSelected,
                            ]}
                        >
                            <Text style={[
                                styles.listItemText,
                                selectedFile === item.filename && styles.listItemTextSelected,
                            ]}>
                                {item.timestamp}
                            </Text>
                            <Text style={styles.listItemFilename}>{item.filename}</Text>
                        </Pressable>
                    )}
                />
            </View>
        </View>
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
    hintText: {
        color: '#64748B',
        marginTop: 8,
        textAlign: 'center',
        fontSize: 13,
    },
    container: {
        flex: 1,
        backgroundColor: '#0F172A',
    },
    previewArea: {
        height: 260,
        alignItems: 'center',
        justifyContent: 'center',
        borderBottomWidth: 1,
        borderBottomColor: 'rgba(147, 197, 253, 0.15)',
        padding: 16,
    },
    previewImage: {
        width: 220,
        height: 220,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: 'rgba(147, 197, 253, 0.3)',
    },
    previewPlaceholder: {
        color: '#475569',
        fontSize: 14,
        fontWeight: '500',
    },
    listContainer: {
        flex: 1,
        padding: 16,
    },
    listHeader: {
        color: '#94A3B8',
        fontSize: 11,
        fontWeight: '600',
        letterSpacing: 1.2,
        textTransform: 'uppercase',
        marginBottom: 12,
    },
    listItem: {
        paddingVertical: 12,
        paddingHorizontal: 16,
        backgroundColor: 'rgba(30, 58, 138, 0.25)',
        borderRadius: 10,
        marginBottom: 8,
        borderWidth: 1,
        borderColor: 'rgba(147, 197, 253, 0.15)',
    },
    listItemSelected: {
        backgroundColor: 'rgba(59, 130, 246, 0.25)',
        borderColor: '#3B82F6',
    },
    listItemText: {
        color: '#BFDBFE',
        fontSize: 14,
        fontWeight: '600',
    },
    listItemTextSelected: {
        color: '#93C5FD',
    },
    listItemFilename: {
        color: '#475569',
        fontSize: 11,
        marginTop: 2,
    },
});
