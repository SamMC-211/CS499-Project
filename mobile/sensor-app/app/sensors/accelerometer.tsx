import { useState, useEffect } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Accelerometer } from 'expo-sensors';
import { EventSubscription } from 'expo-modules-core';
export default function AccelerometerScreen() {
    const [{ x, y, z }, setData] = useState({
        x: 0,
        y: 0,
        z: 0,
    });
    const [subscription, setSubscription] = useState<EventSubscription | null>(null);

    const _slow = () => Accelerometer.setUpdateInterval(1000);
    const _fast = () => Accelerometer.setUpdateInterval(100);

    const _subscribe = () => {
        setSubscription(Accelerometer.addListener(setData));
    };

    const _unsubscribe = () => {
        subscription && subscription.remove();
        setSubscription(null);
    };

    useEffect(() => {
        _subscribe();
        return () => _unsubscribe();
    }, []);

    return (
        <View style={styles.container}>
            <View style={styles.graph}>
                <View
                    style={{
                        flexGrow: 1,
                        width: 200 * Math.abs(x),
                        backgroundColor: '#60A5FA',
                    }}
                />
                <View
                    style={{
                        flexGrow: 1,
                        width: 200 * Math.abs(y),
                        backgroundColor: '#93C5FD',
                    }}
                />
                <View
                    style={{
                        flexGrow: 1,
                        width: 200 * Math.abs(z),
                        backgroundColor: '#BFDBFE',
                    }}
                />
            </View>
            <Text style={styles.text}>Accelerometer: (in gs where 1g = 9.81 m/s^2)</Text>
            <Text style={styles.text}>x: {x}</Text>
            <Text style={styles.text}>y: {y}</Text>
            <Text style={styles.text}>z: {z}</Text>
            <View style={styles.buttonContainer}>
                <TouchableOpacity onPress={subscription ? _unsubscribe : _subscribe} style={styles.button}>
                    <Text style={styles.btnText}>{subscription ? 'On' : 'Off'}</Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={_slow} style={[styles.button, styles.middleButton]}>
                    <Text style={styles.btnText}>Slow</Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={_fast} style={styles.button}>
                    <Text style={styles.btnText}>Fast</Text>
                </TouchableOpacity>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        paddingHorizontal: 20,
        backgroundColor: '#0F172A',
    },
    graph: {
        flexDirection: 'column',
        height: 200,
        width: 200,
        borderWidth: 1,
        borderColor: 'rgba(147, 197, 253, 0.2)',
        borderRadius: 8,
        gap: 10,
        overflow: 'hidden',
    },
    text: {
        textAlign: 'center',
        color: '#CBD5E1',
        fontSize: 13,
    },
    buttonContainer: {
        flexDirection: 'row',
        alignItems: 'stretch',
        marginTop: 15,
        borderRadius: 8,
        overflow: 'hidden',
    },
    button: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'rgba(30, 58, 138, 0.3)',
        padding: 12,
    },
    middleButton: {
        borderLeftWidth: 1,
        borderRightWidth: 1,
        borderColor: 'rgba(147, 197, 253, 0.15)',
    },
    btnText: {
        color: '#93C5FD',
        fontWeight: '500',
    },
});
