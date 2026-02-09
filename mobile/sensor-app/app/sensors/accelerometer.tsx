import { useState, useEffect } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Accelerometer } from 'expo-sensors';
import { EventSubscription } from 'expo-modules-core';
import { green } from 'react-native-reanimated/lib/typescript/Colors';

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
                        backgroundColor: 'blue',
                    }}
                />
                <View
                    style={{
                        flexGrow: 1,
                        width: 200 * Math.abs(y),
                        backgroundColor: 'green',
                    }}
                />
                <View
                    style={{
                        flexGrow: 1,
                        width: 200 * Math.abs(z),
                        backgroundColor: 'red',
                    }}
                />
            </View>
            <Text style={styles.text}>Accelerometer: (in gs where 1g = 9.81 m/s^2)</Text>
            <Text style={styles.text}>x: {x}</Text>
            <Text style={styles.text}>y: {y}</Text>
            <Text style={styles.text}>z: {z}</Text>
            <View style={styles.buttonContainer}>
                <TouchableOpacity onPress={subscription ? _unsubscribe : _subscribe} style={styles.button}>
                    <Text>{subscription ? 'On' : 'Off'}</Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={_slow} style={[styles.button, styles.middleButton]}>
                    <Text>Slow</Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={_fast} style={styles.button}>
                    <Text>Fast</Text>
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
    },
    graph: {
        flexDirection: 'column',
        height: 200,
        width: 200,
        borderWidth: 1,
        borderColor: 'black',
        gap: 10,
    },
    text: {
        textAlign: 'center',
    },
    buttonContainer: {
        flexDirection: 'row',
        alignItems: 'stretch',
        marginTop: 15,
    },
    button: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#eee',
        padding: 10,
    },
    middleButton: {
        borderLeftWidth: 1,
        borderRightWidth: 1,
        borderColor: '#ccc',
    },
});
