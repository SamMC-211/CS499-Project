import { View, Text, StyleSheet, Button, TouchableOpacity } from 'react-native';
import { useEffect, useState } from 'react';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';

export default function CameraScreen() {
    // const [hasPermission, setHasPermission] = useState<boolean | null>(null);
    // const [cameraRef, setCameraRef] = useState<Camera | null>(null);
    // const [type, setType] = useState(CameraType.back);

    const [facing, setFacing] = useState<CameraType>('back');
    const [permission, requestPermission] = useCameraPermissions();

    // //Request camera permissions
    // useEffect(() => {
    //     (async () => {
    //         const { status } = await Camera.requestCameraPermissionsAsync();
    //         setHasPermission(status === 'granted');
    //     })();
    // }, []);

    // if (hasPermission === null) {
    //     return <Text>Requestion camera permission...</Text>;
    // }

    // if (hasPermission === false) {
    //     return <Text>No access to camera</Text>;
    // }

    if (!permission) {
        return (
            <View style={styles.container}>
                <Text>Camera permissions still loading</Text>
            </View>
        );
    }

    if (!permission.granted) {
        return (
            <View style={styles.container}>
                <Text>We need your permission to show the camera</Text>
                <Button onPress={requestPermission} title='Grant Permission' />
            </View>
        );
    }

    function toggleCameraFacing() {
        setFacing((current) => (current === 'back' ? 'front' : 'back'));
    }

    return (
        // <View style={styles.container}>
        //     <Camera style={styles.camera} type={type} ref={(ref) => setCameraRef(ref)} />
        //     <Text>Camera Screen</Text>
        // </View>

        <View style={styles.container}>
            <CameraView style={styles.camera} facing={facing} />
            <View style={styles.buttonContainer}>
                <TouchableOpacity style={styles.button} onPress={toggleCameraFacing}>
                    <Text style={styles.text}>Flip Camera</Text>
                </TouchableOpacity>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
    },
    message: {
        textAlign: 'center',
        paddingBottom: 10,
    },
    camera: {
        flex: 1,
    },
    buttonContainer: {
        position: 'absolute',
        bottom: 64,
        flexDirection: 'row',
        backgroundColor: 'transparent',
        width: '100%',
        paddingHorizontal: 64,
    },
    button: {
        flex: 1,
        alignItems: 'center',
    },
    text: {
        fontSize: 24,
        fontWeight: 'bold',
        color: 'white',
    },
});
