import { Text, View, Pressable } from 'react-native';
import { Link, router } from 'expo-router';
import NavButton from '@/components/Nav_Button';

export default function Index() {
    return (
        <View
            style={{
                flex: 1,
                justifyContent: 'center',
                alignItems: 'center',
                gap: 10,
            }}
        >
            <Text>vigilāre</Text>
            <Text>Sensor Demos</Text>

            {/* You can use this but best practice is to use LINKS for UI Navigation */}
            {/* <Pressable onPress={() => router.push('/sensors/camera')}>
                <Text>Camera</Text>
            </Pressable> */}

            <NavButton label='Camera' href='/sensors/camera' />
            <NavButton label='Accelerometer' href='/sensors/accelerometer' />
        </View>
    );
}
