import { Text, View, StyleSheet } from 'react-native';
import NavButton from '@/components/Nav_Button';

export default function Index() {
    return (
        <View style={styles.container}>
            <View style={styles.header}>
                <Text style={styles.title}>vigilare</Text>
                <Text style={styles.subtitle}>Drowsiness Detection</Text>
            </View>

            <View style={styles.nav}>
                {/* You can use this but best practice is to use LINKS for UI Navigation */}
                {/* <Pressable onPress={() => router.push('/sensors/camera')}>
                    <Text>Camera</Text>
                </Pressable> */}

                <NavButton label='Drowsiness Detection' href='/sensors/drowsiness' primary />
                <NavButton label='Camera' href='/sensors/camera' />
                <NavButton label='Accelerometer' href='/sensors/accelerometer' />
            </View>

            <Text style={styles.version}>v2.0</Text>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#0F172A',
        padding: 24,
    },
    header: {
        alignItems: 'center',
        marginBottom: 48,
    },
    title: {
        fontSize: 32,
        fontWeight: '300',
        color: '#BFDBFE',
        letterSpacing: 4,
    },
    subtitle: {
        fontSize: 13,
        color: '#64748B',
        marginTop: 8,
        fontWeight: '500',
        letterSpacing: 1,
    },
    nav: {
        width: '100%',
        gap: 12,
        paddingHorizontal: 16,
    },
    version: {
        position: 'absolute',
        bottom: 32,
        color: '#334155',
        fontSize: 12,
        fontWeight: '500',
    },
});
