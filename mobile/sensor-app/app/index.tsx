import { Text, View, StyleSheet, Switch, Platform } from 'react-native';
import { router } from 'expo-router';
import { useCallback } from 'react';
import NavButton from '@/components/Nav_Button';
import { useBackgroundMode } from '../contexts/BackgroundContext';

export default function Index() {
    const { enabled, supported, enable, disable } = useBackgroundMode();

    // Toggle handler: enabling walks the user through permissions, starts the
    // foreground service, and navigates to the detection screen so the Camera
    // component mounts immediately. If the user bails on a required permission
    // we do not navigate (enable() returns false). Disabling stops the service.
    const onToggle = useCallback(
        async (next: boolean) => {
            if (next) {
                const ok = await enable();
                if (ok) {
                    router.push('/sensors/drowsiness');
                }
            } else {
                await disable();
            }
        },
        [enable, disable],
    );

    return (
        <View style={styles.container}>
            <View style={styles.header}>
                <Text style={styles.title}>vigilare</Text>
                <Text style={styles.subtitle}>Drowsiness Detection</Text>
            </View>

            <View style={styles.nav}>
                <NavButton label='Drowsiness Detection' href='/sensors/drowsiness' primary />
                <NavButton label='Camera' href='/sensors/camera' />
                <NavButton label='Accelerometer' href='/sensors/accelerometer' />
                <NavButton label='Debug Gallery' href='/sensors/debug-gallery' />
            </View>

            <View style={styles.bgCard}>
                <View style={styles.bgRow}>
                    <View style={styles.bgTextWrap}>
                        <Text style={styles.bgTitle}>Background Detection</Text>
                        <Text style={styles.bgSubtitle}>
                            {Platform.OS !== 'android'
                                ? 'Android only'
                                : enabled
                                  ? 'Running — keeps monitoring with screen off'
                                  : 'Off — toggle to monitor in background'}
                        </Text>
                    </View>
                    <Switch
                        value={enabled}
                        onValueChange={onToggle}
                        disabled={!supported}
                        trackColor={{ false: '#1E293B', true: '#1D4ED8' }}
                        thumbColor={enabled ? '#93C5FD' : '#64748B'}
                    />
                </View>
                {enabled && (
                    <Text style={styles.bgHint}>
                        Stops automatically if you close the app.
                    </Text>
                )}
            </View>

            <Text style={styles.version}>v2.1</Text>
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
    bgCard: {
        width: '100%',
        marginTop: 24,
        paddingHorizontal: 16,
    },
    bgRow: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(30, 58, 138, 0.25)',
        borderRadius: 12,
        borderWidth: 1.5,
        borderColor: 'rgba(147, 197, 253, 0.25)',
        paddingVertical: 14,
        paddingHorizontal: 16,
        gap: 12,
    },
    bgTextWrap: {
        flex: 1,
    },
    bgTitle: {
        fontSize: 15,
        fontWeight: '700',
        color: '#93C5FD',
        letterSpacing: 0.4,
    },
    bgSubtitle: {
        fontSize: 12,
        color: '#94A3B8',
        marginTop: 2,
        fontWeight: '500',
    },
    bgHint: {
        color: '#475569',
        fontSize: 11,
        marginTop: 8,
        textAlign: 'center',
        fontStyle: 'italic',
    },
    version: {
        position: 'absolute',
        bottom: 32,
        color: '#334155',
        fontSize: 12,
        fontWeight: '500',
    },
});
