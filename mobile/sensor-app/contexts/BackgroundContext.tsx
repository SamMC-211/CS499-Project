import React, {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useRef,
    useState,
} from 'react';
import { Alert, Linking } from 'react-native';
import {
    isBackgroundSupported,
    startBackgroundService,
    stopBackgroundService,
    hasNotificationPermission,
    requestNotificationPermission,
    hasOverlayPermission,
    openOverlayPermissionSettings,
    isIgnoringBatteryOptimizations,
    requestIgnoreBatteryOptimizations,
} from '../native/BackgroundDetection';

type BackgroundContextValue = {
    enabled: boolean;
    supported: boolean;
    /** Walks the user through any missing permissions, then starts the service. */
    enable: () => Promise<boolean>;
    disable: () => Promise<void>;
    toggle: () => Promise<boolean>;
};

const BackgroundContext = createContext<BackgroundContextValue | undefined>(
    undefined,
);

function alertAsync(title: string, message: string): Promise<boolean> {
    return new Promise((resolve) => {
        Alert.alert(
            title,
            message,
            [
                { text: 'Not now', style: 'cancel', onPress: () => resolve(false) },
                { text: 'Continue', onPress: () => resolve(true) },
            ],
            { cancelable: false },
        );
    });
}

/**
 * Walk the user through the three permissions / settings we need for true
 * background operation:
 *
 *   1. POST_NOTIFICATIONS — required (Android 13+) so the foreground service
 *      notification is visible; without this the FGS still starts but Android
 *      will give it less priority and is more likely to kill it.
 *   2. SYSTEM_ALERT_WINDOW — needed so the tier-3 alert can show over other
 *      apps. Has no in-app prompt; we open the settings page.
 *   3. IGNORE_BATTERY_OPTIMIZATIONS — recommended; Doze otherwise throttles
 *      our service during long screen-off stretches.
 *
 * Returns false only if the user explicitly bails. Missing optional permissions
 * are non-fatal — we still let the service start, the user just gets a degraded
 * background experience.
 */
async function walkPermissions(): Promise<boolean> {
    // (1) Notification permission — explicit runtime prompt.
    if (!(await hasNotificationPermission())) {
        const ok = await alertAsync(
            'Notification permission',
            'Background Detection needs to post a small persistent notification so Android does not kill the app while you are driving. Tap Continue to grant it.',
        );
        if (!ok) return false;
        await requestNotificationPermission();
    }

    // (2) Overlay permission — sent to settings.
    if (!(await hasOverlayPermission())) {
        const ok = await alertAsync(
            'Display over other apps',
            "To show a drowsiness alert over your nav app, vigilare needs the 'Display over other apps' permission. You will be sent to the system settings; enable it for vigilare and return here.",
        );
        if (ok) {
            await openOverlayPermissionSettings();
            // We do not block here — user can flip the toggle again after returning.
        }
    }

    // (3) Battery optimization — sent to settings.
    if (!(await isIgnoringBatteryOptimizations())) {
        const ok = await alertAsync(
            'Battery optimization',
            'To keep monitoring with the screen off, Android needs to stop throttling vigilare in the background. Tap Continue and allow the exemption.',
        );
        if (ok) {
            await requestIgnoreBatteryOptimizations();
        }
    }

    return true;
}

export function BackgroundProvider({ children }: { children: React.ReactNode }) {
    const [enabled, setEnabled] = useState(false);
    const enabledRef = useRef(false);
    useEffect(() => {
        enabledRef.current = enabled;
    }, [enabled]);

    const enable = useCallback(async (): Promise<boolean> => {
        if (!isBackgroundSupported) {
            Alert.alert(
                'Not supported',
                'Background Detection is only available on Android.',
            );
            return false;
        }
        const ok = await walkPermissions();
        if (!ok) return false;

        const started = await startBackgroundService();
        setEnabled(true);
        if (!started) {
            console.warn('Background service did not start cleanly');
        }
        return true;
    }, []);

    const disable = useCallback(async () => {
        await stopBackgroundService();
        setEnabled(false);
    }, []);

    const toggle = useCallback(async (): Promise<boolean> => {
        if (enabledRef.current) {
            await disable();
            return false;
        } else {
            return await enable();
        }
    }, [enable, disable]);

    // If the provider unmounts (full app teardown / hot reload during dev),
    // tear down the service. Hard close via swipe-from-recents is also covered
    // natively by Service.onTaskRemoved.
    useEffect(() => {
        return () => {
            if (enabledRef.current) {
                void stopBackgroundService();
            }
        };
    }, []);

    const value = useMemo<BackgroundContextValue>(
        () => ({
            enabled,
            supported: isBackgroundSupported,
            enable,
            disable,
            toggle,
        }),
        [enabled, enable, disable, toggle],
    );

    // Silence unused-import warning for Linking — kept available for future
    // overlay-grant deep-link follow-ups.
    void Linking;

    return (
        <BackgroundContext.Provider value={value}>
            {children}
        </BackgroundContext.Provider>
    );
}

export function useBackgroundMode(): BackgroundContextValue {
    const ctx = useContext(BackgroundContext);
    if (!ctx) {
        throw new Error(
            'useBackgroundMode must be used within a BackgroundProvider',
        );
    }
    return ctx;
}
