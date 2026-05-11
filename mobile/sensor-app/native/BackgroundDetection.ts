// Thin wrapper around the BackgroundDetection native module installed by
// plugins/withBackgroundDetection.js. On platforms other than Android the
// module is absent — calls become no-ops / sensible defaults so JS code can
// stay platform-agnostic.

import { NativeModules, Platform } from 'react-native';

type BackgroundDetectionNative = {
    // Service / activity lifecycle
    start: () => Promise<boolean>;
    stop: () => Promise<boolean>;

    // POST_NOTIFICATIONS (Android 13+ runtime permission)
    hasNotificationPermission: () => Promise<boolean>;
    requestNotificationPermission: () => Promise<boolean>;

    // SYSTEM_ALERT_WINDOW — needed to draw a tier-3 alert over other apps.
    // Granting is via settings page; no in-app dialog exists for this permission.
    hasOverlayPermission: () => Promise<boolean>;
    openOverlayPermissionSettings: () => Promise<boolean>;

    // Battery optimization exemption — without this, Doze may throttle our
    // foreground service when the screen is off for extended periods.
    isIgnoringBatteryOptimizations: () => Promise<boolean>;
    requestIgnoreBatteryOptimizations: () => Promise<boolean>;

    // Tier-3 system overlay (TYPE_APPLICATION_OVERLAY) — draws over other apps.
    // showAlertOverlay() returns false if SYSTEM_ALERT_WINDOW is not granted.
    showAlertOverlay: () => Promise<boolean>;
    hideAlertOverlay: () => Promise<boolean>;
};

const native = NativeModules.BackgroundDetection as
    | BackgroundDetectionNative
    | undefined;

export const isBackgroundSupported = Platform.OS === 'android' && !!native;

function wrap<T>(
    fn: (() => Promise<T>) | undefined,
    fallback: T,
    label: string,
): () => Promise<T> {
    return async () => {
        if (!fn) return fallback;
        try {
            return await fn();
        } catch (e) {
            console.warn(`${label} failed`, e);
            return fallback;
        }
    };
}

export const startBackgroundService = wrap(
    native ? native.start.bind(native) : undefined,
    false,
    'startBackgroundService',
);

export const stopBackgroundService = wrap(
    native ? native.stop.bind(native) : undefined,
    false,
    'stopBackgroundService',
);

export const hasNotificationPermission = wrap(
    native ? native.hasNotificationPermission.bind(native) : undefined,
    true,
    'hasNotificationPermission',
);

export const requestNotificationPermission = wrap(
    native ? native.requestNotificationPermission.bind(native) : undefined,
    false,
    'requestNotificationPermission',
);

export const hasOverlayPermission = wrap(
    native ? native.hasOverlayPermission.bind(native) : undefined,
    true,
    'hasOverlayPermission',
);

export const openOverlayPermissionSettings = wrap(
    native ? native.openOverlayPermissionSettings.bind(native) : undefined,
    false,
    'openOverlayPermissionSettings',
);

export const isIgnoringBatteryOptimizations = wrap(
    native ? native.isIgnoringBatteryOptimizations.bind(native) : undefined,
    true,
    'isIgnoringBatteryOptimizations',
);

export const requestIgnoreBatteryOptimizations = wrap(
    native ? native.requestIgnoreBatteryOptimizations.bind(native) : undefined,
    false,
    'requestIgnoreBatteryOptimizations',
);

export const showAlertOverlay = wrap(
    native ? native.showAlertOverlay.bind(native) : undefined,
    false,
    'showAlertOverlay',
);

export const hideAlertOverlay = wrap(
    native ? native.hideAlertOverlay.bind(native) : undefined,
    false,
    'hideAlertOverlay',
);
