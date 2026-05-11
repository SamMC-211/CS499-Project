// Local Expo config plugin that installs the background-detection foreground
// service plus its React Native bridge, AND the MainActivity changes required
// to keep VisionCamera's Camera component alive while the app is backgrounded
// or the device is locked.
//
// Why MainActivity has to be patched: VisionCamera releases the camera session
// whenever the host Activity drops below RESUMED. A foreground service keeps
// the *process* alive but does not keep the Activity RESUMED. Two techniques
// are wired in here:
//
//   * Picture-in-Picture — when the user presses home, MainActivity.onUserLeaveHint
//     auto-calls enterPictureInPictureMode(...). The Activity stays RESUMED in
//     a small floating window, so the camera + frame processor keep running
//     while the user uses another app (e.g. a nav app).
//
//   * Show-when-locked + keep-screen-on + turn-screen-on — toggled from JS via
//     BackgroundDetectionModule.start(). FLAG_KEEP_SCREEN_ON prevents the
//     display from auto-sleeping (VisionCamera needs the display alive to
//     keep the camera session open), setShowWhenLocked keeps the Activity
//     visible behind the keyguard, setTurnScreenOn wakes the display when
//     a critical alert needs the driver's attention.
//
// This plugin also adds permissions (FOREGROUND_SERVICE, *_CAMERA, WAKE_LOCK,
// POST_NOTIFICATIONS, SYSTEM_ALERT_WINDOW) and a <service android:foregroundServiceType="camera">
// entry. All native source is materialised here so `npx expo prebuild --clean`
// stays reproducible — nothing in android/ has to be hand-edited.

const fs = require('fs');
const path = require('path');

const {
    withAndroidManifest,
    withDangerousMod,
    withMainActivity,
    withMainApplication,
    AndroidConfig,
} = require('@expo/config-plugins');

const PACKAGE_NAME = 'com.anonymous.sensorapp';

const REQUIRED_PERMISSIONS = [
    'android.permission.FOREGROUND_SERVICE',
    'android.permission.FOREGROUND_SERVICE_CAMERA',
    'android.permission.WAKE_LOCK',
    'android.permission.POST_NOTIFICATIONS',
    'android.permission.SYSTEM_ALERT_WINDOW',
];

function withPermissions(config) {
    return withAndroidManifest(config, (mod) => {
        const manifest = mod.modResults.manifest;
        manifest['uses-permission'] = manifest['uses-permission'] || [];

        for (const perm of REQUIRED_PERMISSIONS) {
            const already = manifest['uses-permission'].some(
                (p) => p.$ && p.$['android:name'] === perm,
            );
            if (!already) {
                manifest['uses-permission'].push({ $: { 'android:name': perm } });
            }
        }
        return mod;
    });
}

function withServiceEntry(config) {
    return withAndroidManifest(config, (mod) => {
        const app = AndroidConfig.Manifest.getMainApplicationOrThrow(mod.modResults);
        app.service = app.service || [];

        const exists = app.service.some(
            (s) =>
                s.$ &&
                s.$['android:name'] === `.BackgroundDetectionService`,
        );

        if (!exists) {
            app.service.push({
                $: {
                    'android:name': '.BackgroundDetectionService',
                    'android:enabled': 'true',
                    'android:exported': 'false',
                    'android:foregroundServiceType': 'camera',
                },
            });
        }
        return mod;
    });
}

// Add PiP + resizeable + the extra configChanges flags needed for PiP to work
// without recreating the Activity on enter/exit.
function withMainActivityManifestAttrs(config) {
    return withAndroidManifest(config, (mod) => {
        const app = AndroidConfig.Manifest.getMainApplicationOrThrow(mod.modResults);
        const main = (app.activity || []).find(
            (a) => a.$ && a.$['android:name'] === '.MainActivity',
        );
        if (!main) return mod;

        main.$['android:supportsPictureInPicture'] = 'true';
        main.$['android:resizeableActivity'] = 'true';

        const existing = main.$['android:configChanges'] || '';
        const needed = ['screenSize', 'smallestScreenSize', 'screenLayout'];
        const parts = existing.split('|').filter(Boolean);
        for (const n of needed) {
            if (!parts.includes(n)) parts.push(n);
        }
        main.$['android:configChanges'] = parts.join('|');
        return mod;
    });
}

const SERVICE_KT = `package ${PACKAGE_NAME}

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat

/**
 * Foreground service that keeps the JS engine alive while the screen is off
 * or the app is backgrounded. Holds a partial WakeLock so the CPU does not
 * sleep, and posts a persistent notification so Android does not kill the
 * process.
 *
 * The service does NOT own the camera — VisionCamera's frame processor still
 * runs on the Activity. What keeps the camera alive across home / lock is a
 * combination of (a) this service preventing Doze/standby, and (b) the
 * window flags + Picture-in-Picture flip applied to MainActivity from JS via
 * BackgroundDetectionModule.start().
 */
class BackgroundDetectionService : Service() {

    companion object {
        const val CHANNEL_ID = "vigilare_background_detection"
        const val NOTIFICATION_ID = 4012
        const val ACTION_START = "vigilare.action.START_BACKGROUND_DETECTION"
        const val ACTION_STOP = "vigilare.action.STOP_BACKGROUND_DETECTION"
        const val WAKE_TAG = "vigilare:BackgroundDetectionWakeLock"
    }

    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopForegroundCompat()
                stopSelf()
                return START_NOT_STICKY
            }
            else -> {
                startForeground(NOTIFICATION_ID, buildNotification())
                acquireWakeLock()
            }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        releaseWakeLock()
        super.onDestroy()
    }

    override fun onTaskRemoved(rootIntent: Intent?) {
        // User swiped the app off recents -> stop detection.
        stopForegroundCompat()
        stopSelf()
        super.onTaskRemoved(rootIntent)
    }

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            if (nm.getNotificationChannel(CHANNEL_ID) == null) {
                val channel = NotificationChannel(
                    CHANNEL_ID,
                    "Drowsiness Detection",
                    NotificationManager.IMPORTANCE_LOW
                ).apply {
                    description = "Active while background drowsiness detection is running."
                    setShowBadge(false)
                }
                nm.createNotificationChannel(channel)
            }
        }
    }

    private fun buildNotification(): Notification {
        val launchIntent = packageManager.getLaunchIntentForPackage(packageName)
        val contentIntent = launchIntent?.let {
            PendingIntent.getActivity(
                this,
                0,
                it,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
        }

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .setContentTitle("vigilare — Background Detection")
            .setContentText("Monitoring driver alertness")
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setContentIntent(contentIntent)
            .build()
    }

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) return
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        val wl = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKE_TAG)
        wl.setReferenceCounted(false)
        wl.acquire()
        wakeLock = wl
    }

    private fun releaseWakeLock() {
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
    }

    @Suppress("DEPRECATION")
    private fun stopForegroundCompat() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            stopForeground(true)
        }
    }
}
`;

const MODULE_KT = `package ${PACKAGE_NAME}

import android.Manifest
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.provider.Settings
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.modules.core.PermissionAwareActivity
import com.facebook.react.modules.core.PermissionListener

/**
 * JS-facing bridge for the background detection feature.
 *
 *   start() / stop() — flip the foreground service + activity keep-alive
 *                      flags (FLAG_KEEP_SCREEN_ON, setShowWhenLocked, etc.)
 *                      and arm MainActivity.backgroundDetectionActive so
 *                      onUserLeaveHint can enter Picture-in-Picture mode.
 *
 *   requestNotificationPermission() — Android 13+ runtime POST_NOTIFICATIONS.
 *   hasOverlayPermission() / openOverlayPermissionSettings() — SYSTEM_ALERT_WINDOW.
 *   isIgnoringBatteryOptimizations() / requestIgnoreBatteryOptimizations() —
 *       prompts the user to whitelist the app from Doze.
 */
class BackgroundDetectionModule(reactContext: ReactApplicationContext)
    : ReactContextBaseJavaModule(reactContext) {

    companion object {
        private const val NOTIFICATION_REQUEST_CODE = 4011
    }

    // System-overlay state — only ever touched on the main thread.
    private var overlayView: View? = null
    private var overlayAnimator: ObjectAnimator? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun getName(): String = "BackgroundDetection"

    @ReactMethod
    fun start(promise: Promise) {
        try {
            // Tell MainActivity to enter PiP the next time the user backgrounds the app.
            MainActivity.backgroundDetectionActive = true

            reactApplicationContext.currentActivity?.runOnUiThread { applyKeepAliveFlags(true) }

            val ctx = reactApplicationContext
            val intent = Intent(ctx, BackgroundDetectionService::class.java).apply {
                action = BackgroundDetectionService.ACTION_START
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                ctx.startForegroundService(intent)
            } else {
                ctx.startService(intent)
            }
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("BG_START_FAILED", e)
        }
    }

    @ReactMethod
    fun stop(promise: Promise) {
        try {
            MainActivity.backgroundDetectionActive = false

            reactApplicationContext.currentActivity?.runOnUiThread { applyKeepAliveFlags(false) }

            // Tear down the alert overlay if one was left up (defensive — JS
            // should already have called hideAlertOverlay on tier drop, but on
            // a forced stop we cannot rely on that).
            mainHandler.post { removeOverlayInternal() }

            val ctx = reactApplicationContext
            val intent = Intent(ctx, BackgroundDetectionService::class.java).apply {
                action = BackgroundDetectionService.ACTION_STOP
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                ctx.startForegroundService(intent)
            } else {
                ctx.startService(intent)
            }
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("BG_STOP_FAILED", e)
        }
    }

    private fun applyKeepAliveFlags(enabled: Boolean) {
        val activity = reactApplicationContext.currentActivity ?: return
        val window = activity.window
        if (enabled) {
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                activity.setShowWhenLocked(true)
                activity.setTurnScreenOn(true)
            } else {
                @Suppress("DEPRECATION")
                window.addFlags(
                    WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                        WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON or
                        WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD
                )
            }
        } else {
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
                activity.setShowWhenLocked(false)
                activity.setTurnScreenOn(false)
            } else {
                @Suppress("DEPRECATION")
                window.clearFlags(
                    WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                        WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON or
                        WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD
                )
            }
        }
    }

    @ReactMethod
    fun hasNotificationPermission(promise: Promise) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            promise.resolve(true); return
        }
        val granted = ContextCompat.checkSelfPermission(
            reactApplicationContext,
            Manifest.permission.POST_NOTIFICATIONS
        ) == PackageManager.PERMISSION_GRANTED
        promise.resolve(granted)
    }

    @ReactMethod
    fun requestNotificationPermission(promise: Promise) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            promise.resolve(true); return
        }
        val activity = reactApplicationContext.currentActivity as? PermissionAwareActivity
        if (activity == null) {
            promise.resolve(false); return
        }
        try {
            val listener = object : PermissionListener {
                override fun onRequestPermissionsResult(
                    requestCode: Int,
                    permissions: Array<String>,
                    grantResults: IntArray
                ): Boolean {
                    val granted = grantResults.isNotEmpty() &&
                        grantResults[0] == PackageManager.PERMISSION_GRANTED
                    promise.resolve(granted)
                    return true
                }
            }
            activity.requestPermissions(
                arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                NOTIFICATION_REQUEST_CODE,
                listener
            )
        } catch (e: Exception) {
            promise.reject("PERM_REQ_FAILED", e)
        }
    }

    @ReactMethod
    fun hasOverlayPermission(promise: Promise) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            promise.resolve(Settings.canDrawOverlays(reactApplicationContext))
        } else {
            promise.resolve(true)
        }
    }

    @ReactMethod
    fun openOverlayPermissionSettings(promise: Promise) {
        try {
            val intent = Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:" + reactApplicationContext.packageName)
            ).apply { addFlags(Intent.FLAG_ACTIVITY_NEW_TASK) }
            reactApplicationContext.startActivity(intent)
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("OPEN_SETTINGS_FAILED", e)
        }
    }

    @ReactMethod
    fun isIgnoringBatteryOptimizations(promise: Promise) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val pm = reactApplicationContext
                .getSystemService(Context.POWER_SERVICE) as PowerManager
            promise.resolve(
                pm.isIgnoringBatteryOptimizations(reactApplicationContext.packageName)
            )
        } else {
            promise.resolve(true)
        }
    }

    @ReactMethod
    @SuppressWarnings("BatteryLife")
    fun requestIgnoreBatteryOptimizations(promise: Promise) {
        try {
            val intent = Intent(
                Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                Uri.parse("package:" + reactApplicationContext.packageName)
            ).apply { addFlags(Intent.FLAG_ACTIVITY_NEW_TASK) }
            reactApplicationContext.startActivity(intent)
            promise.resolve(true)
        } catch (e: Exception) {
            promise.reject("OPEN_SETTINGS_FAILED", e)
        }
    }

    // ------------------------------------------------------------------
    // System overlay (tier-3 alert that draws over other apps)
    // ------------------------------------------------------------------
    //
    // Builds a full-screen view with a pulsing red layer + warning text
    // and attaches it to WindowManager using TYPE_APPLICATION_OVERLAY.
    // The view is touch-passthrough (FLAG_NOT_TOUCHABLE) so the user can
    // still operate whatever app is underneath — the alert is purely
    // visual + the EAS audio playing in JS is what grabs attention.
    //
    // showAlertOverlay() requires SYSTEM_ALERT_WINDOW; if the user hasn't
    // granted it, the method resolves false and the JS layer falls back
    // to the in-app overlay only.

    @ReactMethod
    fun showAlertOverlay(promise: Promise) {
        val ctx = reactApplicationContext
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
            && !Settings.canDrawOverlays(ctx)
        ) {
            promise.resolve(false); return
        }
        mainHandler.post {
            try {
                if (overlayView != null) {
                    promise.resolve(true); return@post
                }
                val view = buildOverlayView(ctx)
                val wm = ctx.getSystemService(Context.WINDOW_SERVICE) as WindowManager
                val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                } else {
                    @Suppress("DEPRECATION")
                    WindowManager.LayoutParams.TYPE_PHONE
                }
                val params = WindowManager.LayoutParams(
                    WindowManager.LayoutParams.MATCH_PARENT,
                    WindowManager.LayoutParams.MATCH_PARENT,
                    type,
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                        WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
                        WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                        WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                        WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS or
                        WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                    PixelFormat.TRANSLUCENT
                )
                params.gravity = Gravity.TOP or Gravity.START
                wm.addView(view, params)
                overlayView = view
                startPulseAnimation(view)
                promise.resolve(true)
            } catch (e: Exception) {
                overlayView = null
                promise.reject("OVERLAY_SHOW_FAILED", e)
            }
        }
    }

    @ReactMethod
    fun hideAlertOverlay(promise: Promise) {
        mainHandler.post {
            try {
                removeOverlayInternal()
                promise.resolve(true)
            } catch (e: Exception) {
                promise.resolve(false)
            }
        }
    }

    // Called from stop() too; safe to call when nothing is showing.
    private fun removeOverlayInternal() {
        try {
            overlayAnimator?.cancel()
        } catch (_: Exception) {
        }
        overlayAnimator = null

        val v = overlayView ?: return
        try {
            val wm = reactApplicationContext
                .getSystemService(Context.WINDOW_SERVICE) as WindowManager
            wm.removeView(v)
        } catch (_: Exception) {
            // Already removed / never attached.
        }
        overlayView = null
    }

    private fun buildOverlayView(ctx: Context): View {
        // Outer frame = touch-passthrough black-ish backdrop holding the
        // pulsing red layer (index 0) plus the centred warning text.
        val frame = FrameLayout(ctx).apply {
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        }

        val redLayer = View(ctx).apply {
            setBackgroundColor(Color.parseColor("#DC2626"))
            alpha = 0.55f
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        }
        frame.addView(redLayer)

        // Vertically centred warning text stack.
        val textStack = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(48, 48, 48, 48)
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            ).apply {
                gravity = Gravity.CENTER
            }
        }

        val title = TextView(ctx).apply {
            text = "DRIVER ALERT"
            setTextColor(Color.parseColor("#FEF08A"))
            textSize = 42f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            gravity = Gravity.CENTER
            setShadowLayer(8f, 0f, 2f, Color.BLACK)
        }
        val subtitle = TextView(ctx).apply {
            text = "Drowsiness detected — pull over now"
            setTextColor(Color.WHITE)
            textSize = 22f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            gravity = Gravity.CENTER
            setShadowLayer(6f, 0f, 2f, Color.BLACK)
            val lp = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            lp.topMargin = 24
            layoutParams = lp
        }

        textStack.addView(title)
        textStack.addView(subtitle)
        frame.addView(textStack)

        // Stash the red layer so the animator can find it by id-less lookup.
        frame.setTag(redLayer)
        return frame
    }

    private fun startPulseAnimation(view: View) {
        val red = (view as? FrameLayout)?.getTag() as? View ?: return
        val anim = ObjectAnimator.ofFloat(red, "alpha", 0.55f, 0.92f).apply {
            duration = 600
            repeatMode = ValueAnimator.REVERSE
            repeatCount = ValueAnimator.INFINITE
        }
        anim.start()
        overlayAnimator = anim
    }
}
`;

const PACKAGE_KT = `package ${PACKAGE_NAME}

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager

class BackgroundDetectionPackage : ReactPackage {
    override fun createNativeModules(
        reactContext: ReactApplicationContext
    ): List<NativeModule> = listOf(BackgroundDetectionModule(reactContext))

    override fun createViewManagers(
        reactContext: ReactApplicationContext
    ): List<ViewManager<*, *>> = emptyList()
}
`;

function withNativeSources(config) {
    return withDangerousMod(config, [
        'android',
        async (mod) => {
            const projectRoot = mod.modRequest.platformProjectRoot;
            const packagePath = PACKAGE_NAME.split('.').join('/');
            const sourceDir = path.join(
                projectRoot,
                'app',
                'src',
                'main',
                'java',
                packagePath,
            );

            if (!fs.existsSync(sourceDir)) {
                fs.mkdirSync(sourceDir, { recursive: true });
            }

            fs.writeFileSync(
                path.join(sourceDir, 'BackgroundDetectionService.kt'),
                SERVICE_KT,
            );
            fs.writeFileSync(
                path.join(sourceDir, 'BackgroundDetectionModule.kt'),
                MODULE_KT,
            );
            fs.writeFileSync(
                path.join(sourceDir, 'BackgroundDetectionPackage.kt'),
                PACKAGE_KT,
            );

            return mod;
        },
    ]);
}

function withPackageRegistration(config) {
    return withMainApplication(config, (mod) => {
        let src = mod.modResults.contents;

        const importLine = `import ${PACKAGE_NAME}.BackgroundDetectionPackage`;
        if (!src.includes(importLine)) {
            src = src.replace(
                /(package [^\n]+\n)/,
                `$1\n${importLine}\n`,
            );
        }

        if (!src.includes('BackgroundDetectionPackage()')) {
            src = src.replace(
                /PackageList\(this\)\.packages\.apply\s*\{([\s\S]*?)\}/,
                (match, body) => {
                    if (body.includes('BackgroundDetectionPackage()')) return match;
                    const injected =
                        `${body}\n              add(BackgroundDetectionPackage())\n            `;
                    return `PackageList(this).packages.apply {${injected}}`;
                },
            );
        }

        mod.modResults.contents = src;
        return mod;
    });
}

// Patch MainActivity.kt so it (a) exposes a static flag we can flip from the
// native module and (b) enters PiP when the user backgrounds the app.
function withMainActivityPipPatch(config) {
    return withMainActivity(config, (mod) => {
        let src = mod.modResults.contents;

        const pipImport = 'import android.app.PictureInPictureParams';
        if (!src.includes(pipImport)) {
            src = src.replace(
                /(package [^\n]+\n)/,
                `$1\nimport android.app.PictureInPictureParams\nimport android.util.Rational\n`,
            );
        }

        if (!src.includes('backgroundDetectionActive')) {
            src = src.replace(
                /class MainActivity\s*:\s*ReactActivity\s*\(\)\s*\{/,
                `class MainActivity : ReactActivity() {
  // Flipped by BackgroundDetectionModule.start()/.stop(). When true and the
  // user backgrounds the app, we auto-enter Picture-in-Picture so VisionCamera
  // keeps capturing frames.
  companion object {
    @JvmStatic var backgroundDetectionActive: Boolean = false
  }
`,
            );
        }

        if (!src.includes('onUserLeaveHint')) {
            // Insert before the final closing brace of the class file.
            src = src.replace(
                /}\s*$/,
                `  override fun onUserLeaveHint() {
    super.onUserLeaveHint()
    if (backgroundDetectionActive && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
      try {
        val params = PictureInPictureParams.Builder()
          .setAspectRatio(Rational(9, 16))
          .build()
        enterPictureInPictureMode(params)
      } catch (e: Exception) {
        // PiP unsupported / blocked on this device — fall through silently.
      }
    }
  }
}
`,
            );
        }

        mod.modResults.contents = src;
        return mod;
    });
}

module.exports = function withBackgroundDetection(config) {
    config = withPermissions(config);
    config = withServiceEntry(config);
    config = withMainActivityManifestAttrs(config);
    config = withNativeSources(config);
    config = withPackageRegistration(config);
    config = withMainActivityPipPatch(config);
    return config;
};
