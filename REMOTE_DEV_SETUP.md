# Remote Development Setup: Tailscale + SSH + VS Code

Goal: SSH into this Windows machine from your Chromebook (Linux) using VS Code, so you can build and edit code remotely.

---

## Step 1: Create Your Own Tailscale Account

You need your own tailnet — not your friend's.

1. Go to [tailscale.com](https://tailscale.com) and sign up (free tier is plenty, supports up to 3 devices)
2. On **this Windows machine**, open Tailscale in the system tray
3. Click your account name > **Log out**
4. Log back in with **your** new account

Your Windows machine will get a Tailscale IP in the `100.x.x.x` range. Find it:
- Tailscale tray icon > click it — your IP is shown, or
- Open a terminal and run: `tailscale ip -4`

Write that IP down — you'll need it later.

---

## Step 2: Enable OpenSSH Server on Windows

Open **PowerShell as Administrator** and run these three commands in order:

```powershell
# Install the OpenSSH Server feature
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# Start the SSH service now
Start-Service sshd

# Make it start automatically on boot
Set-Service -Name sshd -StartupType Automatic
```

Verify it worked:
```powershell
Get-Service sshd
# Should show: Status = Running
```

Windows Firewall should allow SSH automatically after install, but if you have issues, run:
```powershell
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

---

## Step 3: Keep This Computer Awake

SSH dies the moment Windows sleeps. Before you leave:

1. Open **Settings > System > Power & Sleep**
2. Set both "Screen" and "Sleep" to **Never** (while plugged in)

Or via PowerShell:
```powershell
# Disable sleep while plugged in
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
```

---

## Step 4: Set Up Tailscale on Your Chromebook

In your Chromebook's **Linux terminal**:

```bash
# Install Tailscale on Linux (Crostini)
curl -fsSL https://tailscale.com/install.sh | sh

# Start it and authenticate
sudo tailscale up
```

A browser link will appear — open it and log in with the **same Tailscale account** you used on Windows.

Once connected, verify both machines see each other:
```bash
tailscale status
# Should list your Windows machine with its 100.x.x.x IP
```

---

## Step 5: Test the SSH Connection

From the Chromebook Linux terminal:

```bash
ssh yourusername@100.x.x.x
# Replace with your Windows username and Tailscale IP
# e.g.: ssh samsu@100.64.12.34
```

It will ask for your Windows account password. If it connects, you're good.

> **Note on username:** use your Windows username exactly as it appears. If your username has a space, wrap it in quotes or use the short version.

---

## Step 6: Install VS Code on Chromebook and Connect Remotely

**Install VS Code in Linux:**
```bash
# Download and install the .deb package
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list
sudo apt update && sudo apt install code
```

**Install the Remote-SSH extension:**
```bash
code --install-extension ms-vscode-remote.remote-ssh
```

Or open VS Code > Extensions sidebar > search "Remote - SSH" > Install.

**Connect to Windows:**

1. Press `Ctrl+Shift+P` in VS Code
2. Type: `Remote-SSH: Connect to Host`
3. Enter: `yourusername@100.x.x.x`
4. Enter your Windows password when prompted
5. VS Code will install a small server on Windows and connect

Once connected, use **File > Open Folder** and navigate to `C:/Users/samsu/CS499-Project`.

---

## Step 7: Optional — Save the SSH Config

So you don't have to type the IP every time:

Edit `~/.ssh/config` on your Chromebook:

```
Host my-desktop
    HostName 100.x.x.x
    User samsu
```

Now you can connect with just:
```bash
ssh my-desktop
```

And in VS Code Remote-SSH, `my-desktop` will appear as a saved host.

---

## Quick Reference Checklist (Before You Leave)

- [ ] Tailscale running on Windows and logged into your account
- [ ] `sshd` service is running (`Get-Service sshd`)
- [ ] Sleep disabled on Windows
- [ ] Computer plugged in
- [ ] Test SSH from another device before you walk out the door

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` | `sshd` isn't running — run `Start-Service sshd` on Windows |
| `Connection timed out` | Tailscale isn't running on one of the machines |
| `Permission denied` | Wrong username or password; check Windows username exactly |
| VS Code can't find `code` command on Chromebook | Open VS Code manually, then `Ctrl+Shift+P` > "Shell Command: Install 'code' command" |
| Windows went to sleep | You forgot Step 3 — remote in via Tailscale web console or ask someone to wake it |

---

## Installing the App on Your Phone Without a USB Cable

Since your phone won't be physically connected to the remote desktop, you have three options for getting builds onto it.

---

### Option 1: EAS Build (Expo Cloud Builds)

Expo builds the APK on their servers and gives you a download link. No phone connection needed at all.

**One-time setup:**
```bash
npm install -g eas-cli
eas login
eas build:configure
```

Add a `preview` profile to `eas.json` in your project root (create it if it doesn't exist):
```json
{
  "build": {
    "preview": {
      "android": {
        "buildType": "apk"
      }
    },
    "production": {
      "android": {
        "buildType": "apk"
      }
    }
  }
}
```

**Build and install:**
```bash
cd mobile/sensor-app

# Build a preview APK (development client)
eas build -p android --profile preview

# Build a production APK
eas build -p android --profile production
```

When the build finishes (~10-15 min), EAS gives you a QR code and a download URL. Open it on your phone, download the APK, and install it. You'll need to allow "Install unknown apps" in Android settings for your browser.

**Free tier:** ~30 builds/month. Best for milestone/release builds.

---

### Option 2: Local Build + Serve APK via HTTP (Best for Daily Dev)

Build the APK on the remote desktop, then serve it over HTTP so your phone can download it through a browser via Tailscale.

**Step 1 — Build the APK:**
```bash
cd mobile/sensor-app

# Debug APK (faster, includes dev tools)
cd android && ./gradlew assembleDebug

# Release APK
cd android && ./gradlew assembleRelease

# Or use Expo CLI (builds debug by default)
npx expo run:android
```

APK output locations:
```
android/app/build/outputs/apk/debug/app-debug.apk
android/app/build/outputs/apk/release/app-release.apk
```

**Step 2 — Serve the APK over HTTP:**
```bash
# Navigate to the APK output folder
cd android/app/build/outputs/apk/debug

# Start a simple HTTP server on port 8080
python -m http.server 8080
```

**Step 3 — Download on your phone:**

On your phone's browser, navigate to:
```
http://<your-desktop-tailscale-ip>:8080/app-debug.apk
```

Find your desktop's Tailscale IP with:
```bash
tailscale ip -4
```

The APK will download and prompt you to install. Enable "Install unknown apps" for your browser in **Android Settings > Apps > [your browser] > Install unknown apps**.

**To stop the server:** `Ctrl+C` in the terminal running the HTTP server.

---

### Option 3: ADB over WiFi via Tailscale

Connect your phone wirelessly to the desktop over Tailscale and push APKs directly with ADB. Also enables live reload during development.

**Prerequisites:**
- Android 11+ (supports wireless debugging natively)
- Both your phone and desktop connected to the same Tailscale network
- Android Developer Options enabled on your phone

**Step 1 — Enable Wireless Debugging on your phone:**
1. Settings > About Phone > tap "Build Number" 7 times to enable Developer Options
2. Settings > Developer Options > enable **Wireless Debugging**
3. Tap **Wireless Debugging** > tap **Pair device with pairing code**
4. Note the pairing IP:port and the 6-digit code shown on screen

**Step 2 — Pair ADB from the desktop:**
```bash
# Pair once using the pairing port (not the connection port)
adb pair <phone-ip>:<pairing-port>
# Enter the 6-digit code when prompted
```

**Step 3 — Connect ADB:**
```bash
# Use the connection IP:port shown on the Wireless Debugging screen
adb connect <phone-ip>:<connection-port>

# Verify connection
adb devices
# Should show your phone listed as "device"
```

**Step 4 — Install APK or run Expo:**
```bash
# Install a pre-built APK
adb install android/app/build/outputs/apk/debug/app-debug.apk

# Or run Expo with live reload (full dev workflow)
cd mobile/sensor-app
npx expo run:android
```

**Note:** Your phone's local IP may change if it reconnects to WiFi. If ADB loses connection, repeat Step 3. For a more stable setup, assign your phone a static IP on your router or use Tailscale on your phone and connect via its `100.x.x.x` Tailscale IP.

---

### Which Option Should You Use?

| Scenario | Best Option |
|---|---|
| Daily development and iteration | **Option 2** — local build + HTTP server |
| Release or milestone builds | **Option 1** — EAS Build |
| Live reload / active debugging session | **Option 3** — ADB over WiFi |
| No Android SDK installed on desktop | **Option 1** — EAS Build |

**Recommended daily workflow:**
1. Edit code on the remote desktop via SSH + VS Code
2. Build: `cd android && ./gradlew assembleDebug`
3. Serve: `python -m http.server 8080` from the APK output folder
4. Download and install on phone via `http://<tailscale-ip>:8080/app-debug.apk`
5. Test, iterate, repeat
